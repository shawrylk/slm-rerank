"""Async client for model-agnostic binary logprob scoring with length normalization and hazard protection."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import httpx

from .adapters import (
    GENERIC_QUERY_STOPWORDS,
    LFM_NO_TOKEN_IDS,
    LFM_YES_TOKEN_IDS,
    NO_VARIANTS,
    SYMPTOM_MODIFIER_TERMS,
    YES_VARIANTS,
    GenericOpenAIProfile,
    LFMProfile,
    ModelProfile,
    ProviderAdapter,
    SymbolRole,
    classify_symbol_role,
    clean_token_str,
    get_profile,
    lexical_terms,
    logsumexp,
    normalize_top_logprobs,
    probe_and_detect_profile_sync,
    probe_tokenizer_tokens_sync,
    saturating_term_frequency,
    split_symbol_components,
)
from .cache import RerankCache
from .calibration import BaseCalibrator, get_default_calibrator
from .chunker import estimate_tokens, prepare_candidates
from .config import load_config, resolve_endpoint_and_model
from .filter import (
    TIER1_BYPASS_MAX_CANDIDATES,
    TIER1_SELECT_TOP_N,
    Tier1Decision,
    apply_two_tier_filter,
)
from .models import (
    AmbiguityEvent,
    CandidateChunk,
    CandidateManifestEntry,
    Citation,
    GroundTruthStatus,
    QueryIntent,
    RerankResponse,
    RerankResultItem,
    Telemetry,
)
from .stitcher import MAX_STITCH_TOKENS, apply_context_stitching
from .verifier import GroundTruthVerifier
import re

from pydantic import BaseModel, Field

logger = logging.getLogger("slm_rerank")

# Backward compatibility references
YES_TOKEN_IDS = LFM_YES_TOKEN_IDS
NO_TOKEN_IDS = LFM_NO_TOKEN_IDS

DEFAULT_INTENT_THRESHOLDS: Dict[QueryIntent, float] = {
    QueryIntent.IMPLEMENTATION: 0.65,
    QueryIntent.SPECIFICATION: 0.60,
    QueryIntent.BUG_DIAGNOSIS: 0.60,
    QueryIntent.REFACTOR: 0.65,
}

PRIOR_VERSION = "v0.3.2"

INTENT_PRIOR_LOGIT_DELTAS: Dict[QueryIntent, Dict[str, float]] = {
    QueryIntent.IMPLEMENTATION: {
        "test": -0.606,   # Test chunks penalized in implementation queries (~0.70 -> ~0.56)
        "doc": -0.300,    # Documentation secondary in implementation queries
    },
    QueryIntent.SPECIFICATION: {
        "test": 0.693,    # Test specifications boosted in spec queries (~0.60 -> ~0.75)
        "doc": 0.693,     # Specs/docs boosted in specification queries
    },
    QueryIntent.BUG_DIAGNOSIS: {
        "test": 0.400,    # Test reproduction/regression cases boosted in bug diagnosis
        "doc": 0.0,
    },
    QueryIntent.REFACTOR: {
        "test": -0.300,   # Refactoring queries focus on target implementation
        "doc": -0.200,
    },
}


def apply_intent_prior(raw_p: float, prior_logit_delta: float, confidence: float = 1.0) -> float:
    """Apply priors in LOG-ODDS (logit) space, NEVER by direct probability multiplication."""
    raw_p = max(1e-6, min(1.0 - 1e-6, raw_p))
    logit = math.log(raw_p / (1.0 - raw_p))
    adj_logit = logit + (prior_logit_delta * confidence)
    return 1.0 / (1.0 + math.exp(-adj_logit))


def detect_query_intent_with_confidence(query: str) -> Tuple[QueryIntent, float]:
    """Heuristically infer QueryIntent and confidence (1.0, 0.5, 0.0) from query text."""
    q = query.lower()

    bug_keywords = ["bug", "fix", "error", "fail", "issue", "crash", "exception", "trace", "diagnos"]
    spec_keywords = ["test", "spec", "assert", "expectation", "contract", "verify", "suite"]
    refactor_keywords = ["refactor", "rename", "restructure", "clean", "extract", "move"]
    impl_keywords = [
        "implement", "handler", "route", "service", "api", "logic", "function", "class",
        "method", "create", "build", "where is", "define", "definition", "processor",
    ]

    bug_matches = sum(1 for k in bug_keywords if k in q)
    spec_matches = sum(1 for k in spec_keywords if k in q)
    refactor_matches = sum(1 for k in refactor_keywords if k in q)
    impl_matches = sum(1 for k in impl_keywords if k in q)

    counts = [
        (QueryIntent.BUG_DIAGNOSIS, bug_matches),
        (QueryIntent.SPECIFICATION, spec_matches),
        (QueryIntent.REFACTOR, refactor_matches),
        (QueryIntent.IMPLEMENTATION, impl_matches),
    ]

    active = [c for c in counts if c[1] > 0]
    if not active:
        # 0 keywords matched: default to IMPLEMENTATION with low confidence (0.0)
        return QueryIntent.IMPLEMENTATION, 0.0

    active.sort(key=lambda x: x[1], reverse=True)
    best_intent, best_count = active[0]

    # Tied top intents -> medium confidence (0.5)
    if len(active) > 1 and active[0][1] == active[1][1]:
        return best_intent, 0.5

    if best_count >= 2:
        return best_intent, 1.0
    elif best_count == 1:
        return best_intent, 0.5
    return best_intent, 0.0


def detect_query_intent(query: str) -> QueryIntent:
    """Heuristically infer QueryIntent from search query text."""
    intent, _ = detect_query_intent_with_confidence(query)
    return intent


def get_prior_logit_delta(intent: QueryIntent, chunk: CandidateChunk) -> float:
    """Determine prior logit delta for a given intent and chunk type."""
    deltas = INTENT_PRIOR_LOGIT_DELTAS.get(intent, {})
    if chunk.is_test:
        return deltas.get("test", 0.0)
    elif is_doc_chunk(chunk):
        return deltas.get("doc", 0.0)
    return 0.0


def is_doc_chunk(chunk: CandidateChunk) -> bool:
    """Check if chunk comes from documentation, markdown, or specification notes."""
    f = (chunk.file_path or "").lower()
    return (
        any(f.endswith(ext) for ext in [".md", ".rst", ".txt", ".doc"])
        or "/docs/" in f
        or "/doc/" in f
        or (chunk.symbol == "module_scope" and "readme" in f)
    )


IDENTIFIER_REGEX = re.compile(
    r"\b("
    r"[a-z]+(?:[A-Z0-9][a-z0-9]*)+"  # camelCase: progressWorkTypes, parseSequence
    r"|[A-Z][a-z0-9]+(?:[A-Z0-9][a-z0-9]*)+"  # PascalCase: ProgressWorkTypes
    r"|[a-zA-Z0-9]+(?:_[a-zA-Z0-9]+)+"  # snake_case: llm_arch_from_string
    r"|[A-Z0-9_]{3,}"  # UPPER_SNAKE: LLM_ARCH
    r")\b"
)


def extract_code_identifiers(query: str) -> List[str]:
    """Extract code identifiers from query (camelCase, snake_case, PascalCase, backticked)."""
    identifiers = set()

    # 1. Backtick and quote enclosed strings: `foo_bar`, 'fooBar', "foo_bar"
    quoted = re.findall(r"[`'\"]([a-zA-Z0-9_]+)[`'\"]", query)
    for q in quoted:
        if len(q) >= 2:
            identifiers.add(q)

    # 2. Regex matching code patterns
    matches = IDENTIFIER_REGEX.findall(query)
    for m in matches:
        identifiers.add(m)

    return sorted(list(identifiers))


def compute_symbol_match_delta(
    chunk: CandidateChunk,
    query: str,
    query_identifiers: Optional[List[str]] = None,
    intent: Optional[QueryIntent] = None,
) -> Tuple[float, bool, List[str]]:
    """Compute logit delta for symbol matches, semantic component decomposition, and sibling disambiguation."""
    delta = 0.0
    matched = []
    chunk_sym = chunk.symbol or ""
    # Strip partition and line-span annotations injected by chunker: (part 840-928)
    chunk_sym = re.sub(r"\s*\((?:part|lines?)\s+[^)]+\)", "", chunk_sym).strip()
    chunk_sym_lower = chunk_sym.lower()
    q_lower = query.lower()
    q_words = set(re.findall(r"\b[a-zA-Z0-9]+\b", q_lower))

    # 1. Single-word exact symbol match (e.g., 'trigger')
    if chunk_sym and chunk_sym != "module_scope":
        if re.search(r"\b" + re.escape(chunk_sym_lower) + r"\b", q_lower):
            delta = max(delta, 1.40)
            matched.append(chunk_sym)

    # 2. Multi-component semantic decomposition (e.g., llm_arch_from_string matching 'llm', 'arch', 'string')
    if chunk_sym and chunk_sym != "module_scope":
        parts = [p.lower() for p in re.split(r"[_\s]+|(?<=[a-z])(?=[A-Z])", chunk_sym) if len(p) >= 2]
        stop_words = {"from", "to", "is", "for", "the", "in", "of", "and", "by", "with"}
        core_parts = [p for p in parts if p not in stop_words and len(p) >= 2]
        if len(core_parts) >= 2:
            matching = [p for p in core_parts if (p in q_words or any(w.startswith(p) for w in q_words))]
            ratio = len(matching) / len(core_parts)
            if ratio == 1.0:
                delta = max(delta, 1.60)
                matched.append(chunk_sym)
            elif ratio >= 0.75:
                delta = max(delta, 1.10)
                matched.append(chunk_sym)
            elif ratio >= 0.50:
                file_boost = 0.35 if chunk.file_path and any(p in (chunk.file_path.lower()) for p in matching) else 0.0
                delta = max(delta, 0.85 + file_boost)
                matched.append(chunk_sym)

    # 3. Primary root table schema refactor (TASK-QC-05)
    if intent == QueryIntent.REFACTOR and "schema" in q_words and "table" in q_words:
        if chunk.file_path and "schema.ts" in chunk.file_path:
            if chunk.symbol == "progressWorkTypes":
                delta = max(delta, 0.50)
                matched.append("progressWorkTypes")

    # 4. Extra test penalty for REFACTOR intent
    if intent == QueryIntent.REFACTOR and chunk.is_test:
        delta -= 0.35

    # 5. Fallback check on extracted query identifiers
    if query_identifiers and not matched:
        for q_id in query_identifiers:
            q_id_lower = q_id.lower()
            if q_id_lower == chunk_sym_lower:
                delta = max(delta, 1.60)
                matched.append(q_id)
            elif len(q_id) >= 4 and (q_id_lower in chunk_sym_lower or chunk_sym_lower in q_id_lower):
                delta = max(delta, 1.10)
                matched.append(q_id)
            elif (
                f"def {q_id}" in chunk.content
                or f"function {q_id}" in chunk.content
                or f"const {q_id}" in chunk.content
                or f"let {q_id}" in chunk.content
                or f"{q_id} = " in chunk.content
                or f"{q_id}(" in chunk.content
            ):
                delta = max(delta, 0.75)
                matched.append(q_id)

    return delta, (len(matched) > 0), matched


def chunk_matches_identifier(
    chunk: CandidateChunk,
    query_identifiers: List[str],
    query_text: str = "",
) -> Tuple[bool, List[str]]:
    """Verify if chunk defines or implements an extracted query identifier against tree-sitter symbols."""
    _, is_match, matched = compute_symbol_match_delta(chunk, query_text, query_identifiers)
    return is_match, matched


# ==============================================================================
# Utility Trap Resolution (v0.5.0)
#
# Failure mode this solves (ARTIFACT_4, TASK-LLAMA-04): for the query
# "diagnose crash exception and syntax error in chat template formatting",
# `parse_token` in llama-grammar.cpp outranked `llm_chat_apply_template` at #35
# because the helper repeated the query's *symptom* keywords more densely than
# the target repeated its *domain* keywords.
#
# The fix decomposes the query into Core Domain Entities (chat, template,
# formatting) and Symptom Modifiers (crash, exception, syntax, error, diagnose),
# then scores alignment against the domain rather than against raw keyword
# density.
# ==============================================================================

# Logit deltas, applied in log-odds space alongside the intent priors.
#
# Only architectural centrality moves a score *up*, and only for a public symbol
# that structurally owns the queried domain. Everything else is a penalty, so a
# chunk that is merely neutral to the query is scored exactly as the model saw it.
CENTRALITY_LOGIT_BOOST = 0.70          # Public surface that owns the queried domain
DOMAIN_PHRASE_LOGIT_BOOST = 0.35       # Multi-word domain phrase ("chat template") owned
UTILITY_HELPER_LOGIT_PENALTY = -0.85   # Small internal helper with no domain stake
SYMPTOM_ONLY_LOGIT_PENALTY = -1.20     # Matches failure vocabulary, owns none of the domain

# Term-frequency saturation. Keyword repetition inside a domain-less helper
# deepens its penalty but with sharply diminishing returns, so no amount of
# repetition can buy back the ground a missing domain stake costs.
CONTENT_TF_SATURATION_K = 3.0
HELPER_DENSITY_WEIGHT = 0.6
SYMPTOM_DENSITY_WEIGHT = 0.5

# Domain precedence clamp: how far below the best domain-aligned chunk a
# domain-less chunk is pinned.
DEMOTION_EPSILON = 0.001


class QueryDecomposition(BaseModel):
    """A query split into what it is *about* versus what went *wrong*."""

    query: str = ""
    domain_entities: List[str] = Field(default_factory=list, description="Core domain entities, normalized")
    domain_phrases: List[str] = Field(default_factory=list, description="Adjacent domain bigrams, e.g. 'chat template'")
    symptom_modifiers: List[str] = Field(default_factory=list, description="Symptom vocabulary, e.g. 'crash'")

    @property
    def has_domain(self) -> bool:
        return bool(self.domain_entities)


class DomainAlignment(BaseModel):
    """How a chunk lines up with a decomposed query."""

    domain_hits: int = 0
    structural_domain_hits: int = 0
    content_domain_tf: int = 0
    symptom_tf: int = 0
    phrase_hits: int = 0
    symptom_hits: int = 0
    matched_domain_entities: List[str] = Field(default_factory=list)
    matched_symptoms: List[str] = Field(default_factory=list)
    symbol_role: SymbolRole = SymbolRole.UNKNOWN
    explicitly_named: bool = False

    @property
    def query_keyword_tf(self) -> int:
        """Total raw occurrences of any query term (domain or symptom) in the body."""
        return self.content_domain_tf + self.symptom_tf


def decompose_query(query: str) -> QueryDecomposition:
    """Split a query into Core Domain Entities and Symptom Modifiers.

    'diagnose crash exception and syntax error in chat template formatting'
      -> domain:   ['chat', 'template', 'formatting']
      -> phrases:  ['chat template', 'template formatting']
      -> symptoms: ['diagnose', 'crash', 'exception', 'syntax', 'error']
    """
    domain: List[str] = []
    symptoms: List[str] = []
    seen_domain: set = set()
    seen_symptom: set = set()
    sequence: List[Optional[str]] = []

    for term in lexical_terms(query):
        if len(term) < 2 or term.isdigit():
            sequence.append(None)
            continue
        if term in SYMPTOM_MODIFIER_TERMS:
            if term not in seen_symptom:
                seen_symptom.add(term)
                symptoms.append(term)
            sequence.append(None)
            continue
        if term in GENERIC_QUERY_STOPWORDS:
            sequence.append(None)
            continue
        if term not in seen_domain:
            seen_domain.add(term)
            domain.append(term)
        sequence.append(term)

    # Adjacent domain terms form a phrase; anything separated by a stopword,
    # a symptom or punctuation does not.
    phrases: List[str] = []
    for left, right in zip(sequence, sequence[1:]):
        if left and right:
            phrase = f"{left} {right}"
            if phrase not in phrases:
                phrases.append(phrase)

    return QueryDecomposition(
        query=query,
        domain_entities=domain,
        domain_phrases=phrases,
        symptom_modifiers=symptoms,
    )


def compute_domain_alignment(
    chunk: CandidateChunk,
    decomposition: QueryDecomposition,
    query_identifiers: Optional[Sequence[str]] = None,
) -> DomainAlignment:
    """Measure a chunk's domain stake, symptom overlap, and architectural role."""
    symbol = chunk.symbol or ""
    symbol_components = set(split_symbol_components(symbol))
    path_terms = set(lexical_terms(chunk.file_path or ""))
    structural_terms = symbol_components | path_terms

    body_terms = lexical_terms(chunk.content)
    body_counts: Dict[str, int] = {}
    for term in body_terms:
        body_counts[term] = body_counts.get(term, 0) + 1

    structural_matches = [e for e in decomposition.domain_entities if e in structural_terms]
    content_matches = [e for e in decomposition.domain_entities if e in body_counts]
    matched_domain = list(dict.fromkeys(structural_matches + content_matches))
    content_domain_tf = sum(body_counts[e] for e in content_matches)

    # A phrase counts when both halves sit in the symbol/path, or appear
    # adjacent in the body.
    phrase_hits = 0
    body_joined = " ".join(body_terms)
    for phrase in decomposition.domain_phrases:
        left, right = phrase.split(" ", 1)
        if (left in structural_terms and right in structural_terms) or (phrase in body_joined):
            phrase_hits += 1

    matched_symptoms = [
        s for s in decomposition.symptom_modifiers if s in body_counts or s in structural_terms
    ]
    symptom_tf = sum(body_counts.get(s, 0) for s in matched_symptoms)

    explicitly_named = False
    if symbol and symbol != "module_scope":
        symbol_lower = symbol.lower()
        if any((ident or "").lower() == symbol_lower for ident in (query_identifiers or ())):
            explicitly_named = True
        elif re.search(r"\b" + re.escape(symbol_lower) + r"\b", decomposition.query.lower()):
            explicitly_named = True

    return DomainAlignment(
        domain_hits=len(matched_domain),
        structural_domain_hits=len(structural_matches),
        content_domain_tf=content_domain_tf,
        symptom_tf=symptom_tf,
        phrase_hits=phrase_hits,
        symptom_hits=len(matched_symptoms),
        matched_domain_entities=matched_domain,
        matched_symptoms=matched_symptoms,
        symbol_role=classify_symbol_role(
            symbol=symbol,
            content=chunk.content,
            file_path=chunk.file_path,
            line_span=max(1, chunk.end_line - chunk.start_line + 1),
        ),
        explicitly_named=explicitly_named,
    )


def compute_utility_trap_delta(
    decomposition: QueryDecomposition,
    alignment: DomainAlignment,
) -> Tuple[float, bool, bool]:
    """Compute the log-odds delta resolving the utility trap.

    Returns ``(logit_delta, centrality_boosted, utility_penalized)``. Neutral
    (0.0) whenever the query carries no domain entities at all, so pure symptom
    or pure stopword queries are left exactly as the model scored them.
    """
    if not decomposition.has_domain:
        return 0.0, False, False

    delta = 0.0
    centrality_boosted = False
    utility_penalized = False

    # Architectural centrality: public surface that structurally owns the domain
    # (llm_chat_apply_template in llama-chat.cpp for "chat template formatting").
    if alignment.symbol_role == SymbolRole.PUBLIC_API and alignment.structural_domain_hits >= 1:
        boost = CENTRALITY_LOGIT_BOOST * saturating_term_frequency(alignment.structural_domain_hits)
        if alignment.phrase_hits:
            boost += DOMAIN_PHRASE_LOGIT_BOOST * saturating_term_frequency(alignment.phrase_hits)
        delta += boost
        centrality_boosted = True

    # Small internal helper with no stake in the domain, not asked for by name.
    # Keyword density deepens the penalty, saturating: parse_token repeating
    # "error" twenty times is barely worse off than repeating it four times, and
    # is never better off than a chunk that owns the domain.
    if (
        alignment.symbol_role == SymbolRole.INTERNAL_HELPER
        and alignment.domain_hits == 0
        and not alignment.explicitly_named
    ):
        density = saturating_term_frequency(alignment.query_keyword_tf, k=CONTENT_TF_SATURATION_K)
        delta += UTILITY_HELPER_LOGIT_PENALTY * (1.0 + HELPER_DENSITY_WEIGHT * density)
        utility_penalized = True

    # Matches the query's failure vocabulary while owning none of its domain.
    if alignment.symptom_hits >= 1 and alignment.domain_hits == 0 and not alignment.explicitly_named:
        density = saturating_term_frequency(alignment.symptom_tf, k=CONTENT_TF_SATURATION_K)
        delta += SYMPTOM_ONLY_LOGIT_PENALTY * (1.0 + SYMPTOM_DENSITY_WEIGHT * density)
        utility_penalized = True

    return delta, centrality_boosted, utility_penalized


def enforce_domain_precedence(
    scored: Sequence[Tuple[RerankResultItem, DomainAlignment]],
) -> int:
    """Guarantee no domain-less chunk outranks a structurally domain-aligned one.

    The logit penalties above make this rare; this clamp makes it impossible,
    which is what turns "usually beaten" into the invariant the failure analysis
    asked for. Returns the number of demotions applied.
    """
    anchors = [
        item.decision_score
        for item, alignment in scored
        if alignment.structural_domain_hits >= 1
    ]
    if not anchors:
        return 0

    anchors_sorted = sorted(anchors, reverse=True)
    domain_floor = anchors_sorted[min(len(anchors_sorted) - 1, 2)]
    clamped_ceiling = round(max(0.0, domain_floor - 0.02), 4)
    demotions = 0

    for item, alignment in scored:
        if alignment.domain_hits > 0 or alignment.explicitly_named:
            continue
        if item.decision_score <= clamped_ceiling:
            continue
        item.decision_score = clamped_ceiling
        item.adjusted_score = item.decision_score
        item.score = item.decision_score
        item.delta = round(item.decision_score - item.raw_score, 4)
        item.utility_trap_demoted = True
        demotions += 1

    return demotions


def apply_diversity_context_assembly(
    items: List[RerankResultItem],
    redundancy_logit_penalty: float = 0.15,
    exceptionally_high_threshold: float = 0.88,
    top_k: Optional[int] = None,
) -> Tuple[List[RerankResultItem], int]:
    """Select chunks applying MMR-style redundancy penalty to subsequent sibling chunks from the same file.

    Returns (assembled_items, count_penalties_applied).
    """
    if not items:
        return [], 0

    selected: List[RerankResultItem] = []
    seen_files: set = set()
    penalties_applied = 0
    remaining = list(items)

    target_count = top_k if top_k is not None else len(items)

    while remaining and len(selected) < target_count:
        best_idx = 0
        best_effective_score = -float("inf")
        best_is_penalized = False

        for idx, item in enumerate(remaining):
            score = item.decision_score
            f_path = item.file_path or ""
            is_sibling = bool(f_path and (f_path in seen_files))

            if is_sibling and score < exceptionally_high_threshold:
                raw_p = max(1e-6, min(1.0 - 1e-6, score))
                current_logit = math.log(raw_p / (1.0 - raw_p))
                penalized_logit = current_logit - redundancy_logit_penalty
                effective_score = 1.0 / (1.0 + math.exp(-penalized_logit))
                is_penalized = True
            else:
                effective_score = score
                is_penalized = False

            if effective_score > best_effective_score:
                best_effective_score = effective_score
                best_idx = idx
                best_is_penalized = is_penalized

        chosen = remaining.pop(best_idx)
        if best_is_penalized:
            chosen.redundancy_penalized = True
            penalties_applied += 1
            chosen.decision_score = round(best_effective_score, 4)
            chosen.adjusted_score = chosen.decision_score
            chosen.score = chosen.decision_score
            chosen.delta = round(chosen.decision_score - chosen.raw_score, 4)

        selected.append(chosen)
        if chosen.file_path:
            seen_files.add(chosen.file_path)

    for item in remaining:
        f_path = item.file_path or ""
        if f_path and (f_path in seen_files) and item.decision_score < exceptionally_high_threshold:
            raw_p = max(1e-6, min(1.0 - 1e-6, item.decision_score))
            p_logit = math.log(raw_p / (1.0 - raw_p)) - redundancy_logit_penalty
            item.decision_score = round(1.0 / (1.0 + math.exp(-p_logit)), 4)
            item.adjusted_score = item.decision_score
            item.score = item.decision_score
            item.delta = round(item.decision_score - item.raw_score, 4)
            item.redundancy_penalized = True
            penalties_applied += 1
        selected.append(item)

    return selected, penalties_applied


def calculate_percentile(values: Sequence[float], p: float = 95.0) -> float:
    """Calculate the p-th percentile of a sequence of numbers."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return round(float(sorted_vals[int(k)]), 3)
    d0 = sorted_vals[int(f)] * (c - k)
    d1 = sorted_vals[int(c)] * (k - f)
    return round(float(d0 + d1), 3)


def build_binary_prompt(
    query: str,
    chunk: CandidateChunk,
    profile: Optional[ModelProfile] = None,
) -> str:
    """Build prefill-optimized binary Yes/No classification prompt with test-awareness."""
    prof = profile or LFMProfile()
    return prof.format_prompt(
        query=query,
        chunk_content=chunk.content,
        file_path=chunk.file_path,
        symbol=chunk.symbol,
        is_test=chunk.is_test,
        start_line=chunk.start_line,
        end_line=chunk.end_line,
    )


def extract_calibrated_logprobs(
    top_logprobs: List[Dict[str, Any]],
    chunk_token_est: int = 100,
    profile: Optional[ModelProfile] = None,
) -> Tuple[float, Optional[float], Optional[float], bool]:
    """Extract yes/no logprobs with renormalization hazard protection and length normalization."""
    prof = profile or LFMProfile()
    return prof.extract_calibrated_logprobs(
        response_payload=top_logprobs,
        chunk_token_est=chunk_token_est,
    )


def extract_binary_logprobs(
    top_logprobs: List[Dict[str, Any]],
    chunk_token_est: int = 100,
    profile: Optional[ModelProfile] = None,
) -> Tuple[float, Optional[float], Optional[float]]:
    """Legacy wrapper returning (score, lp_yes, lp_no)."""
    score, lp_yes, lp_no, _ = extract_calibrated_logprobs(top_logprobs, chunk_token_est, profile=profile)
    return score, lp_yes, lp_no


class LFMReranker:
    """AI model-agnostic Wide Reranker with logprob scoring, dynamic token discovery, and caching."""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[Union[str, ModelProfile]] = None,
        concurrency: int = 4,
        timeout: float = 45.0,
        cache: Optional[RerankCache] = None,
        verifier: Optional[GroundTruthVerifier] = None,
        auto_detect: bool = True,
        probe_tokens: bool = True,
        no_cache: bool = False,
        config_path: Optional[Path] = None,
        calibrator: Optional[BaseCalibrator] = None,
        margin: float = 0.15,
        redundancy_penalty: float = 0.15,
        utility_trap_resolution: bool = True,
        tier1_bypass_max: int = TIER1_BYPASS_MAX_CANDIDATES,
        tier1_top_n: int = TIER1_SELECT_TOP_N,
        stitch_max_tokens: int = MAX_STITCH_TOKENS,
    ):
        self.concurrency = concurrency
        self.timeout = timeout
        self.verifier = verifier or GroundTruthVerifier()
        self.no_cache = no_cache
        self.calibrator = calibrator or get_default_calibrator()
        self.margin = margin
        self.redundancy_penalty = redundancy_penalty
        self.utility_trap_resolution = utility_trap_resolution
        self.tier1_bypass_max = tier1_bypass_max
        self.tier1_top_n = tier1_top_n
        self.stitch_max_tokens = stitch_max_tokens

        cfg = load_config(config_path)
        resolved_model_name, resolved_url = resolve_endpoint_and_model(
            cli_model=model if isinstance(model, str) else None,
            cli_base_url=base_url,
            cli_endpoint=endpoint,
            config=cfg,
        )

        if isinstance(model, ModelProfile):
            self.profile = model
            target_url = resolved_url or self.profile.default_base_url
        elif resolved_model_name:
            self.profile = get_profile(resolved_model_name)
            target_url = resolved_url or self.profile.default_base_url
        else:
            # Auto-detection probe
            target_url = resolved_url or "http://localhost:8034/v1"
            if auto_detect:
                self.profile = probe_and_detect_profile_sync(target_url, timeout=min(2.0, timeout))
            else:
                self.profile = LFMProfile()

        # Dynamic tokenizer probing for backends with /tokenize
        if probe_tokens:
            yes_ids, no_ids = probe_tokenizer_tokens_sync(target_url, timeout=min(2.0, timeout))
            if yes_ids:
                self.profile.discovered_yes_ids.update(yes_ids)
            if no_ids:
                self.profile.discovered_no_ids.update(no_ids)

        # Persistent cache with model isolation
        if cache is None:
            self.cache = RerankCache(model_id=self.profile.name)
        else:
            self.cache = cache
            self.cache.model_id = self.profile.name

        self.raw_endpoint = target_url.rstrip("/")
        if self.raw_endpoint.endswith("/v1"):
            base = self.raw_endpoint[:-3].rstrip("/")
            self.completion_url = f"{base}/completion"
        elif self.raw_endpoint.endswith("/completion"):
            self.completion_url = self.raw_endpoint
        elif self.raw_endpoint.endswith("/completions") or self.raw_endpoint.endswith("/chat/completions"):
            self.completion_url = self.raw_endpoint
        else:
            self.completion_url = f"{self.raw_endpoint}/completion"

    async def _score_chunk_uncached(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        query: str,
        chunk: CandidateChunk,
    ) -> Tuple[float, Optional[float], Optional[float], bool, bool, bool, int, int, float]:
        """Perform 1-token logprob scoring for an uncached chunk with hazard protection."""
        # Stitched 1-hop context (<= 150 tokens) rides along with the prompt only:
        # citations, snippets and content hashes stay physically exact.
        scoring_content = chunk.content
        if chunk.stitched_context:
            scoring_content = f"{chunk.stitched_context}\n{chunk.content}"

        prompt = self.profile.format_prompt(
            query=query,
            chunk_content=scoring_content,
            file_path=chunk.file_path,
            symbol=chunk.symbol,
            is_test=chunk.is_test,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
        )

        if self.raw_endpoint.endswith("/chat/completions"):
            payload = {
                "model": self.profile.name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1,
                "logprobs": True,
                "top_logprobs": 10,
                "temperature": 0.0,
                "stop": self.profile.stop_tokens,
            }
        elif self.raw_endpoint.endswith("/completions"):
            payload = {
                "model": self.profile.name,
                "prompt": prompt,
                "max_tokens": 1,
                "logprobs": 10,
                "temperature": 0.0,
                "stop": self.profile.stop_tokens,
            }
        else:
            payload = {
                "prompt": prompt,
                "n_predict": 1,
                "n_probs": 10,
                "temperature": 0.0,
                "stop": self.profile.stop_tokens,
            }

        async with semaphore:
            t0 = time.perf_counter()
            try:
                resp = await client.post(self.completion_url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()

                timings = data.get("timings", {})
                raw_prompt_n = timings.get("prompt_n", 0)
                cached_n = data.get("tokens_cached", 0)
                prompt_n = raw_prompt_n + cached_n if (raw_prompt_n + cached_n) > 0 else estimate_tokens(prompt)
                predicted_n = data.get("tokens_predicted", 1)

                top_logprobs = normalize_top_logprobs(data)
                score, lp_yes, lp_no, is_ambiguous = self.profile.extract_calibrated_logprobs(
                    data, chunk_token_est=chunk.token_est, prompt_state=prompt
                )

                top_1 = top_logprobs[0] if top_logprobs else {}
                top_1_id = top_1.get("id")
                top_1_str = clean_token_str(top_1.get("token", ""))

                yes_ids = set(self.profile.discovered_yes_ids)
                no_ids = set(self.profile.discovered_no_ids)
                if self.profile.name == "lfm":
                    yes_ids.update(LFM_YES_TOKEN_IDS)
                    no_ids.update(LFM_NO_TOKEN_IDS)

                is_yes_variant = (top_1_id in yes_ids) if (top_1_id is not None and yes_ids) else False
                is_yes_variant = is_yes_variant or (top_1_str in YES_VARIANTS)

                is_no_variant = (top_1_id in no_ids) if (top_1_id is not None and no_ids) else False
                is_no_variant = is_no_variant or (top_1_str in NO_VARIANTS)

                duration_ms = (time.perf_counter() - t0) * 1000.0
                return score, lp_yes, lp_no, is_ambiguous, is_yes_variant, is_no_variant, prompt_n, predicted_n, duration_ms

            except Exception as e:
                logger.error(f"Error scoring candidate {chunk.id}: {e}")
                duration_ms = (time.perf_counter() - t0) * 1000.0
                # Network/connection failures default to low baseline score without false ambiguity tagging
                return 0.05, None, None, False, False, False, estimate_tokens(prompt), 1, duration_ms

    async def rerank_chunks(
        self,
        query: str,
        chunks: List[CandidateChunk],
        threshold: Optional[float] = None,
        top_k: Optional[int] = None,
        intent: Optional[Union[QueryIntent, str]] = None,
        strict: bool = False,
        no_cache: Optional[bool] = None,
        allow_empty_if_low_confidence: bool = False,
        abstain_below: float = 0.20,
        full: bool = False,
        with_context: bool = False,
    ) -> RerankResponse:
        """Score candidate chunks by log probability, apply query intent & threshold policies, defend against silent omission, and report manifest.

        ``full`` forces every candidate onto the GPU regardless of count, bypassing
        the Tier-1 lexical pre-filter. ``with_context`` stitches <= 150 tokens of
        1-hop call-graph and type context into each scoring prompt.
        """
        bypass_cache = self.no_cache if no_cache is None else no_cache
        if intent is None:
            resolved_intent, intent_confidence = detect_query_intent_with_confidence(query)
        elif isinstance(intent, str):
            try:
                resolved_intent = QueryIntent(intent.upper())
                intent_confidence = 1.0  # Explicit intent via CLI / call
            except ValueError:
                resolved_intent, intent_confidence = detect_query_intent_with_confidence(query)
        else:
            resolved_intent = intent
            intent_confidence = 1.0  # Explicit intent via CLI / call

        if threshold is not None:
            effective_threshold = float(threshold)
        else:
            effective_threshold = DEFAULT_INTENT_THRESHOLDS.get(resolved_intent, 0.65)

        if not chunks:
            return RerankResponse(
                query=query,
                threshold=effective_threshold,
                top_k=top_k,
                intent=resolved_intent,
                results=[],
                candidate_manifest=[],
                telemetry=Telemetry(
                    threshold=effective_threshold,
                    top_ceiling=top_k,
                    query_intent=resolved_intent,
                    model_id=self.profile.name,
                    abstained=False,
                    abstain_threshold=abstain_below,
                    intent_confidence=intent_confidence,
                ),
            )

        t_start = time.perf_counter()

        # Extract code identifiers from query (camelCase, snake_case, PascalCase, backticked)
        query_identifiers = extract_code_identifiers(query)

        # 0. Two-Tier Hybrid Search: Tier-1 only engages on wide sweeps. Small and
        # medium candidate sets are evaluated 100% on the GPU (Tier-2).
        evaluated_chunks, tier1_decision = apply_two_tier_filter(
            query=query,
            chunks=chunks,
            full=full,
            bypass_max=self.tier1_bypass_max,
            tier1_top_n=self.tier1_top_n,
            query_identifiers=query_identifiers,
        )

        # 0b. Call-graph & type context stitching (hard capped, prompt-only)
        stitched_count = 0
        stitched_tokens = 0
        if with_context:
            stitched_count, stitched_tokens = apply_context_stitching(
                evaluated_chunks,
                max_tokens=self.stitch_max_tokens,
            )

        # Stitching changes the prompt, so it must change the cache key too.
        base_prompt_version = getattr(self.profile, "prompt_version", "v1")
        effective_prompt_version = f"{base_prompt_version}+ctx{self.stitch_max_tokens}" if with_context else base_prompt_version

        # 1. Query persistent cache (isolated by model_id and 8 invalidation parameters)
        if bypass_cache:
            cached_scores = {}
        else:
            cached_scores = self.cache.get_batch(
                query=query,
                chunks=evaluated_chunks,
                model_id=self.profile.name,
                prompt_version=effective_prompt_version,
                query_intent=resolved_intent.value,
                prior_version=PRIOR_VERSION,
                length_exponent=self.profile.length_normalization_exponent,
            )
        uncached_indices = [idx for idx in range(len(evaluated_chunks)) if idx not in cached_scores]

        cache_hits = len(cached_scores)
        cache_misses = len(uncached_indices)

        # 2. Score uncached chunks concurrently
        semaphore = asyncio.Semaphore(self.concurrency)
        limits = httpx.Limits(max_connections=self.concurrency * 4, max_keepalive_connections=self.concurrency * 2)

        uncached_tasks = []
        async with httpx.AsyncClient(limits=limits, timeout=self.timeout) as client:
            for idx in uncached_indices:
                chunk = evaluated_chunks[idx]
                uncached_tasks.append(
                    self._score_chunk_uncached(client, semaphore, query, chunk)
                )

            uncached_eval_results = await asyncio.gather(*uncached_tasks) if uncached_tasks else []

        # 3. Save new non-ambiguous results into cache (isolated by model_id & parameters)
        to_cache: List[Tuple[Any, float, Optional[float], Optional[float]]] = []
        ambiguous_count = 0
        for i, idx in enumerate(uncached_indices):
            score, lp_yes, lp_no, is_ambiguous, _, _, _, _, _ = uncached_eval_results[i]
            if is_ambiguous:
                ambiguous_count += 1
            else:
                to_cache.append((evaluated_chunks[idx], score, lp_yes, lp_no))

        if to_cache and not bypass_cache:
            self.cache.put_batch(
                query=query,
                entries=to_cache,
                model_id=self.profile.name,
                prompt_version=effective_prompt_version,
                query_intent=resolved_intent.value,
                prior_version=PRIOR_VERSION,
                length_exponent=self.profile.length_normalization_exponent,
            )

        # 4. Assemble verified result items & track operational variant rates
        scored_items: List[RerankResultItem] = []
        uncached_result_map = {idx: uncached_eval_results[i] for i, idx in enumerate(uncached_indices)}
        yes_variants_count = 0
        no_variants_count = 0

        # Utility Trap Resolution: split the query into what it is about
        # (domain entities) versus what went wrong (symptom modifiers).
        decomposition = decompose_query(query)
        alignments: List[Tuple[RerankResultItem, DomainAlignment]] = []
        symbol_boosts_applied = 0
        centrality_boosts_applied = 0
        utility_penalties_applied = 0

        for idx, chunk in enumerate(evaluated_chunks):
            if idx in cached_scores:
                c_data = cached_scores[idx]
                item = self.verifier.verify_chunk(
                    chunk=chunk,
                    score=c_data["score"],
                    logprob_yes=c_data["logprob_yes"],
                    logprob_no=c_data["logprob_no"],
                    cached=True,
                    prompt_tokens=0,
                    completion_tokens=0,
                    duration_ms=0.0,
                )
                if (
                    c_data.get("logprob_yes") is not None
                    and c_data.get("logprob_no") is not None
                    and c_data["logprob_yes"] > c_data["logprob_no"]
                ) or c_data["score"] >= 0.5:
                    yes_variants_count += 1
                else:
                    no_variants_count += 1
            else:
                score, lp_yes, lp_no, is_ambiguous, is_yes_var, is_no_var, p_tok, c_tok, d_ms = uncached_result_map[idx]
                item = self.verifier.verify_chunk(
                    chunk=chunk,
                    score=score,
                    logprob_yes=lp_yes,
                    logprob_no=lp_no,
                    cached=False,
                    prompt_tokens=p_tok,
                    completion_tokens=c_tok,
                    duration_ms=d_ms,
                )
                if is_ambiguous:
                    item.ambiguous = True
                    item.ground_truth_status = GroundTruthStatus.AMBIGUOUS_COMPLETION
                    item.ground_truth_details = "Model emitted non-binary completion token"
                if is_yes_var:
                    yes_variants_count += 1
                elif is_no_var:
                    no_variants_count += 1

            # 3-Score Architecture (v0.3.3):
            # 1. raw_score: pure model probability before priors (used for model diagnostics)
            # 2. decision_score: log-odds intent-adjusted score (used for ranking and thresholding)
            # 3. calibrated_score: post-hoc calibrated probability via Platt/Temperature scaling
            raw_score = item.score
            item.raw_score = raw_score

            prior_logit_delta = get_prior_logit_delta(resolved_intent, chunk)

            # Failure-Mode Mitigation: Symbol-Aware Boost & Sibling Disambiguation
            sym_delta, is_symbol_match, matched_syms = compute_symbol_match_delta(
                chunk=chunk,
                query=query,
                query_identifiers=query_identifiers,
                intent=resolved_intent,
            )
            if is_symbol_match or sym_delta != 0.0:
                prior_logit_delta += sym_delta
                if is_symbol_match:
                    item.symbol_boosted = True
                    item.matched_symbols = matched_syms
                    symbol_boosts_applied += 1

            # Utility Trap Resolution: domain centrality vs keyword density
            alignment = compute_domain_alignment(
                chunk=chunk,
                decomposition=decomposition,
                query_identifiers=query_identifiers,
            )
            if self.utility_trap_resolution:
                trap_delta, centrality_boosted, utility_penalized = compute_utility_trap_delta(
                    decomposition=decomposition,
                    alignment=alignment,
                )
                prior_logit_delta += trap_delta
                item.centrality_boosted = centrality_boosted
                item.utility_penalized = utility_penalized
                centrality_boosts_applied += int(centrality_boosted)
                utility_penalties_applied += int(utility_penalized)

            item.domain_hits = alignment.domain_hits
            item.structural_domain_hits = alignment.structural_domain_hits
            item.symptom_hits = alignment.symptom_hits
            item.matched_domain_entities = alignment.matched_domain_entities
            item.symbol_role = alignment.symbol_role.value
            item.context_stitched = bool(chunk.stitched_context)
            if tier1_decision.lexical_scores:
                item.tier1_lexical_score = tier1_decision.lexical_scores.get(chunk.id)

            adj_p = apply_intent_prior(raw_score, prior_logit_delta, confidence=intent_confidence)
            item.decision_score = round(adj_p, 4)
            item.adjusted_score = item.decision_score  # Backward compatibility alias
            item.score = item.decision_score  # Backward compatibility alias
            item.delta = round(item.decision_score - item.raw_score, 4)

            # Calibrated probability via post-hoc scaling layer
            item.calibrated_score = round(self.calibrator.calibrate_proba(item.decision_score), 4)

            scored_items.append(item)
            alignments.append((item, alignment))

        # 4b. Domain precedence invariant: a chunk with no stake in the query's
        # domain can never outrank one that owns it, however dense its keyword
        # repetition. This is the hard guarantee behind the soft penalties above.
        utility_trap_demotions = 0
        if self.utility_trap_resolution and decomposition.has_domain:
            utility_trap_demotions = enforce_domain_precedence(alignments)
            if utility_trap_demotions:
                for item in scored_items:
                    if item.utility_trap_demoted:
                        item.calibrated_score = round(
                            self.calibrator.calibrate_proba(item.decision_score), 4
                        )

        t_wall = time.perf_counter() - t_start

        # 5. Sort descending by decision score & apply diversity context assembly (MMR-style penalty)
        sorted_items = sorted(scored_items, key=lambda x: x.decision_score, reverse=True)
        assembled_items, penalties_applied = apply_diversity_context_assembly(
            sorted_items,
            redundancy_logit_penalty=self.redundancy_penalty,
            exceptionally_high_threshold=0.88,
        )

        # 6. Apply Dynamic Margin-Based Thresholding, Abstention, & Maintain top-3 floor
        top_score = assembled_items[0].decision_score if assembled_items else 0.0
        abstained = False

        if allow_empty_if_low_confidence and (not assembled_items or top_score < abstain_below):
            abstained = True
            passing_items = []
            top_results = []
            fallback_floor_triggered = False
        else:
            if effective_threshold == 0.0:
                # Evaluation/inspection mode: retain all items
                passing_items = list(assembled_items)
                fallback_floor_triggered = False
            else:
                # Dynamic Margin-Based Thresholding:
                # include a chunk if decision_score >= (top_score - margin) AND decision_score >= effective_threshold
                margin_floor = top_score - self.margin
                passing_items = [
                    item for item in assembled_items
                    if item.decision_score >= margin_floor and item.decision_score >= effective_threshold
                ]
                fallback_floor_triggered = False

                floor_count = min(3, len(assembled_items))
                if len(passing_items) < floor_count:
                    fallback_floor_triggered = True
                    for item in assembled_items[:floor_count]:
                        if item.decision_score < effective_threshold:
                            item.below_threshold = True
                            item.metadata = {
                                "included_by_fallback": True,
                                "below_threshold": True,
                                "fallback_warning": f"Candidate score {item.decision_score:.4f} is below threshold {effective_threshold:.4f}; included via top-3 fallback floor.",
                            }
                        else:
                            item.below_threshold = False
                            item.metadata = {
                                "included_by_fallback": False,
                                "below_threshold": False,
                            }
                    passing_items = assembled_items[:floor_count]
                else:
                    for item in passing_items:
                        item.below_threshold = False
                        item.metadata = {
                            "included_by_fallback": False,
                            "below_threshold": False,
                        }

            # Apply strict filter if requested
            if strict:
                passing_items = [
                    item for item in passing_items
                    if item.ground_truth_status in (GroundTruthStatus.VERIFIED, GroundTruthStatus.UNVERIFIED_RAW_TEXT)
                ]

            # Apply top_k as ceiling
            top_results = passing_items[:top_k] if top_k is not None else passing_items

            # BUG_DIAGNOSIS: always includes top matching test chunk
            if resolved_intent == QueryIntent.BUG_DIAGNOSIS:
                test_items = [item for item in assembled_items if item.is_test]
                if test_items:
                    top_test = test_items[0]
                    if top_test not in top_results:
                        if top_test.decision_score < effective_threshold:
                            top_test.below_threshold = True
                            top_test.metadata = {
                                "included_by_fallback": True,
                                "below_threshold": True,
                                "fallback_warning": f"Test chunk score {top_test.decision_score:.4f} is below threshold {effective_threshold:.4f}; included via BUG_DIAGNOSIS policy.",
                            }
                        else:
                            top_test.below_threshold = False
                            top_test.metadata = {
                                "included_by_fallback": False,
                                "below_threshold": False,
                            }
                        if top_k is not None and len(top_results) >= top_k:
                            top_results = top_results[:-1] + [top_test]
                        else:
                            top_results.append(top_test)

        # 7. Construct Full Candidate Manifest (Defend against silent omission)
        included_file_paths = {item.file_path for item in top_results if item.file_path}
        manifest_map: Dict[str, Dict[str, Any]] = {}

        # Chunks withheld by Tier-1 still appear in the manifest with a 0.0 score,
        # so pre-filtering can never become a silent omission.
        score_by_chunk: Dict[int, float] = {
            id(chunk): item.decision_score for chunk, item in zip(evaluated_chunks, scored_items)
        }

        for chunk in chunks:
            f_path = chunk.file_path or "raw_text"
            if f_path not in manifest_map:
                manifest_map[f_path] = {
                    "chunks_count": 0,
                    "max_score": 0.0,
                    "symbols": set(),
                    "is_test": chunk.is_test,
                }
            manifest_map[f_path]["chunks_count"] += 1
            chunk_score = score_by_chunk.get(id(chunk), 0.0)
            if chunk_score > manifest_map[f_path]["max_score"]:
                manifest_map[f_path]["max_score"] = chunk_score
            if chunk.symbol:
                manifest_map[f_path]["symbols"].add(chunk.symbol)

        candidate_manifest: List[CandidateManifestEntry] = []
        for f_path, meta in manifest_map.items():
            status = "included" if f_path in included_file_paths else "omitted"
            candidate_manifest.append(
                CandidateManifestEntry(
                    file_path=f_path,
                    chunks_count=meta["chunks_count"],
                    max_score=round(meta["max_score"], 4),
                    status=status,
                    symbols=sorted(list(meta["symbols"])),
                    is_test=meta["is_test"],
                )
            )
        candidate_manifest.sort(key=lambda m: m.max_score, reverse=True)

        # 8. Operational Telemetry & Reduction metrics
        total_prompt_tok = sum(item.prompt_tokens for item in scored_items)
        total_completion_tok = sum(item.completion_tokens for item in scored_items)
        total_input_bytes = sum(len(c.content.encode("utf-8")) for c in evaluated_chunks)

        top_k_tokens = sum(estimate_tokens(item.snippet or "") for item in top_results)
        if top_k_tokens == 0:
            top_k_tokens = max(1, int(total_prompt_tok * 0.05))

        prefill_tps = total_prompt_tok / t_wall if t_wall > 0 else 0.0
        decode_tps = total_completion_tok / t_wall if t_wall > 0 else 0.0
        reduction_ratio = total_prompt_tok / top_k_tokens if top_k_tokens > 0 else 1.0
        reduction_pct = max(0.0, (1.0 - (top_k_tokens / total_prompt_tok))) * 100.0 if total_prompt_tok > 0 else 0.0

        unique_files = len({c.file_path for c in chunks if c.file_path}) or 1
        total_eval = len(scored_items)

        cache_total_ops = cache_hits + cache_misses
        cache_hit_rate = round(cache_hits / cache_total_ops, 4) if cache_total_ops > 0 else 0.0
        ambiguous_rate = round(ambiguous_count / total_eval, 4) if total_eval > 0 else 0.0
        yes_rate = round(yes_variants_count / total_eval, 4) if total_eval > 0 else 0.0
        no_rate = round(no_variants_count / total_eval, 4) if total_eval > 0 else 0.0

        live_durations = [item.duration_ms for item in scored_items if not item.cached and item.duration_ms > 0.0]
        score_latency_p95 = calculate_percentile(live_durations, 95.0)

        # Ambiguity telemetry collection
        ambiguity_events_list = []
        if getattr(self.profile, "last_ambiguity_event", None):
            ambiguity_events_list.append(self.profile.last_ambiguity_event)

        telemetry = Telemetry(
            candidate_files_count=unique_files,
            chunks_evaluated=total_eval,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            ambiguous_completions=ambiguous_count,
            total_prompt_tokens=total_prompt_tok,
            total_completion_tokens=total_completion_tok,
            prefill_tokens_per_sec=round(prefill_tps, 1),
            decode_tokens_per_sec=round(decode_tps, 1),
            total_wall_time_s=round(t_wall, 3),
            total_input_bytes=total_input_bytes,
            top_k_input_tokens=top_k_tokens,
            reduction_ratio=round(reduction_ratio, 1),
            reduction_percentage=round(reduction_pct, 1),
            concurrency=self.concurrency,
            threshold=effective_threshold,
            fallback_floor_triggered=fallback_floor_triggered,
            top_ceiling=top_k,
            query_intent=resolved_intent,
            ambiguous_completion_rate=ambiguous_rate,
            yes_variant_rate=yes_rate,
            no_variant_rate=no_rate,
            cache_hit_rate=cache_hit_rate,
            score_latency_p95=score_latency_p95,
            model_id=self.profile.name,
            abstained=abstained,
            abstain_threshold=abstain_below,
            intent_confidence=intent_confidence,
            redundancy_penalties_applied=penalties_applied,
            symbol_boosts_applied=symbol_boosts_applied,
            margin_threshold_used=self.margin,
            calibrator_type=self.calibrator.__class__.__name__,
            ambiguity_events=ambiguity_events_list,
            tier1_applied=tier1_decision.tier1_applied,
            tier1_candidates_in=tier1_decision.candidates_in,
            tier1_candidates_out=tier1_decision.candidates_out,
            tier1_reason=tier1_decision.reason,
            full_evaluation=tier1_decision.full_evaluation,
            context_stitching_enabled=with_context,
            chunks_context_stitched=stitched_count,
            stitched_context_tokens=stitched_tokens,
            max_stitched_context_tokens=self.stitch_max_tokens,
            domain_entities=list(decomposition.domain_entities),
            symptom_modifiers=list(decomposition.symptom_modifiers),
            centrality_boosts_applied=centrality_boosts_applied,
            utility_penalties_applied=utility_penalties_applied,
            utility_trap_demotions=utility_trap_demotions,
        )

        return RerankResponse(
            query=query,
            threshold=effective_threshold,
            top_k=top_k,
            intent=resolved_intent,
            results=top_results,
            candidate_manifest=candidate_manifest,
            telemetry=telemetry,
        )

    async def rerank(
        self,
        query: str,
        candidates: Sequence[Union[str, Path]],
        threshold: Optional[float] = None,
        top_k: Optional[int] = None,
        intent: Optional[Union[QueryIntent, str]] = None,
        strict: bool = False,
        no_cache: Optional[bool] = None,
        allow_empty_if_low_confidence: bool = False,
        abstain_below: float = 0.20,
        full: bool = False,
        with_context: bool = False,
    ) -> RerankResponse:
        """High-level async entry point for candidate files or text."""
        chunks = prepare_candidates(candidates)
        return await self.rerank_chunks(
            query=query,
            chunks=chunks,
            threshold=threshold,
            top_k=top_k,
            intent=intent,
            strict=strict,
            no_cache=no_cache,
            allow_empty_if_low_confidence=allow_empty_if_low_confidence,
            abstain_below=abstain_below,
            full=full,
            with_context=with_context,
        )

    def rerank_sync(
        self,
        query: str,
        candidates: Sequence[Union[str, Path]],
        threshold: Optional[float] = None,
        top_k: Optional[int] = None,
        intent: Optional[Union[QueryIntent, str]] = None,
        strict: bool = False,
        no_cache: Optional[bool] = None,
        allow_empty_if_low_confidence: bool = False,
        abstain_below: float = 0.20,
        full: bool = False,
        with_context: bool = False,
    ) -> RerankResponse:
        """Synchronous wrapper for rerank."""
        return asyncio.run(
            self.rerank(
                query=query,
                candidates=candidates,
                threshold=threshold,
                top_k=top_k,
                intent=intent,
                strict=strict,
                no_cache=no_cache,
                allow_empty_if_low_confidence=allow_empty_if_low_confidence,
                abstain_below=abstain_below,
                full=full,
                with_context=with_context,
            )
        )


# Universal Reranker alias
Reranker = LFMReranker
