"""Two-Tier Hybrid Search: cheap lexical pre-filter (Tier-1) in front of SLM scoring (Tier-2).

Tier-2 (1-token logprob scoring on the GPU) is the authoritative ranker and is
deliberately never skipped for small-to-medium candidate sets. Tier-1 exists
only to keep very wide sweeps tractable, and is tuned for recall, not speed:

* ``<= TIER1_BYPASS_MAX_CANDIDATES`` candidates  -> Tier-1 is bypassed entirely,
  100% of candidates are evaluated on the GPU.
* ``> TIER1_BYPASS_MAX_CANDIDATES`` candidates   -> Tier-1 ranks lexically and
  forwards the top ``TIER1_SELECT_TOP_N`` to Tier-2.
* ``full=True`` (CLI ``--full``)                 -> Tier-1 is bypassed regardless
  of candidate count.

Because the select width (80) is wider than the bypass ceiling (60), no chunk is
ever dropped until a sweep exceeds 80 candidates, and exact symbol matches are
scored far above every other signal so a named target cannot fall out of the
retained window.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from .adapters import (
    GENERIC_QUERY_STOPWORDS,
    SYMPTOM_MODIFIER_TERMS,
    lexical_terms,
    normalize_term,
    saturating_term_frequency,
    split_symbol_components,
)
from .models import CandidateChunk

# Candidate-count policy. Quality over speed: bypass ceiling < select width.
TIER1_BYPASS_MAX_CANDIDATES: int = 60
TIER1_SELECT_TOP_N: int = 80

# Lexical signal weights. The exact-symbol weight dominates every other term so
# an explicitly named target is always inside the retained window.
EXACT_SYMBOL_WEIGHT: float = 10.0
SYMBOL_COMPONENT_WEIGHT: float = 4.0
PATH_MATCH_WEIGHT: float = 2.0
CONTENT_COVERAGE_WEIGHT: float = 3.0
CONTENT_FREQUENCY_WEIGHT: float = 1.0
SYMPTOM_ONLY_WEIGHT: float = 0.25


class Tier1Decision(BaseModel):
    """Audit record of what the Tier-1 pre-filter did to a candidate set."""

    tier1_applied: bool = Field(default=False, description="Whether lexical pre-filtering actually ran")
    candidates_in: int = Field(default=0, description="Candidate chunks offered to Tier-1")
    candidates_out: int = Field(default=0, description="Candidate chunks forwarded to Tier-2 GPU scoring")
    reason: str = Field(default="", description="Machine-readable reason for the bypass/filter decision")
    lexical_scores: Dict[str, float] = Field(default_factory=dict, description="Chunk ID -> Tier-1 lexical score")
    dropped_chunk_ids: List[str] = Field(default_factory=list, description="Chunk IDs withheld from Tier-2")

    @property
    def full_evaluation(self) -> bool:
        """True when every offered candidate reached Tier-2."""
        return self.candidates_out >= self.candidates_in


def build_query_terms(query: str) -> Tuple[List[str], List[str]]:
    """Split a query into (content terms, symptom modifier terms), both normalized.

    Content terms keep their order and exclude generic stopwords; symptom terms
    are retained separately so a chunk cannot ride to the top of Tier-1 on
    failure vocabulary alone.
    """
    content_terms: List[str] = []
    symptom_terms: List[str] = []
    seen: set = set()

    for term in lexical_terms(query):
        if len(term) < 2 or term in seen:
            continue
        seen.add(term)
        if term in SYMPTOM_MODIFIER_TERMS:
            symptom_terms.append(term)
        elif term not in GENERIC_QUERY_STOPWORDS:
            content_terms.append(term)

    return content_terms, symptom_terms


def compute_lexical_score(
    chunk: CandidateChunk,
    content_terms: Sequence[str],
    symptom_terms: Sequence[str] = (),
    query_identifiers: Optional[Sequence[str]] = None,
) -> float:
    """Score a chunk's lexical affinity to a query without touching the GPU.

    Signals, in descending authority: exact symbol match, symbol component
    overlap, file-path overlap, distinct-term coverage in the body, and finally
    a saturating raw term frequency so keyword repetition cannot dominate.
    """
    symbol = chunk.symbol or ""
    symbol_lower = symbol.lower()
    score = 0.0

    # 1. Exact symbol match against an identifier lifted from the query.
    for ident in query_identifiers or ():
        if ident and ident.lower() == symbol_lower:
            score += EXACT_SYMBOL_WEIGHT
            break

    if not content_terms and not symptom_terms:
        return round(score, 4)

    term_set = set(content_terms)

    # 2. Symbol component overlap (llm_chat_apply_template vs "chat template").
    components = [c for c in split_symbol_components(symbol) if c and c != "module_scope"]
    if components and term_set:
        overlap = len(term_set & set(components))
        if overlap:
            score += SYMBOL_COMPONENT_WEIGHT * (overlap / len(set(components)))

    # 3. File path overlap (llama-chat.cpp vs "chat").
    if chunk.file_path and term_set:
        path_terms = set(lexical_terms(chunk.file_path))
        path_overlap = len(term_set & path_terms)
        if path_overlap:
            score += PATH_MATCH_WEIGHT * saturating_term_frequency(path_overlap)

    # 4/5. Body coverage and saturating body frequency.
    if term_set or symptom_terms:
        body_terms = lexical_terms(chunk.content)
        if body_terms:
            body_counts: Dict[str, int] = {}
            for t in body_terms:
                body_counts[t] = body_counts.get(t, 0) + 1

            matched_terms = [t for t in term_set if t in body_counts]
            if term_set:
                score += CONTENT_COVERAGE_WEIGHT * (len(matched_terms) / len(term_set))

            total_tf = sum(body_counts[t] for t in matched_terms)
            score += CONTENT_FREQUENCY_WEIGHT * saturating_term_frequency(total_tf, k=3.0)

            symptom_tf = sum(body_counts.get(t, 0) for t in symptom_terms)
            score += SYMPTOM_ONLY_WEIGHT * saturating_term_frequency(symptom_tf, k=3.0)

    return round(score, 4)


def apply_two_tier_filter(
    query: str,
    chunks: Sequence[CandidateChunk],
    full: bool = False,
    bypass_max: int = TIER1_BYPASS_MAX_CANDIDATES,
    tier1_top_n: int = TIER1_SELECT_TOP_N,
    query_identifiers: Optional[Sequence[str]] = None,
) -> Tuple[List[CandidateChunk], Tier1Decision]:
    """Decide which candidates reach Tier-2 GPU scoring.

    Returns ``(chunks_for_tier2, decision)``. Retained chunks keep their original
    input order so downstream indices and the candidate manifest stay stable.
    """
    all_chunks = list(chunks)
    total = len(all_chunks)

    if full:
        return all_chunks, Tier1Decision(
            tier1_applied=False,
            candidates_in=total,
            candidates_out=total,
            reason="full_bypass_flag",
        )

    if total <= bypass_max:
        return all_chunks, Tier1Decision(
            tier1_applied=False,
            candidates_in=total,
            candidates_out=total,
            reason="small_candidate_set",
        )

    content_terms, symptom_terms = build_query_terms(query)
    scores: Dict[str, float] = {}
    ranked: List[Tuple[float, int, CandidateChunk]] = []

    for idx, chunk in enumerate(all_chunks):
        lex = compute_lexical_score(
            chunk,
            content_terms=content_terms,
            symptom_terms=symptom_terms,
            query_identifiers=query_identifiers,
        )
        scores[chunk.id] = lex
        # Negative index keeps the sort stable on ties (earlier input wins).
        ranked.append((lex, -idx, chunk))

    ranked.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
    retained = ranked[:tier1_top_n]
    retained_ids = {entry[2].id for entry in retained}

    kept = [c for c in all_chunks if c.id in retained_ids]
    dropped_ids = [c.id for c in all_chunks if c.id not in retained_ids]

    reason = "tier1_top_n_covers_all" if not dropped_ids else "tier1_lexical_prefilter"

    return kept, Tier1Decision(
        tier1_applied=True,
        candidates_in=total,
        candidates_out=len(kept),
        reason=reason,
        lexical_scores=scores,
        dropped_chunk_ids=dropped_ids,
    )
