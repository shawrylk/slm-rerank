"""Comprehensive Unit Tests for slm-rerank v0.5.0 Quality Upgrades.

Covers:
1. Two-Tier Hybrid Search (slm_rerank/filter.py)
   - Dynamic threshold cutoff (<= 60 bypass, > 60 pre-filtering to top 80)
   - Force full bypass (--full)
   - Lexical scoring signals (exact symbol, path, saturating frequency)
2. Call-Graph & Type Context Stitching (slm_rerank/stitcher.py)
   - Hard token cap (<= 150 tokens)
   - Enclosing type, interface, and caller extraction
   - Graceful fallback on malformed or missing context
3. Utility Trap Resolution & Architectural Centrality (slm_rerank/client.py & adapters.py)
   - Query decomposition (Domain entities vs Symptom modifiers)
   - BM25 term frequency saturation
   - Symbol role classification (PUBLIC_API vs INTERNAL_HELPER)
   - Domain alignment and precedence (resolving TASK-LLAMA-04 failure mode)
"""

import math
import pytest
from pathlib import Path

from slm_rerank.adapters import (
    SymbolRole,
    classify_symbol_role,
    lexical_terms,
    normalize_term,
    saturating_term_frequency,
    split_symbol_components,
    SYMPTOM_MODIFIER_TERMS,
)
from slm_rerank.client import (
    DomainAlignment,
    QueryDecomposition,
    compute_domain_alignment,
    compute_utility_trap_delta,
    decompose_query,
    enforce_domain_precedence,
)
from slm_rerank.filter import (
    TIER1_BYPASS_MAX_CANDIDATES,
    TIER1_SELECT_TOP_N,
    Tier1Decision,
    apply_two_tier_filter,
    build_query_terms,
    compute_lexical_score,
)
from slm_rerank.models import CandidateChunk, QueryIntent
from slm_rerank.stitcher import (
    MAX_STITCH_TOKENS,
    StitchedContext,
    extract_called_identifiers,
    extract_definitions,
    find_callers,
    find_enclosing_type,
    stitch_chunk_context,
)


# ==============================================================================
# 1. Two-Tier Hybrid Search Tests
# ==============================================================================

def test_tier1_bypass_under_threshold():
    """Candidate sets <= 60 must bypass Tier-1 entirely for zero recall loss."""
    candidates = [
        CandidateChunk(
            id=f"src/file_{i}.py:1-10:func_{i}",
            file_path=f"src/file_{i}.py",
            start_line=1,
            end_line=10,
            symbol=f"func_{i}",
            content=f"def func_{i}(): pass",
            token_est=10,
        )
        for i in range(50)
    ]
    retained, decision = apply_two_tier_filter(
        query="find file_10 func_10",
        chunks=candidates,
        full=False,
    )
    assert len(retained) == 50
    assert decision.full_evaluation is True
    assert decision.tier1_applied is False
    assert "small_candidate_set" in decision.reason.lower() or "bypass" in decision.reason.lower()


def test_tier1_prefilter_over_threshold():
    """Candidate sets > 60 must activate Tier-1 pre-filter and retain top 80."""
    candidates = [
        CandidateChunk(
            id=f"src/file_{i}.py:1-10:func_{i}",
            file_path=f"src/file_{i}.py",
            start_line=1,
            end_line=10,
            symbol=f"func_{i}",
            content=f"def func_{i}(): pass # keyword_{i % 5}",
            token_est=10,
        )
        for i in range(120)
    ]
    # Make chunk 42 an exact symbol match
    candidates[42].symbol = "critical_target_symbol"

    retained, decision = apply_two_tier_filter(
        query="look for critical_target_symbol in file_42",
        chunks=candidates,
        full=False,
    )
    assert len(retained) == TIER1_SELECT_TOP_N
    assert decision.tier1_applied is True
    assert decision.candidates_in == 120
    assert decision.candidates_out == TIER1_SELECT_TOP_N
    assert any(c.symbol == "critical_target_symbol" for c in retained)


def test_tier1_force_full_bypass():
    """CLI --full / full_scan=True must bypass Tier-1 even with 100+ candidates."""
    candidates = [
        CandidateChunk(
            id=f"src/file_{i}.py:1-10:func_{i}",
            file_path=f"src/file_{i}.py",
            start_line=1,
            end_line=10,
            symbol=f"func_{i}",
            content=f"def func_{i}(): pass",
            token_est=10,
        )
        for i in range(100)
    ]
    retained, decision = apply_two_tier_filter(
        query="general search",
        chunks=candidates,
        full=True,
    )
    assert len(retained) == 100
    assert decision.tier1_applied is False
    assert decision.full_evaluation is True
    assert "forced" in decision.reason.lower() or "full" in decision.reason.lower()


def test_lexical_scoring_signals():
    """Verify exact symbol match strongly outscores generic term repetitions."""
    target_chunk = CandidateChunk(
        id="src/router.py:1-10:dispatch_route",
        file_path="src/router.py",
        start_line=1,
        end_line=10,
        symbol="dispatch_route",
        content="def dispatch_route(req): return handle(req)",
        token_est=20,
    )
    repetitive_chunk = CandidateChunk(
        id="src/utils.py:1-10:helper",
        file_path="src/utils.py",
        start_line=1,
        end_line=10,
        symbol="helper",
        content="route route route route route handle handle handle",
        token_est=20,
    )
    content_terms, symptom_terms = build_query_terms("find dispatch_route handler")
    target_score = compute_lexical_score(target_chunk, content_terms, symptom_terms)
    rep_score = compute_lexical_score(repetitive_chunk, content_terms, symptom_terms)
    assert target_score > rep_score


# ==============================================================================
# 2. Call-Graph & Type Context Stitching Tests
# ==============================================================================

SAMPLE_CODE = """
class ChatProcessor:
    def __init__(self, config):
        self.config = config

    def apply_template(self, messages):
        return self._render(messages)

    def _render(self, messages):
        raw = parse_token(messages)
        return raw
"""

def test_context_stitcher_budget_and_extraction(tmp_path):
    """Context stitcher must extract enclosing class and adhere to <= 150 token cap."""
    fpath = tmp_path / "chat_proc.py"
    fpath.write_text(SAMPLE_CODE)

    chunk = CandidateChunk(
        id="chat_proc.py:8-11:_render",
        file_path=str(fpath),
        start_line=8,
        end_line=11,
        symbol="_render",
        content="    def _render(self, messages):\n        raw = parse_token(messages)\n        return raw",
        token_est=25,
    )

    ctx = stitch_chunk_context(chunk, max_tokens=MAX_STITCH_TOKENS)
    assert not ctx.is_empty
    assert ctx.token_est <= MAX_STITCH_TOKENS
    assert "ChatProcessor" in ctx.text
    assert ctx.enclosing_type is not None


def test_context_stitcher_graceful_missing_file():
    """Missing file must produce empty StitchedContext without throwing exceptions."""
    chunk = CandidateChunk(
        id="none:1-1:foo",
        file_path="/non/existent/path.py",
        start_line=1,
        end_line=1,
        symbol="foo",
        content="def foo(): pass",
        token_est=10,
    )
    ctx = stitch_chunk_context(chunk, max_tokens=150)
    assert ctx.is_empty
    assert ctx.text == ""


# ==============================================================================
# 3. Utility Trap Resolution & Centrality Tests
# ==============================================================================

def test_bm25_saturating_term_frequency():
    """Verify diminishing returns of term repetitions in saturating_term_frequency."""
    tf1 = saturating_term_frequency(1, k=1.2)
    tf2 = saturating_term_frequency(2, k=1.2)
    tf5 = saturating_term_frequency(5, k=1.2)
    tf20 = saturating_term_frequency(20, k=1.2)

    # 1 vs 2 gains significantly
    gain_1_to_2 = tf2 - tf1
    # 5 vs 20 gains very little despite 4x increase
    gain_5_to_20 = tf20 - tf5

    assert tf1 > 0.4
    assert tf20 < 1.0
    assert gain_1_to_2 > (gain_5_to_20 / 3)


def test_symbol_role_classification():
    """Classify domain export vs internal helper function."""
    # llama-chat.cpp export
    pub_role = classify_symbol_role(
        symbol="llm_chat_apply_template",
        content="int32_t llm_chat_apply_template(...) { ... }",
        file_path="llama-chat.cpp",
        line_span=80,
    )
    assert pub_role == SymbolRole.PUBLIC_API

    # Small internal static helper with utility verb 'parse_'
    helper_role = classify_symbol_role(
        symbol="parse_token",
        content="static uint32_t parse_token(const char * src) { ... }",
        file_path="llama-grammar.cpp",
        line_span=25,
    )
    assert helper_role == SymbolRole.INTERNAL_HELPER


def test_query_decomposition_llama_04():
    """Verify query decomposition on TASK-LLAMA-04."""
    query = "diagnose crash exception and syntax error in chat template formatting"
    decomp = decompose_query(query)

    assert "crash" in decomp.symptom_modifiers or "exception" in decomp.symptom_modifiers
    assert "syntax" in decomp.symptom_modifiers or "error" in decomp.symptom_modifiers
    assert "chat" in decomp.domain_entities
    assert "template" in decomp.domain_entities or "formatting" in decomp.domain_entities


def test_utility_trap_delta_lifts_target_over_helper():
    """Directly test that llm_chat_apply_template receives positive delta while parse_token is penalized."""
    query = "diagnose crash exception and syntax error in chat template formatting"
    decomp = decompose_query(query)

    # Target chunk: domain owner
    target_chunk = CandidateChunk(
        id="src/llama-chat.cpp:100-150:llm_chat_apply_template",
        file_path="src/llama-chat.cpp",
        start_line=100,
        end_line=150,
        symbol="llm_chat_apply_template",
        content="""
        int32_t llm_chat_apply_template(
            const struct llama_model * model,
            const char * tmpl,
            const struct llama_chat_message * chat,
            size_t n_msg,
            bool add_ass,
            char * buf,
            int32_t length
        ) {
            // Apply formatting for jinja chat template
            return 0;
        }
        """,
        token_est=120,
    )

    # Competitor chunk: helper function with repetitive error/syntax terms
    helper_chunk = CandidateChunk(
        id="src/llama-grammar.cpp:185-229:parse_token",
        file_path="src/llama-grammar.cpp",
        start_line=185,
        end_line=229,
        symbol="parse_token",
        content="""
        static uint32_t parse_token(const char * src) {
            // error: syntax error in token parsing
            // exception thrown on invalid syntax error format
            if (!src) throw std::runtime_error("syntax error crash");
            return 0;
        }
        """,
        token_est=45,
    )

    target_align = compute_domain_alignment(target_chunk, decomp)
    target_delta, target_boost, target_pen = compute_utility_trap_delta(decomp, target_align)

    helper_align = compute_domain_alignment(helper_chunk, decomp)
    helper_delta, helper_boost, helper_pen = compute_utility_trap_delta(decomp, helper_align)

    assert target_align.domain_hits >= 2
    assert target_align.symbol_role == SymbolRole.PUBLIC_API
    assert target_delta > 0.5
    assert target_boost is True

    assert helper_align.domain_hits == 0
    assert helper_align.symbol_role == SymbolRole.INTERNAL_HELPER
    assert helper_delta < 0.0
    assert helper_pen is True

    assert target_delta > helper_delta
