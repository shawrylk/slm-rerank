"""Unit tests for QueryIntent policies, hybrid thresholding, top-3 floor, and operational telemetry tracking."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import patch

from slm_rerank.client import (
    DEFAULT_INTENT_THRESHOLDS,
    LFMReranker,
    calculate_percentile,
    detect_query_intent,
    is_doc_chunk,
)
from slm_rerank.models import (
    CandidateChunk,
    GroundTruthStatus,
    QueryIntent,
    RerankResponse,
    Telemetry,
)


def test_query_intent_detection():
    assert detect_query_intent("how is the payment processor implemented") == QueryIntent.IMPLEMENTATION
    assert detect_query_intent("test assertions and specification for trigger route") == QueryIntent.SPECIFICATION
    assert detect_query_intent("fix panic nil pointer error in stream handler") == QueryIntent.BUG_DIAGNOSIS
    assert detect_query_intent("refactor and extract module helpers") == QueryIntent.REFACTOR


def test_default_intent_thresholds():
    assert DEFAULT_INTENT_THRESHOLDS[QueryIntent.IMPLEMENTATION] == 0.65
    assert DEFAULT_INTENT_THRESHOLDS[QueryIntent.SPECIFICATION] == 0.60
    assert DEFAULT_INTENT_THRESHOLDS[QueryIntent.BUG_DIAGNOSIS] == 0.60
    assert DEFAULT_INTENT_THRESHOLDS[QueryIntent.REFACTOR] == 0.65


def test_percentile_calculation():
    # Empty
    assert calculate_percentile([]) == 0.0

    # Single element
    assert calculate_percentile([10.0], 95.0) == 10.0

    # 100 elements 1..100
    vals = list(range(1, 101))
    p95 = calculate_percentile(vals, 95.0)
    assert 94.0 <= p95 <= 96.0


def test_is_doc_chunk():
    doc_chunk = CandidateChunk(
        id="d1",
        file_path="docs/architecture.md",
        content="1: # Architecture",
        start_line=1,
        end_line=1,
    )
    code_chunk = CandidateChunk(
        id="c1",
        file_path="src/service.py",
        content="1: def run(): pass",
        start_line=1,
        end_line=1,
    )
    assert is_doc_chunk(doc_chunk) is True
    assert is_doc_chunk(code_chunk) is False


def test_implementation_intent_penalizes_test_chunks():
    async def _test():
        reranker = LFMReranker()

        with tempfile.TemporaryDirectory() as tmpdir:
            f_auth = Path(tmpdir) / "auth.py"
            f_auth.write_text("def auth(): pass\n")
            f_test = Path(tmpdir) / "test_auth.py"
            f_test.write_text("def test_auth(): pass\n")

            chunks = [
                CandidateChunk(
                    id="impl_1",
                    file_path=str(f_auth),
                    content="1: def auth(): pass",
                    start_line=1,
                    end_line=1,
                    is_test=False,
                ),
                CandidateChunk(
                    id="test_1",
                    file_path=str(f_test),
                    content="1: def test_auth(): pass",
                    start_line=1,
                    end_line=1,
                    is_test=True,
                ),
            ]

            async def mock_score(client, sem, query, chunk):
                return 0.70, -0.35, -1.2, False, True, False, 100, 1, 15.0

            with patch.object(reranker, "_score_chunk_uncached", side_effect=mock_score):
                resp = await reranker.rerank_chunks(
                    query="authenticate user request",
                    chunks=chunks,
                    intent=QueryIntent.IMPLEMENTATION,
                )

                assert resp.intent == QueryIntent.IMPLEMENTATION
                assert resp.threshold == 0.65

                # impl chunk should keep score 0.70
                impl_res = next(r for r in resp.results if r.candidate_id == "impl_1")
                assert impl_res.score == 0.70

                # test chunk should be penalized: 0.70 * 0.8 = 0.56
                test_res = next(r for r in resp.results if r.candidate_id == "test_1")
                assert test_res.score == 0.56
                # Top item should be implementation chunk
                assert resp.results[0].candidate_id == "impl_1"

    asyncio.run(_test())


def test_specification_intent_boosts_test_and_doc_chunks():
    async def _test():
        reranker = LFMReranker()

        with tempfile.TemporaryDirectory() as tmpdir:
            f_auth = Path(tmpdir) / "auth.py"
            f_auth.write_text("def auth(): pass\n")
            f_test = Path(tmpdir) / "test_auth.py"
            f_test.write_text("def test_auth(): pass\n")

            chunks = [
                CandidateChunk(
                    id="impl_1",
                    file_path=str(f_auth),
                    content="1: def auth(): pass",
                    start_line=1,
                    end_line=1,
                    is_test=False,
                ),
                CandidateChunk(
                    id="test_1",
                    file_path=str(f_test),
                    content="1: def test_auth(): pass",
                    start_line=1,
                    end_line=1,
                    is_test=True,
                ),
            ]

            # Both raw scores 0.60
            async def mock_score(client, sem, query, chunk):
                return 0.60, -0.5, -0.5, False, True, False, 100, 1, 12.0

            with patch.object(reranker, "_score_chunk_uncached", side_effect=mock_score):
                resp = await reranker.rerank_chunks(
                    query="test assertion specifications for auth",
                    chunks=chunks,
                    intent=QueryIntent.SPECIFICATION,
                )

                assert resp.intent == QueryIntent.SPECIFICATION
                assert resp.threshold == 0.60

                test_res = next(r for r in resp.results if r.candidate_id == "test_1")
                # Boosted by 25%: 0.60 * 1.25 = 0.75
                assert test_res.score == 0.75
                assert resp.results[0].candidate_id == "test_1"

    asyncio.run(_test())


def test_bug_diagnosis_always_includes_top_test_chunk():
    async def _test():
        reranker = LFMReranker()

        with tempfile.TemporaryDirectory() as tmpdir:
            files = {}
            for name in ["a.py", "b.py", "c.py", "d.py", "test_a.py"]:
                p = Path(tmpdir) / name
                p.write_text(f"# {name}\n")
                files[name] = str(p)

            chunks = [
                CandidateChunk(id="c1", file_path=files["a.py"], content="1: a", start_line=1, end_line=1, is_test=False),
                CandidateChunk(id="c2", file_path=files["b.py"], content="1: b", start_line=1, end_line=1, is_test=False),
                CandidateChunk(id="c3", file_path=files["c.py"], content="1: c", start_line=1, end_line=1, is_test=False),
                CandidateChunk(id="c4", file_path=files["d.py"], content="1: d", start_line=1, end_line=1, is_test=False),
                CandidateChunk(id="t1", file_path=files["test_a.py"], content="1: t", start_line=1, end_line=1, is_test=True),
            ]

            scores = {
                "c1": 0.95,
                "c2": 0.90,
                "c3": 0.85,
                "c4": 0.80,
                "t1": 0.45,  # Below threshold 0.60
            }

            async def mock_score(client, sem, query, chunk):
                return scores[chunk.id], -0.2, -1.5, False, True, False, 80, 1, 10.0

            with patch.object(reranker, "_score_chunk_uncached", side_effect=mock_score):
                resp = await reranker.rerank_chunks(
                    query="investigate bug in stream processor",
                    chunks=chunks,
                    top_k=3,
                    intent=QueryIntent.BUG_DIAGNOSIS,
                )

                assert resp.intent == QueryIntent.BUG_DIAGNOSIS
                assert resp.threshold == 0.60

                # top_k is 3, and t1 must be included even though it had score 0.45 (< 0.60)
                result_ids = [r.candidate_id for r in resp.results]
                assert "t1" in result_ids
                assert len(resp.results) == 3
                t1_item = next(r for r in resp.results if r.candidate_id == "t1")
                assert t1_item.below_threshold is True

    asyncio.run(_test())


def test_top_3_floor_with_below_threshold_flags():
    async def _test():
        reranker = LFMReranker()

        with tempfile.TemporaryDirectory() as tmpdir:
            files = {}
            for name in ["a.py", "b.py", "c.py", "d.py"]:
                p = Path(tmpdir) / name
                p.write_text(f"# {name}\n")
                files[name] = str(p)

            chunks = [
                CandidateChunk(id="c1", file_path=files["a.py"], content="1: a", start_line=1, end_line=1),
                CandidateChunk(id="c2", file_path=files["b.py"], content="1: b", start_line=1, end_line=1),
                CandidateChunk(id="c3", file_path=files["c.py"], content="1: c", start_line=1, end_line=1),
                CandidateChunk(id="c4", file_path=files["d.py"], content="1: d", start_line=1, end_line=1),
            ]

            # Only c1 clears threshold 0.65
            scores = {"c1": 0.80, "c2": 0.50, "c3": 0.40, "c4": 0.20}

            async def mock_score(client, sem, query, chunk):
                return scores[chunk.id], -0.3, -1.0, False, True, False, 90, 1, 10.0

            with patch.object(reranker, "_score_chunk_uncached", side_effect=mock_score):
                resp = await reranker.rerank_chunks(
                    query="query",
                    chunks=chunks,
                    threshold=0.65,
                )

                # Top 3 floor maintained
                assert len(resp.results) == 3
                assert resp.telemetry.fallback_floor_triggered is True

                c1 = next(r for r in resp.results if r.candidate_id == "c1")
                assert c1.below_threshold is False

                c2 = next(r for r in resp.results if r.candidate_id == "c2")
                assert c2.below_threshold is True

                c3 = next(r for r in resp.results if r.candidate_id == "c3")
                assert c3.below_threshold is True

    asyncio.run(_test())


def test_operational_telemetry_tracking_fields():
    async def _test():
        with tempfile.TemporaryDirectory() as tmpdir:
            from slm_rerank.cache import RerankCache
            cache = RerankCache(db_path=Path(tmpdir) / "cache.db")
            reranker = LFMReranker(cache=cache)

            chunks = [
                CandidateChunk(id="c1", file_path=None, content="1: alpha_fn", start_line=1, end_line=1),
                CandidateChunk(id="c2", file_path=None, content="1: beta_fn", start_line=1, end_line=1),
                CandidateChunk(id="c3", file_path=None, content="1: gamma_fn", start_line=1, end_line=1),
                CandidateChunk(id="c4", file_path=None, content="1: delta_fn", start_line=1, end_line=1),
            ]

            # c1: yes variant (15ms)
            # c2: yes variant (25ms)
            # c3: no variant (35ms)
            # c4: ambiguous (50ms)
            returns = {
                "c1": (0.90, -0.1, -2.5, False, True, False, 100, 1, 15.0),
                "c2": (0.85, -0.2, -2.0, False, True, False, 110, 1, 25.0),
                "c3": (0.10, -2.5, -0.1, False, False, True, 105, 1, 35.0),
                "c4": (0.05, None, None, True, False, False, 120, 1, 50.0),
            }

            async def mock_score(client, sem, query, chunk):
                return returns[chunk.id]

            with patch.object(reranker, "_score_chunk_uncached", side_effect=mock_score):
                # 1. Cold pass
                resp = await reranker.rerank_chunks(
                    query="cold query telemetry",
                    chunks=chunks,
                )

                t = resp.telemetry
                assert t.chunks_evaluated == 4
                assert t.ambiguous_completions == 1
                assert t.ambiguous_completion_rate == 0.25
                assert t.yes_variant_rate == 0.50  # 2 out of 4
                assert t.no_variant_rate == 0.25   # 1 out of 4
                assert t.cache_hit_rate == 0.0     # 0 cached out of 4
                assert t.score_latency_p95 > 35.0  # p95 among 15, 25, 35, 50

                # 2. Warm pass (non-ambiguous c1, c2, c3 cached)
                resp_warm = await reranker.rerank_chunks(
                    query="cold query telemetry",
                    chunks=chunks[:3],
                )
                assert resp_warm.telemetry.cache_hits == 3
                assert resp_warm.telemetry.cache_hit_rate == 1.0

    asyncio.run(_test())


def test_apply_intent_prior_log_odds():
    """Verify log-odds prior formulation in logit space."""
    import math
    from slm_rerank.client import apply_intent_prior

    # 1. Zero delta leaves probability unchanged
    assert abs(apply_intent_prior(0.50, 0.0) - 0.50) < 1e-5
    assert abs(apply_intent_prior(0.70, 0.0) - 0.70) < 1e-5

    # 2. Confidence = 0.0 leaves probability unchanged
    assert abs(apply_intent_prior(0.80, 0.50, confidence=0.0) - 0.80) < 1e-5

    # 3. Positive delta increases probability
    boosted = apply_intent_prior(0.60, 0.693, confidence=1.0)
    assert boosted > 0.60
    assert abs(boosted - 0.75) < 0.01

    # 4. Negative delta decreases probability
    penalized = apply_intent_prior(0.70, -0.606, confidence=1.0)
    assert penalized < 0.70
    assert abs(penalized - 0.56) < 0.01

    # 5. Half confidence scales logit delta
    half_boosted = apply_intent_prior(0.60, 0.693, confidence=0.5)
    assert 0.60 < half_boosted < boosted


def test_intent_confidence_heuristic_and_scaling():
    from slm_rerank.client import detect_query_intent_with_confidence

    # High confidence: multiple keywords
    intent, conf = detect_query_intent_with_confidence("fix bug and crash error in handler")
    assert intent == QueryIntent.BUG_DIAGNOSIS
    assert conf == 1.0

    intent, conf = detect_query_intent_with_confidence("test assertions and specification contracts")
    assert intent == QueryIntent.SPECIFICATION
    assert conf == 1.0

    # Medium confidence: single keyword
    intent, conf = detect_query_intent_with_confidence("verify token")
    assert intent == QueryIntent.SPECIFICATION
    assert conf == 0.5

    # Low confidence: no keywords match, defaults to IMPLEMENTATION with 0.0 confidence
    intent, conf = detect_query_intent_with_confidence("xyz abc 123")
    assert intent == QueryIntent.IMPLEMENTATION
    assert conf == 0.0


def test_dual_score_architecture_and_fallback_metadata():
    async def _test():
        reranker = LFMReranker()

        with tempfile.TemporaryDirectory() as tmpdir:
            f_a = Path(tmpdir) / "service.py"
            f_a.write_text("def run(): pass\n")
            f_b = Path(tmpdir) / "test_service.py"
            f_b.write_text("def test_run(): pass\n")

            chunks = [
                CandidateChunk(id="c1", file_path=str(f_a), content="1: def run(): pass", start_line=1, end_line=1, is_test=False),
                CandidateChunk(id="c2", file_path=str(f_b), content="1: def test_run(): pass", start_line=1, end_line=1, is_test=True),
            ]

            scores = {"c1": 0.80, "c2": 0.40}
            async def mock_score(client, sem, query, chunk):
                return scores[chunk.id], -0.2, -1.5, False, True, False, 80, 1, 10.0

            with patch.object(reranker, "_score_chunk_uncached", side_effect=mock_score):
                resp = await reranker.rerank_chunks(
                    query="implement service runner",
                    chunks=chunks,
                    threshold=0.65,
                )

                # Check c1 (passed threshold 0.65)
                c1 = next(r for r in resp.results if r.candidate_id == "c1")
                assert c1.raw_score == 0.80
                assert c1.adjusted_score == 0.80  # Non-test in implementation
                assert c1.delta == 0.0
                assert c1.below_threshold is False
                assert c1.metadata["included_by_fallback"] is False

                # Check c2 (retained by fallback floor, penalized by intent)
                c2 = next(r for r in resp.results if r.candidate_id == "c2")
                assert c2.raw_score == 0.40
                assert c2.adjusted_score < 0.40  # Test chunk penalized in IMPLEMENTATION
                assert c2.delta < 0.0
                assert c2.below_threshold is True
                assert c2.metadata["included_by_fallback"] is True
                assert c2.metadata["below_threshold"] is True
                assert "fallback_warning" in c2.metadata

    asyncio.run(_test())


def test_abstention_when_top_score_below_threshold():
    async def _test():
        reranker = LFMReranker()

        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "misc.py"
            f.write_text("def misc(): pass\n")

            chunks = [
                CandidateChunk(id="c1", file_path=str(f), content="1: def misc(): pass", start_line=1, end_line=1),
                CandidateChunk(id="c2", file_path=str(f), content="2: def other(): pass", start_line=1, end_line=1),
            ]

            # All candidate scores very low (< 0.20)
            async def mock_score(client, sem, query, chunk):
                return 0.12, -2.5, -0.1, False, False, True, 80, 1, 10.0

            with patch.object(reranker, "_score_chunk_uncached", side_effect=mock_score):
                # 1. Default without allow_empty maintains fallback floor
                resp_default = await reranker.rerank_chunks(
                    query="completely irrelevant query",
                    chunks=chunks,
                    allow_empty_if_low_confidence=False,
                )
                assert len(resp_default.results) == 2
                assert resp_default.telemetry.abstained is False

                # 2. With allow_empty_if_low_confidence=True, abstains below 0.20
                resp_abstain = await reranker.rerank_chunks(
                    query="completely irrelevant query",
                    chunks=chunks,
                    allow_empty_if_low_confidence=True,
                    abstain_below=0.20,
                )
                assert len(resp_abstain.results) == 0
                assert resp_abstain.telemetry.abstained is True

    asyncio.run(_test())

