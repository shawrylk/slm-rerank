"""Comprehensive Unit & Regression Tests for LFM Wide Reranker v0.3.3 Features.

Covers:
1. Calibration Layer (TemperatureScalingCalibrator and PlattScalingCalibrator with CV)
2. 3-Score Tracking Architecture (raw_score, decision_score, calibrated_score)
3. Ambiguity Recovery and Telemetry (clean_token_str, AmbiguityEvent)
4. Symbol-Aware Boost & Sibling Disambiguation (compute_symbol_match_delta)
5. Diversity-Aware Context Assembly (apply_diversity_context_assembly MMR penalty)
6. Dynamic Margin-Based Thresholding & Top-3 Fallback Floor
7. Bootstrap Confidence Interval Calculation (calculate_bootstrap_confidence_intervals)
"""

import math
import numpy as np
import pytest

from slm_rerank.adapters import clean_token_str, YES_VARIANTS, NO_VARIANTS
from slm_rerank.calibration import (
    BaseCalibrator,
    PlattScalingCalibrator,
    TemperatureScalingCalibrator,
    get_default_calibrator,
    logit,
    sigmoid,
)
from slm_rerank.client import (
    LFMReranker,
    apply_diversity_context_assembly,
    apply_intent_prior,
    compute_symbol_match_delta,
    extract_code_identifiers,
)
from slm_rerank.eval import (
    BootstrapCI,
    calculate_brier_score,
    calculate_ece,
    calculate_bootstrap_confidence_intervals,
    EvalTaskResult,
)
from slm_rerank.models import (
    AmbiguityEvent,
    CandidateChunk,
    Citation,
    GroundTruthStatus,
    QueryIntent,
    RerankResultItem,
    Telemetry,
)


# ==============================================================================
# 1. Calibration Layer Tests
# ==============================================================================

def test_sigmoid_and_logit_numerical_stability():
    """Verify numerical stability of sigmoid and logit helpers with extreme values."""
    assert sigmoid(0.0) == 0.5
    assert sigmoid(100.0) > 0.99999
    assert sigmoid(-100.0) < 0.00001

    assert math.isclose(logit(0.5), 0.0, abs_tol=1e-5)
    # Check extreme clamping bounds
    assert logit(1.0) < 20.0
    assert logit(0.0) > -20.0


def test_temperature_scaling_calibrator_fit_and_calibrate():
    """Test TemperatureScalingCalibrator parameter optimization and calibration."""
    logits = [-2.0, -1.0, 0.0, 1.0, 2.0]
    y_true = [0, 0, 0, 1, 1]

    cal = TemperatureScalingCalibrator(temperature=1.0)
    assert cal.calibrate(0.0) == 0.5

    cal.fit(logits, y_true)
    assert cal.temperature > 0.0
    cal_prob = cal.calibrate_proba(0.731)  # sigmoid(1.0) ~ 0.731
    assert 0.0 <= cal_prob <= 1.0


def test_temperature_scaling_kfold_cv():
    """Test K-fold cross validation on TemperatureScalingCalibrator."""
    rng = np.random.RandomState(42)
    logits = rng.randn(30).tolist()
    y_true = [1 if z > 0.0 else 0 for z in logits]

    cal = TemperatureScalingCalibrator()
    cal.fit_cv(logits, y_true, cv=3)
    assert cal.temperature > 0.0


def test_platt_scaling_calibrator_fit_and_calibrate():
    """Test PlattScalingCalibrator Newton-Raphson fitting and monotonicity constraint."""
    logits = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    y_true = [0, 0, 0, 0, 1, 1, 1]

    cal = PlattScalingCalibrator()
    cal.fit(logits, y_true, l2_reg=1e-3)

    assert cal.a > 0.0  # Monotonicity preserved
    # Check predictions are monotonic with logits
    p_low = cal.calibrate(-2.0)
    p_high = cal.calibrate(2.0)
    assert p_high > p_low


def test_platt_scaling_kfold_cv():
    """Test PlattScalingCalibrator K-fold cross validation with L2 regularization."""
    logits = [-3.0, -2.0, -1.5, -0.5, 0.5, 1.5, 2.0, 3.0] * 3
    y_true = [0, 0, 0, 0, 1, 1, 1, 1] * 3

    cal = PlattScalingCalibrator()
    cal.fit_cv(logits, y_true, cv=3)
    assert cal.a > 0.0
    probs = cal.predict_proba([-1.0, 1.0])
    assert len(probs) == 2
    assert probs[1] > probs[0]


def test_default_calibrator_singleton():
    """Verify get_default_calibrator returns configured pre-calibrated Platt scaler."""
    cal = get_default_calibrator()
    assert isinstance(cal, PlattScalingCalibrator)
    assert cal.a > 0.0


# ==============================================================================
# 2. 3-Score Architecture Tracking
# ==============================================================================

def test_three_score_architecture_fields():
    """Verify RerankResultItem maintains 3 distinct scores and backward compatibility aliases."""
    citation = Citation(file="src/service.ts", start_line=10, end_line=25, symbol="handleRequest")
    item = RerankResultItem(
        candidate_id="cand_1",
        file_path="src/service.ts",
        raw_score=0.75,
        decision_score=0.82,
        calibrated_score=0.78,
        adjusted_score=0.82,
        score=0.82,
        delta=0.07,
        citation=citation,
        symbol="handleRequest",
    )

    assert item.raw_score == 0.75
    assert item.decision_score == 0.82
    assert item.calibrated_score == 0.78
    assert item.adjusted_score == 0.82
    assert item.score == 0.82
    assert item.delta == 0.07


# ==============================================================================
# 3. Ambiguity Token Recovery & Telemetry Tests
# ==============================================================================

def test_clean_token_str_strips_punctuation_and_whitespace():
    """Test token cleaning removes punctuation wrappers and leading/trailing hazards."""
    assert clean_token_str(" yes") == "yes"
    assert clean_token_str("Yes.") == "yes"
    assert clean_token_str("'yes'") == "yes"
    assert clean_token_str('"no"') == "no"
    assert clean_token_str("ĠYes") == "yes"
    assert clean_token_str(" No:") == "no"
    assert clean_token_str("  1  ") == "1"
    assert clean_token_str("+1") == "+1"
    assert clean_token_str("-1") == "-1"

    assert clean_token_str("Yes") in YES_VARIANTS
    assert clean_token_str("1") in YES_VARIANTS
    assert clean_token_str("No") in NO_VARIANTS
    assert clean_token_str("0") in NO_VARIANTS


def test_ambiguity_event_model():
    """Verify AmbiguityEvent structure tracks full prompt context and candidate logprobs."""
    event = AmbiguityEvent(
        token_text="Sure",
        token_id=12345,
        logprob=-1.25,
        prompt_suffix="Is this code relevant? Answer:",
        prompt_state="unrecognized_token",
        chunk_id="chunk_test",
        reason="Emitted non-binary token",
        top_candidates=[{"token": "Sure", "logprob": -1.25}, {"token": "Yes", "logprob": -2.10}],
    )
    assert event.token_text == "Sure"
    assert event.token_id == 12345
    assert len(event.top_candidates) == 2


# ==============================================================================
# 4. Symbol-Aware Boost & Sibling Disambiguation Tests
# ==============================================================================

def test_extract_code_identifiers():
    """Test extraction of camelCase, PascalCase, snake_case, and backticked identifiers."""
    query = "Find `parseConfig` and verify llm_arch_from_string with ProgressWorkTypes"
    ids = extract_code_identifiers(query)
    assert "parseConfig" in ids
    assert "llm_arch_from_string" in ids
    assert "ProgressWorkTypes" in ids


def test_compute_symbol_match_delta_single_word():
    """Test single-word symbol match boost (+1.40 for 'trigger')."""
    chunk = CandidateChunk(
        id="c1",
        file_path="src/features/progress/trigger.ts",
        symbol="trigger",
        content="export const trigger = async () => {};",
    )
    query = "where is the progress trigger handler and route definitions"
    delta, is_match, syms = compute_symbol_match_delta(chunk, query, [], QueryIntent.IMPLEMENTATION)
    assert is_match is True
    assert delta >= 1.40
    assert "trigger" in syms


def test_compute_symbol_match_delta_semantic_decomposition():
    """Test multi-component semantic decomposition boost (+1.60 for llm_arch_from_string)."""
    chunk = CandidateChunk(
        id="c2",
        file_path="src/llama-arch.cpp",
        symbol="llm_arch_from_string",
        content="llm_arch llm_arch_from_string(const std::string & name) { ... }",
    )
    query = "LLM architecture identification, type mapping, and string lookup"
    delta, is_match, syms = compute_symbol_match_delta(chunk, query, [], QueryIntent.IMPLEMENTATION)
    assert is_match is True
    assert delta >= 1.60
    assert "llm_arch_from_string" in syms


def test_compute_symbol_match_delta_schema_refactor():
    """Test schema refactor table boost (+0.50) and test penalty (-0.35)."""
    schema_chunk = CandidateChunk(
        id="c3",
        file_path="src/features/progress/schema.ts",
        symbol="progressWorkTypes",
        content="export const progressWorkTypes = pgTable('progress_work_types', { ... });",
        is_test=False,
    )
    test_chunk = CandidateChunk(
        id="c4",
        file_path="src/features/progress/pipeline.test.ts",
        symbol="createFakeTx",
        content="describe('progress pipeline', () => { ... });",
        is_test=True,
    )
    query = "refactor and restructure progress database table schema definitions and columns"

    delta_schema, is_match, _ = compute_symbol_match_delta(schema_chunk, query, [], QueryIntent.REFACTOR)
    assert is_match is True
    assert delta_schema >= 0.50

    delta_test, _, _ = compute_symbol_match_delta(test_chunk, query, [], QueryIntent.REFACTOR)
    assert delta_test <= -0.35


# ==============================================================================
# 5. Diversity-Aware Context Assembly (MMR Redundancy Penalty) Tests
# ==============================================================================

def test_apply_diversity_context_assembly_penalizes_siblings():
    """Verify MMR redundancy penalty applies to second chunk from same file unless exceptionally high."""
    citation_1 = Citation(file="src/schema.ts", start_line=1, end_line=10, symbol="tableA")
    citation_2 = Citation(file="src/schema.ts", start_line=20, end_line=30, symbol="tableB")
    citation_3 = Citation(file="src/routes.ts", start_line=1, end_line=10, symbol="routeA")

    item1 = RerankResultItem(candidate_id="1", file_path="src/schema.ts", raw_score=0.85, decision_score=0.85, score=0.85, citation=citation_1)
    item2 = RerankResultItem(candidate_id="2", file_path="src/schema.ts", raw_score=0.84, decision_score=0.84, score=0.84, citation=citation_2)
    item3 = RerankResultItem(candidate_id="3", file_path="src/routes.ts", raw_score=0.83, decision_score=0.83, score=0.83, citation=citation_3)

    items = [item1, item2, item3]
    assembled, count_penalties = apply_diversity_context_assembly(
        items,
        redundancy_logit_penalty=0.20,
        exceptionally_high_threshold=0.88,
    )

    assert count_penalties > 0
    # item1 should be first
    assert assembled[0].candidate_id == "1"
    # item3 (from routes.ts) should jump ahead of item2 (from schema.ts, penalized)
    assert assembled[1].candidate_id == "3"
    assert assembled[2].candidate_id == "2"
    assert assembled[2].redundancy_penalized is True


# ==============================================================================
# 6. Dynamic Margin-Based Thresholding Tests
# ==============================================================================

@pytest.mark.anyio
async def test_dynamic_margin_thresholding_logic():
    """Verify client includes chunks if decision_score >= (top_score - margin) AND decision_score >= threshold."""
    reranker = LFMReranker(margin=0.15)

    c1 = CandidateChunk(id="c1", content="let a = 1;", file_path="a.ts", symbol="a")
    c2 = CandidateChunk(id="c2", content="let b = 2;", file_path="b.ts", symbol="b")
    c3 = CandidateChunk(id="c3", content="let c = 3;", file_path="c.ts", symbol="c")
    c4 = CandidateChunk(id="c4", content="let d = 4;", file_path="d.ts", symbol="d")

    # Mock cache put to return controlled scores
    reranker.cache.put_batch(
        query="test query",
        entries=[
            (c1, 0.90, None, None),
            (c2, 0.80, None, None),  # Within margin (0.90 - 0.15 = 0.75) and >= threshold 0.65
            (c3, 0.70, None, None),  # Below margin floor (0.75), but kept by top-3 floor
            (c4, 0.50, None, None),  # Excluded
        ],
        model_id=reranker.profile.name,
        prompt_version="v1",
        query_intent="IMPLEMENTATION",
        prior_version="v0.3.2",
        length_exponent=reranker.profile.length_normalization_exponent,
    )

    resp = await reranker.rerank_chunks(
        query="test query",
        chunks=[c1, c2, c3, c4],
        threshold=0.65,
        intent=QueryIntent.IMPLEMENTATION,
    )

    # Top-3 floor retains 3 items
    assert len(resp.results) == 3
    assert resp.telemetry.margin_threshold_used == 0.15


# ==============================================================================
# 7. Bootstrap Confidence Interval Tests
# ==============================================================================

def test_calculate_bootstrap_confidence_intervals():
    """Test 95% bootstrap confidence interval calculation produces valid lower/upper bounds."""
    task_results = [
        EvalTaskResult(task_id="T1", query="q1", target_file="f1.ts", found=True, rank=1, reciprocal_rank=1.0),
        EvalTaskResult(task_id="T2", query="q2", target_file="f2.ts", found=True, rank=2, reciprocal_rank=0.5),
        EvalTaskResult(task_id="T3", query="q3", target_file="f3.ts", found=True, rank=4, reciprocal_rank=0.25),
        EvalTaskResult(task_id="T4", query="q4", target_file="f4.ts", found=False, rank=None, reciprocal_rank=0.0),
    ]

    task_chunks = [
        {"y_true": [1.0, 0.0], "y_prob_decision": [0.8, 0.2], "y_prob_calibrated": [0.75, 0.15]},
        {"y_true": [1.0, 0.0], "y_prob_decision": [0.7, 0.3], "y_prob_calibrated": [0.65, 0.25]},
        {"y_true": [1.0, 0.0], "y_prob_decision": [0.5, 0.5], "y_prob_calibrated": [0.45, 0.45]},
        {"y_true": [1.0, 0.0], "y_prob_decision": [0.3, 0.7], "y_prob_calibrated": [0.25, 0.65]},
    ]

    cis = calculate_bootstrap_confidence_intervals(task_results, task_chunks, n_resamples=500, seed=42)

    assert "recall_at_1" in cis
    assert "recall_at_3" in cis
    assert "mrr" in cis
    assert "brier_score" in cis
    assert "calibrated_brier_score" in cis

    r3_ci = cis["recall_at_3"]
    assert 0.0 <= r3_ci.ci_lower <= r3_ci.mean <= r3_ci.ci_upper <= 1.0
