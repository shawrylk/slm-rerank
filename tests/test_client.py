"""Unit tests for binary logprob scoring, hazard protection, and prompt construction."""

import pytest
from slm_rerank.client import (
    extract_binary_logprobs,
    extract_calibrated_logprobs,
    logsumexp,
    build_binary_prompt,
)
from slm_rerank.models import CandidateChunk


def test_logsumexp():
    lps = [-0.5, -1.2, -3.0]
    res = logsumexp(lps)
    assert res is not None
    assert res > -0.5  # logsumexp must be greater than max


def test_extract_binary_logprobs_balanced():
    top_logprobs = [
        {"id": 11683, "token": "yes", "logprob": -0.338},
        {"id": 2243, "token": "no", "logprob": -1.495},
    ]
    prob, lp_yes, lp_no = extract_binary_logprobs(top_logprobs, chunk_token_est=120)
    assert 0.70 <= prob <= 0.80
    assert lp_yes == -0.338
    assert lp_no == -1.495


def test_hazard_protection_ambiguous_completion():
    # Top-1 token is punctuation/newline or refusal instead of yes/no
    top_logprobs = [
        {"id": 597, "token": "The", "logprob": -0.10},
        {"id": 2718, "token": "{\n", "logprob": -1.50},
        {"id": 11683, "token": "yes", "logprob": -7.50},
        {"id": 2243, "token": "no", "logprob": -8.00},
    ]
    score, lp_yes, lp_no, is_ambiguous = extract_calibrated_logprobs(top_logprobs, chunk_token_est=100)
    assert is_ambiguous is True
    assert score <= 0.10  # Low baseline score instead of noise renormalization!


def test_extract_binary_logprobs_case_variants():
    top_logprobs = [
        {"id": 11683, "token": "yes", "logprob": -0.40},
        {"id": 12447, "token": "Yes", "logprob": -2.00},
        {"id": 2243, "token": "no", "logprob": -1.80},
        {"id": 4547, "token": "No", "logprob": -3.50},
    ]
    prob, lp_yes, lp_no = extract_binary_logprobs(top_logprobs, chunk_token_est=120)
    assert prob > 0.70


def test_extract_binary_logprobs_no_dominant():
    top_logprobs = [
        {"id": 2243, "token": "no", "logprob": -0.05},
        {"id": 4547, "token": "No", "logprob": -3.20},
    ]
    prob, lp_yes, lp_no = extract_binary_logprobs(top_logprobs, chunk_token_est=120)
    assert prob <= 0.10


def test_build_binary_prompt_with_test_awareness():
    test_chunk = CandidateChunk(
        id="t1",
        file_path="src/features/auth.test.ts",
        start_line=1,
        end_line=20,
        symbol="describe(auth)",
        content="1: describe('auth', () => {})",
        is_test=True,
    )
    prompt = build_binary_prompt("verify token expiration error", test_chunk)
    assert "<think>\n</think>" in prompt
    assert "authoritative test/specification suite" in prompt
    assert "File: src/features/auth.test.ts" in prompt
