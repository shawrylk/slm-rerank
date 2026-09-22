"""Comprehensive unit tests for ModelProfile/ProviderAdapter architecture, dynamic tokens, cache isolation, and auto-detection."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from slm_rerank.adapters import (
    GenericOpenAIProfile,
    GemmaProfile,
    LFMProfile,
    ModelProfile,
    ProviderAdapter,
    QwenProfile,
    RWKVProfile,
    clean_token_str,
    detect_profile_from_model_names,
    fallback_profile_from_url,
    get_profile,
    normalize_top_logprobs,
    probe_and_detect_profile_sync,
    probe_tokenizer_tokens_sync,
)
from slm_rerank.cache import RerankCache
from slm_rerank.cli import parse_args
from slm_rerank.client import LFMReranker
from slm_rerank.config import load_config, resolve_endpoint_and_model, save_config


# ---------------------------------------------------------------------------
# 1. Base interface & built-in profiles properties
# ---------------------------------------------------------------------------


def test_provider_adapter_alias():
    assert ProviderAdapter is ModelProfile


def test_builtin_profile_properties():
    profiles = {
        "lfm": (LFMProfile(), 0.15, "http://localhost:8034/v1"),
        "qwen": (QwenProfile(), 0.12, "http://localhost:8033/v1"),
        "gemma": (GemmaProfile(), 0.12, "http://localhost:11434/v1"),
        "rwkv": (RWKVProfile(), 0.10, "http://localhost:8000/v1"),
        "openai": (GenericOpenAIProfile(), 0.12, "http://localhost:11434/v1"),
    }

    for name, (prof, exp, url) in profiles.items():
        assert prof.name == name
        assert prof.length_normalization_exponent == exp
        assert prof.default_base_url == url


def test_get_profile_factory_and_aliases():
    assert isinstance(get_profile("lfm"), LFMProfile)
    assert isinstance(get_profile("lfm-2.5"), LFMProfile)
    assert isinstance(get_profile("liquid"), LFMProfile)

    qwen = get_profile("qwen")
    assert isinstance(qwen, QwenProfile)
    assert qwen.is_qwq is False

    qwq = get_profile("qwq")
    assert isinstance(qwq, QwenProfile)
    assert qwq.is_qwq is True

    assert isinstance(get_profile("gemma"), GemmaProfile)
    assert isinstance(get_profile("gemma2"), GemmaProfile)

    assert isinstance(get_profile("rwkv"), RWKVProfile)
    assert isinstance(get_profile("rwkv6"), RWKVProfile)

    assert isinstance(get_profile("openai"), GenericOpenAIProfile)
    assert isinstance(get_profile("ollama"), GenericOpenAIProfile)
    assert isinstance(get_profile("vllm"), GenericOpenAIProfile)


# ---------------------------------------------------------------------------
# 2. Prompt Formatting Across All 5 Profiles
# ---------------------------------------------------------------------------


def test_lfm_prompt_formatting():
    profile = LFMProfile()
    prompt = profile.format_prompt(
        query="implement rate limiter",
        chunk_content="def limit(): pass",
        file_path="src/limiter.py",
        symbol="limit",
        is_test=True,
        start_line=10,
        end_line=25,
    )

    assert "<|startoftext|><|im_start|>system" in prompt
    assert "<think>\n</think>" in prompt  # Think bypass for LFM
    assert "File: src/limiter.py" in prompt
    assert "Lines: 10-25" in prompt
    assert "Symbol: limit" in prompt
    assert "authoritative test/specification suite" in prompt
    assert "Code:\ndef limit(): pass" in prompt
    assert prompt.endswith("<think>\n</think>\n")


def test_qwen_prompt_formatting_standard_and_qwq():
    qwen_std = QwenProfile(is_qwq=False)
    p_std = qwen_std.format_prompt(
        query="fix buffer overflow",
        chunk_content="char buf[10];",
        file_path="src/net.c",
        is_test=False,
    )
    assert "<|im_start|>system" in p_std
    assert "<|im_start|>assistant\n" in p_std
    assert "<think>" not in p_std  # No think tags for standard Qwen ChatML

    qwen_qwq = QwenProfile(is_qwq=True)
    p_qwq = qwen_qwq.format_prompt(
        query="fix buffer overflow",
        chunk_content="char buf[10];",
        file_path="src/net.c",
        is_test=False,
    )
    assert "<think>\n</think>" in p_qwq  # Think tags present for QwQ reasoning model


def test_gemma_prompt_formatting():
    profile = GemmaProfile()
    prompt = profile.format_prompt(
        query="user auth session",
        chunk_content="class Session: pass",
        file_path="auth/session.py",
        symbol="Session",
        is_test=False,
    )

    assert "<start_of_turn>user" in prompt
    assert "<end_of_turn>" in prompt
    assert "<start_of_turn>model" in prompt
    assert "auth/session.py" in prompt
    assert "Session" in prompt
    assert "<|im_start|>" not in prompt  # Gemma format does not use ChatML


def test_rwkv_prompt_formatting():
    profile = RWKVProfile()
    prompt = profile.format_prompt(
        query="find linear state attention",
        chunk_content="def rwkv_attention(): pass",
        file_path="model/rwkv.py",
        is_test=False,
    )

    assert "User: " in prompt
    assert "Assistant: " in prompt
    assert "<|im_start|>" not in prompt
    assert "<start_of_turn>" not in prompt
    assert "def rwkv_attention(): pass" in prompt


def test_generic_openai_prompt_formatting():
    profile = GenericOpenAIProfile()
    prompt = profile.format_prompt(
        query="database migration",
        chunk_content="ALTER TABLE users ADD COLUMN age INT;",
        file_path="migrations/001.sql",
        is_test=False,
    )

    assert "<|im_start|>system" in prompt
    assert "<|im_start|>user" in prompt
    assert "<|im_start|>assistant" in prompt
    assert "<think>" not in prompt


# ---------------------------------------------------------------------------
# 3. Dynamic Token Discovery & Logprob Extraction Across All Profiles
# ---------------------------------------------------------------------------


def test_clean_token_str():
    assert clean_token_str("yes") == "yes"
    assert clean_token_str(" Yes ") == "yes"
    assert clean_token_str("Ġyes") == "yes"
    assert clean_token_str(" yes") == "yes"
    assert clean_token_str("NO\n") == "no"
    assert clean_token_str("ĠNo") == "no"
    assert clean_token_str(" No") == "no"
    assert clean_token_str(None) == ""


def test_normalize_top_logprobs_multi_format():
    # 1. llama.cpp completion format
    llama_resp = {
        "completion_probabilities": [
            {
                "top_logprobs": [
                    {"id": 101, "token": "yes", "logprob": -0.2},
                    {"id": 102, "token": "no", "logprob": -1.8},
                ]
            }
        ]
    }
    assert len(normalize_top_logprobs(llama_resp)) == 2

    # 2. OpenAI /v1/chat/completions format
    chat_resp = {
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {
                            "top_logprobs": [
                                {"token": "yes", "logprob": -0.2},
                                {"token": "no", "logprob": -1.8},
                            ]
                        }
                    ]
                }
            }
        ]
    }
    assert len(normalize_top_logprobs(chat_resp)) == 2

    # 3. OpenAI /v1/completions dict format
    comp_resp = {
        "choices": [
            {
                "logprobs": {
                    "top_logprobs": [
                        {"yes": -0.2, "no": -1.8}
                    ]
                }
            }
        ]
    }
    norm = normalize_top_logprobs(comp_resp)
    assert len(norm) == 2
    assert any(x["token"] == "yes" and x["logprob"] == -0.2 for x in norm)

    # 4. Raw list
    raw_list = [{"token": "yes", "logprob": -0.5}]
    assert normalize_top_logprobs(raw_list) == raw_list


def test_logprob_extraction_across_all_profiles():
    profiles = [LFMProfile(), QwenProfile(), GemmaProfile(), RWKVProfile(), GenericOpenAIProfile()]

    payload = [
        {"token": "yes", "logprob": -0.338},
        {"token": "no", "logprob": -1.495},
    ]

    for prof in profiles:
        lp_yes, lp_no, is_ambiguous = prof.extract_yes_no_logprobs(payload)
        assert is_ambiguous is False
        assert abs(lp_yes - (-0.338)) < 1e-4
        assert abs(lp_no - (-1.495)) < 1e-4

        score, s_yes, s_no, amb = prof.extract_calibrated_logprobs(payload, chunk_token_est=120)
        assert amb is False
        assert 0.70 <= score <= 0.80


def test_hazard_protection_across_profiles():
    # Top-1 token is punctuation/preamble without significant yes/no probability
    payload = [
        {"token": "The", "logprob": -0.05},
        {"token": "\n", "logprob": -2.0},
        {"token": "yes", "logprob": -7.0},
        {"token": "no", "logprob": -8.0},
    ]

    for prof in [LFMProfile(), QwenProfile(), GemmaProfile(), RWKVProfile(), GenericOpenAIProfile()]:
        lp_yes, lp_no, is_ambiguous = prof.extract_yes_no_logprobs(payload)
        assert is_ambiguous is True

        score, _, _, amb = prof.extract_calibrated_logprobs(payload, chunk_token_est=100)
        assert amb is True
        assert score == 0.05


def test_dynamic_tokenizer_probe_integration():
    prof = QwenProfile()
    assert len(prof.discovered_yes_ids) == 0

    # Simulate discovering Qwen token IDs from /tokenize
    prof.discovered_yes_ids.add(9999)
    prof.discovered_no_ids.add(8888)

    payload = [
        {"id": 9999, "token": "something_else", "logprob": -0.25},
        {"id": 8888, "token": "other_token", "logprob": -2.10},
    ]

    lp_yes, lp_no, is_ambiguous = prof.extract_yes_no_logprobs(payload)
    assert is_ambiguous is False
    assert abs(lp_yes - (-0.25)) < 1e-4
    assert abs(lp_no - (-2.10)) < 1e-4


def test_length_normalization_exponents():
    lfm = LFMProfile()       # 0.15
    qwen = QwenProfile()     # 0.12
    rwkv = RWKVProfile()     # 0.10

    lp_yes = -0.3
    lp_no = -1.5

    # With a large chunk (500 tokens), higher exponent dampens logit more
    s_lfm = lfm.calibrate_score(lp_yes, lp_no, chunk_token_est=500)
    s_qwen = qwen.calibrate_score(lp_yes, lp_no, chunk_token_est=500)
    s_rwkv = rwkv.calibrate_score(lp_yes, lp_no, chunk_token_est=500)

    # RWKV (0.10) dampens least, so highest score; LFM (0.15) dampens most, so lowest score
    assert s_rwkv > s_qwen > s_lfm


# ---------------------------------------------------------------------------
# 4. Model Isolation in Cache Key
# ---------------------------------------------------------------------------


def test_cache_model_isolation():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "isolated_cache.db"
        cache_lfm = RerankCache(db_path=db_path, model_id="lfm")
        cache_qwen = RerankCache(db_path=db_path, model_id="qwen")

        query = "find memory leak in worker pool"
        chunk = "1: void worker_leak() { malloc(1024); }"

        # Verify computed cache keys are different
        key_lfm, _, _ = RerankCache.compute_cache_key(query, chunk, model_id="lfm")
        key_qwen, _, _ = RerankCache.compute_cache_key(query, chunk, model_id="qwen")
        assert key_lfm != key_qwen

        # Put score under LFM
        cache_lfm.put_batch(query, [(chunk, 0.92, -0.1, -2.5)])

        # Query via LFM cache: Hit!
        hit_lfm = cache_lfm.get(query, chunk)
        assert hit_lfm is not None
        assert abs(hit_lfm["score"] - 0.92) < 1e-4

        # Query via Qwen cache: Miss! No cross-model contamination
        hit_qwen = cache_qwen.get(query, chunk)
        assert hit_qwen is None

        # Put different score under Qwen
        cache_qwen.put_batch(query, [(chunk, 0.81, -0.3, -1.9)])

        # Verify each model retains its own score
        assert abs(cache_lfm.get(query, chunk)["score"] - 0.92) < 1e-4
        assert abs(cache_qwen.get(query, chunk)["score"] - 0.81) < 1e-4

        # Clear only Qwen cache
        cache_qwen.clear(model_id="qwen")
        assert cache_qwen.get(query, chunk) is None
        assert cache_lfm.get(query, chunk) is not None  # LFM remains intact


# ---------------------------------------------------------------------------
# 5. Auto-Detection Logic & Port Fallbacks
# ---------------------------------------------------------------------------


def test_detect_profile_from_model_names():
    assert isinstance(detect_profile_from_model_names(["Qwen2.5-Coder-7B-Instruct"]), QwenProfile)
    assert isinstance(detect_profile_from_model_names(["qwen-3-turbo"]), QwenProfile)
    assert detect_profile_from_model_names(["QwQ-32B"]).is_qwq is True
    assert isinstance(detect_profile_from_model_names(["LFM2.5-8B-A1B-Q8_0.gguf"]), LFMProfile)
    assert isinstance(detect_profile_from_model_names(["liquid-foundation-model-v2"]), LFMProfile)
    assert isinstance(detect_profile_from_model_names(["gemma-2-9b-it"]), GemmaProfile)
    assert isinstance(detect_profile_from_model_names(["gemma-3-1b"]), GemmaProfile)
    assert isinstance(detect_profile_from_model_names(["RWKV-v6-Finch-14B-Q8"]), RWKVProfile)
    assert isinstance(detect_profile_from_model_names(["custom-finetuned-llama"]), GenericOpenAIProfile)
    assert detect_profile_from_model_names([]) is None


def test_fallback_profile_from_url():
    assert isinstance(fallback_profile_from_url("http://localhost:8034/v1"), LFMProfile)
    assert isinstance(fallback_profile_from_url("http://localhost:8033/v1"), QwenProfile)
    assert isinstance(fallback_profile_from_url("http://localhost:8000/v1"), RWKVProfile)
    assert isinstance(fallback_profile_from_url("http://localhost:11434/v1"), GenericOpenAIProfile)
    assert isinstance(fallback_profile_from_url("http://192.168.1.50:9999"), LFMProfile)


def test_probe_and_detect_profile_mocked():
    with patch("httpx.Client.get") as mock_get:
        # Mock Qwen response on /v1/models
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "object": "list",
            "data": [{"id": "Qwen/Qwen2.5-7B-Instruct", "object": "model"}]
        }
        mock_get.return_value = mock_resp

        profile = probe_and_detect_profile_sync("http://remote-server:8000/v1")
        assert isinstance(profile, QwenProfile)


def test_probe_tokenizer_mocked():
    with patch("httpx.Client.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"tokens": [4242]}
        mock_post.return_value = mock_resp

        yes_ids, no_ids = probe_tokenizer_tokens_sync("http://localhost:8034/v1")
        assert 4242 in yes_ids
        assert 4242 in no_ids


# ---------------------------------------------------------------------------
# 6. Configuration file (~/.config/reranker/config.yaml)
# ---------------------------------------------------------------------------


def test_config_file_resolution():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_file = Path(tmpdir) / "config.yaml"
        save_config({
            "model": "qwen",
            "base_url": "http://localhost:8033/v1",
            "endpoints": {
                "lfm": "http://localhost:8034/v1",
                "qwen": "http://localhost:8033/v1",
                "gemma": "http://localhost:11434/v1",
            }
        }, config_path=cfg_file)

        cfg = load_config(cfg_file)
        assert cfg["model"] == "qwen"
        assert cfg["base_url"] == "http://localhost:8033/v1"

        # Resolution precedence: CLI override
        m, u = resolve_endpoint_and_model(cli_model="gemma", config=cfg)
        assert m == "gemma"
        assert u == "http://localhost:11434/v1"  # From endpoints mapping!

        # Resolution precedence: Config default
        m2, u2 = resolve_endpoint_and_model(config=cfg)
        assert m2 == "qwen"
        assert u2 == "http://localhost:8033/v1"


# ---------------------------------------------------------------------------
# 7. CLI Argument Parsing & Selection
# ---------------------------------------------------------------------------


def test_cli_flags_and_backward_compatibility():
    # New flags
    args = parse_args(["--query", "auth", "--model", "qwen", "--base-url", "http://localhost:8033/v1"])
    assert args.query == "auth"
    assert args.model == "qwen"
    assert args.base_url == "http://localhost:8033/v1"
    assert args.endpoint == "http://localhost:8033/v1"  # Alias preserved!

    # Short flags
    args_short = parse_args(["-q", "auth", "-m", "rwkv", "-e", "http://localhost:8000/v1"])
    assert args_short.model == "rwkv"
    assert args_short.base_url == "http://localhost:8000/v1"

    # Optional config flag
    args_cfg = parse_args(["-q", "test", "--config", "/tmp/custom.yaml"])
    assert args_cfg.config == "/tmp/custom.yaml"


# ---------------------------------------------------------------------------
# 8. Live Server Connectivity (Ports 8034 / 8033)
# ---------------------------------------------------------------------------


def test_live_server_8034_if_available():
    """Live integration test against port 8034 (LFM)."""
    import httpx
    try:
        r = httpx.get("http://localhost:8034/v1/models", timeout=1.0)
        if r.status_code != 200:
            pytest.skip("Port 8034 not responding with 200")
    except Exception:
        pytest.skip("Port 8034 offline")

    reranker = LFMReranker(endpoint="http://localhost:8034/v1")
    assert reranker.profile.name == "lfm"
    assert len(reranker.profile.discovered_yes_ids) > 0

    resp = reranker.rerank_sync(
        query="calculate brier score",
        candidates=["slm_rerank/eval.py"],
        no_cache=True,
    )
    assert len(resp.results) > 0
    assert resp.telemetry.model_id == "lfm"
    top = resp.results[0]
    assert "brier" in (top.symbol or "").lower() or top.score > 0.5


def test_live_server_8033_if_available():
    """Live integration test against port 8033 (Qwen)."""
    import httpx
    try:
        r = httpx.get("http://localhost:8033/v1/models", timeout=1.0)
        if r.status_code != 200:
            pytest.skip("Port 8033 not responding with 200")
    except Exception:
        pytest.skip("Port 8033 offline")

    reranker = LFMReranker(endpoint="http://localhost:8033/v1")
    assert reranker.profile.name == "qwen"


@pytest.mark.parametrize(
    "profile_cls",
    [ModelProfile, LFMProfile, QwenProfile, GemmaProfile, RWKVProfile, GenericOpenAIProfile],
)
def test_no_newline_bearing_stop_token(profile_cls):
    """Any stop token containing a newline ("\\n", "\\n\\n", ...) makes llama.cpp treat
    tokens like "{\\n" as a stop hit, which drops completion_probabilities and
    collapses every score to 0.0. Scoring requests use max_tokens=1, so stop tokens
    buy nothing and only risk suppressing the logprobs we need."""
    offenders = [tok for tok in profile_cls.stop_tokens if "\n" in tok]
    assert not offenders, f"{profile_cls.__name__} has newline-bearing stop tokens: {offenders!r}"


@pytest.mark.parametrize(
    "profile_cls",
    [LFMProfile, QwenProfile, GemmaProfile, RWKVProfile, GenericOpenAIProfile],
)
def test_prompt_asks_for_bare_yes_or_no(profile_cls):
    """"Answer (yes/no):" invites the model to open with punctuation, so the first
    generated token is "(" or a newline rather than yes/no and the yes/no logprobs
    fall outside the top-10. Every profile must use the directive phrasing."""
    prompt = profile_cls().format_prompt(
        query="charge handler",
        chunk_content="function charge() {}",
        file_path="src/pay.ts",
        symbol="charge",
        start_line=1,
        end_line=4,
    )

    assert "Respond only with yes or no." in prompt
    assert "Answer (yes/no):" not in prompt


def test_lfm_prompt_asks_for_bare_yes_or_no():
    prompt = LFMProfile().format_prompt(
        query="charge handler",
        chunk_content="function charge() {}",
        file_path="src/pay.ts",
        symbol="charge",
        start_line=1,
        end_line=4,
    )

    assert "Respond only with yes or no." in prompt
    assert "Answer (yes/no):" not in prompt
    assert prompt.endswith("<|im_start|>assistant\n<think>\n</think>\n")
