"""Model profiles and provider adapters for model-agnostic logprob reranking."""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Type, Union

import httpx

logger = logging.getLogger("slm_rerank")

# Default reference token IDs for LFM (Liquid Foundation Model) BPE vocabulary
LFM_YES_TOKEN_IDS = {11683, 12447, 18171, 17550}  # 'yes', 'Yes', ' yes', ' Yes'
LFM_NO_TOKEN_IDS = {2243, 4547, 794, 2752}        # 'no', 'No', ' no', ' No'

YES_VARIANTS: Set[str] = {"yes", "true", "y", "1", "+1", "relevant", "match", "positive"}
NO_VARIANTS: Set[str] = {"no", "false", "n", "0", "-1", "irrelevant", "negative"}


def clean_token_str(token: str) -> str:
    """Normalize token string by stripping whitespace, BPE space prefixes, punctuation, and lowercasing."""
    if not token:
        return ""
    # Strip common BPE/SentencePiece space prefixes: 'Ġ', ' ', '_'
    s = str(token).replace("Ġ", "").replace(" ", "")
    # Strip whitespace and common punctuation wrappers around completion tokens: .,:;!?'"()[]{}*`
    s = s.strip(" \t\n\r\"'.,;:!?*`~()[]{}")
    return s.lower()


def logsumexp(lps: Sequence[float]) -> Optional[float]:
    """Compute log-sum-exp in a numerically stable way."""
    if not lps:
        return None
    m = max(lps)
    return m + math.log(sum(math.exp(x - m) for x in lps))


def normalize_top_logprobs(response_payload: Any) -> List[Dict[str, Any]]:
    """Extract and normalize top_logprobs into a list of {'token': str, 'logprob': float, 'id': Optional[int]}.

    Supports:
      1. llama.cpp completion endpoint: data['completion_probabilities'][0]['top_logprobs']
      2. OpenAI/vLLM chat completions: data['choices'][0]['logprobs']['content'][0]['top_logprobs']
      3. OpenAI completions: data['choices'][0]['logprobs']['top_logprobs'][0]
      4. Direct list of logprob dicts
      5. Top-level 'top_logprobs' key
    """
    if isinstance(response_payload, list):
        return response_payload

    if not isinstance(response_payload, dict):
        return []

    # 1. llama.cpp completion format
    if "completion_probabilities" in response_payload:
        comp_probs = response_payload.get("completion_probabilities")
        if isinstance(comp_probs, list) and comp_probs:
            first_step = comp_probs[0]
            if isinstance(first_step, dict):
                top_lps = first_step.get("top_logprobs", [])
                if isinstance(top_lps, list):
                    return top_lps

    # 2. OpenAI-compatible /v1/chat/completions or /v1/completions format
    choices = response_payload.get("choices")
    if isinstance(choices, list) and choices:
        c0 = choices[0]
        if isinstance(c0, dict):
            logprobs = c0.get("logprobs")
            if isinstance(logprobs, dict):
                # /v1/chat/completions with logprobs=True
                content = logprobs.get("content")
                if isinstance(content, list) and content:
                    top_lps = content[0].get("top_logprobs")
                    if isinstance(top_lps, list):
                        return top_lps
                # /v1/completions with logprobs=10
                top_lps_list = logprobs.get("top_logprobs")
                if isinstance(top_lps_list, list) and top_lps_list:
                    entry = top_lps_list[0]
                    if isinstance(entry, dict):
                        # May be a mapping of {"token_str": logprob, ...}
                        if "token" in entry or "logprob" in entry:
                            return top_lps_list
                        return [{"token": k, "logprob": float(v)} for k, v in entry.items()]

    # 3. Direct top_logprobs key
    if "top_logprobs" in response_payload:
        top_lps = response_payload.get("top_logprobs")
        if isinstance(top_lps, list):
            return top_lps

    return []


class ModelProfile:
    """Base interface for model-specific prompt formatting, scoring, and logprob extraction."""

    name: str = "generic"
    prompt_version: str = "v1"
    default_base_url: str = "http://localhost:8034/v1"
    length_normalization_exponent: float = 0.12
    stop_tokens: List[str] = ["\n"]

    def __init__(
        self,
        discovered_yes_ids: Optional[Set[int]] = None,
        discovered_no_ids: Optional[Set[int]] = None,
    ):
        self.discovered_yes_ids: Set[int] = set(discovered_yes_ids) if discovered_yes_ids else set()
        self.discovered_no_ids: Set[int] = set(discovered_no_ids) if discovered_no_ids else set()
        self.last_ambiguity_event: Optional[Dict[str, Any]] = None

    def format_prompt(
        self,
        query: str,
        chunk_content: str,
        file_path: Optional[str] = None,
        symbol: Optional[str] = None,
        intent: Optional[Any] = None,
        is_test: bool = False,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Format the binary retrieval prompt for this model."""
        raise NotImplementedError

    def extract_yes_no_logprobs(
        self,
        response_payload: Any,
        prompt_state: Optional[str] = None,
    ) -> Tuple[Optional[float], Optional[float], bool]:
        """Extract (logprob_yes, logprob_no, is_ambiguous) from response payload.

        Uses dynamic string-based token matching with optional discovered token IDs,
        whitespace recovery, and detailed ambiguity telemetry logging.
        """
        self.last_ambiguity_event = None
        top_logprobs = normalize_top_logprobs(response_payload)
        if not top_logprobs:
            self._record_ambiguity(
                token_text="",
                token_id=None,
                logprob=None,
                top_candidates=[],
                prompt_state=prompt_state,
                reason="Empty top_logprobs payload",
            )
            return None, None, True

        top_1 = top_logprobs[0]
        top_1_id = top_1.get("id")
        top_1_raw = str(top_1.get("token", ""))
        top_1_str = clean_token_str(top_1_raw)

        yes_ids = set(self.discovered_yes_ids)
        no_ids = set(self.discovered_no_ids)

        # Retain LFM reference IDs specifically when running under the LFM profile
        if self.name == "lfm":
            yes_ids.update(LFM_YES_TOKEN_IDS)
            no_ids.update(LFM_NO_TOKEN_IDS)

        top_1_is_yes = (top_1_id in yes_ids) if (top_1_id is not None and yes_ids) else False
        top_1_is_yes = top_1_is_yes or (top_1_str in YES_VARIANTS)

        top_1_is_no = (top_1_id in no_ids) if (top_1_id is not None and no_ids) else False
        top_1_is_no = top_1_is_no or (top_1_str in NO_VARIANTS)

        # Leading whitespace/colon/punctuation recovery:
        # If top_1 was pure whitespace or formatting artifact, inspect first non-formatting candidate
        if not top_1_is_yes and not top_1_is_no and (not top_1_str or top_1_raw in {" ", "\n", "\t", "\r", ":"}):
            for candidate in top_logprobs[1:4]:
                c_id = candidate.get("id")
                c_str = clean_token_str(candidate.get("token", ""))
                c_is_yes = (c_id in yes_ids) if (c_id is not None and yes_ids) else (c_str in YES_VARIANTS)
                c_is_no = (c_id in no_ids) if (c_id is not None and no_ids) else (c_str in NO_VARIANTS)
                if c_is_yes:
                    top_1_is_yes = True
                    break
                elif c_is_no:
                    top_1_is_no = True
                    break

        yes_lps: List[float] = []
        no_lps: List[float] = []

        for item in top_logprobs:
            token_id = item.get("id")
            token_raw = str(item.get("token", ""))
            token_clean = clean_token_str(token_raw)
            lp = item.get("logprob")
            if lp is None:
                continue

            item_is_yes = (token_id in yes_ids) if (token_id is not None and yes_ids) else False
            item_is_yes = item_is_yes or (token_clean in YES_VARIANTS)

            item_is_no = (token_id in no_ids) if (token_id is not None and no_ids) else False
            item_is_no = item_is_no or (token_clean in NO_VARIANTS)

            if item_is_yes:
                yes_lps.append(float(lp))
            elif item_is_no:
                no_lps.append(float(lp))

        lp_yes = logsumexp(yes_lps)
        lp_no = logsumexp(no_lps)

        # Hazard Protection: Top-1 token is neither yes nor no
        if not top_1_is_yes and not top_1_is_no:
            has_significant_binary = (
                (lp_yes is not None and math.exp(lp_yes) >= 0.20)
                or (lp_no is not None and math.exp(lp_no) >= 0.20)
            )
            if not has_significant_binary:
                self._record_ambiguity(
                    token_text=top_1_raw,
                    token_id=top_1_id,
                    logprob=top_1.get("logprob"),
                    top_candidates=top_logprobs[:5],
                    prompt_state=prompt_state,
                    reason="Top-1 token is neither yes nor no and binary probability mass < 0.20",
                )
                return lp_yes, lp_no, True

        if lp_yes is None and lp_no is None:
            self._record_ambiguity(
                token_text=top_1_raw,
                token_id=top_1_id,
                logprob=top_1.get("logprob"),
                top_candidates=top_logprobs[:5],
                prompt_state=prompt_state,
                reason="Both lp_yes and lp_no are None",
            )
            return None, None, True

        return lp_yes, lp_no, False

    def _record_ambiguity(
        self,
        token_text: str,
        token_id: Optional[int],
        logprob: Optional[float],
        top_candidates: List[Dict[str, Any]],
        prompt_state: Optional[str] = None,
        reason: str = "",
    ) -> None:
        """Instrument and log detailed ambiguity telemetry whenever non-binary completion occurs."""
        p_suffix = prompt_state[-150:] if prompt_state else ""
        event = {
            "token_text": token_text,
            "token_id": token_id,
            "logprob": round(float(logprob), 4) if logprob is not None else None,
            "prompt_suffix": p_suffix,
            "prompt_state": prompt_state or "",
            "reason": reason,
            "top_candidates": [
                {
                    "token": clean_token_str(t.get("token", "")),
                    "raw": t.get("token", ""),
                    "id": t.get("id"),
                    "logprob": round(float(t.get("logprob", 0.0)), 4) if t.get("logprob") is not None else None,
                }
                for t in top_candidates
            ],
        }
        self.last_ambiguity_event = event
        logger.warning(
            f"[Ambiguity Telemetry] Non-binary token emitted: text={token_text!r}, id={token_id}, "
            f"lp={logprob}, reason='{reason}', candidates={[c.get('token') for c in event['top_candidates']]}"
        )

    def calibrate_score(
        self,
        lp_yes: Optional[float],
        lp_no: Optional[float],
        is_ambiguous: bool = False,
        chunk_token_est: int = 100,
    ) -> float:
        """Apply length-normalized sigmoid calibration to logit difference."""
        if is_ambiguous:
            return 0.05

        if lp_yes is not None and lp_no is None:
            raw_logit = 3.0
        elif lp_no is not None and lp_yes is None:
            raw_logit = -3.0
        elif lp_yes is not None and lp_no is not None:
            raw_logit = lp_yes - lp_no
        else:
            return 0.05

        ref_length = 120.0
        eff_length = max(40.0, float(chunk_token_est))
        length_scale = (ref_length / eff_length) ** self.length_normalization_exponent
        calibrated_logit = raw_logit * length_scale

        try:
            calibrated_prob = 1.0 / (1.0 + math.exp(-calibrated_logit))
        except OverflowError:
            calibrated_prob = 1.0 if calibrated_logit > 0 else 0.0

        return round(calibrated_prob, 4)

    def extract_calibrated_logprobs(
        self,
        response_payload: Any,
        chunk_token_est: int = 100,
        prompt_state: Optional[str] = None,
    ) -> Tuple[float, Optional[float], Optional[float], bool]:
        """Convenience method returning (calibrated_score, lp_yes, lp_no, is_ambiguous)."""
        lp_yes, lp_no, is_ambiguous = self.extract_yes_no_logprobs(response_payload, prompt_state=prompt_state)
        score = self.calibrate_score(
            lp_yes=lp_yes,
            lp_no=lp_no,
            is_ambiguous=is_ambiguous,
            chunk_token_est=chunk_token_est,
        )
        return score, lp_yes, lp_no, is_ambiguous


# ProviderAdapter alias for ModelProfile
ProviderAdapter = ModelProfile


class LFMProfile(ModelProfile):
    """Profile for LFM 2.5 (Liquid Foundation Model) ChatML with think-bypass."""

    name: str = "lfm"
    default_base_url: str = "http://localhost:8034/v1"
    length_normalization_exponent: float = 0.15
    stop_tokens: List[str] = ["<|im_end|>", "\n", "<think>"]

    def format_prompt(
        self,
        query: str,
        chunk_content: str,
        file_path: Optional[str] = None,
        symbol: Optional[str] = None,
        intent: Optional[Any] = None,
        is_test: bool = False,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        file_info = f"File: {file_path}\n" if file_path else ""
        if file_path and start_line is not None and end_line is not None:
            line_info = f"Lines: {start_line}-{end_line}\n"
        elif file_path:
            line_info = "Lines: 1-1\n"
        else:
            line_info = ""

        sym_info = f"Symbol: {symbol}\n" if symbol else ""
        test_guidance = (
            "[Context: This snippet is from an authoritative test/specification suite. "
            "If the query asks for behaviors, error cases, expectations, contracts, or functionality verified here, "
            "treat this test definition as directly relevant.]\n\n"
            if is_test
            else ""
        )

        return (
            "<|startoftext|><|im_start|>system\n"
            "You are a binary code retrieval evaluator. For the given search query and code snippet, "
            "evaluate if the snippet contains the relevant implementation, specification, definition, or answer requested.\n"
            "Respond with exactly \"yes\" if relevant, or \"no\" if not relevant.\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"Query: {query}\n\n"
            f"{file_info}"
            f"{line_info}"
            f"{sym_info}"
            f"{test_guidance}"
            "Code:\n"
            f"{chunk_content}\n\n"
            "Does this snippet contain the relevant code for the query? Answer (yes/no):"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n"
            "</think>\n"
        )


class QwenProfile(ModelProfile):
    """Profile for Qwen (2.5 / 3 / 3.8 / QwQ) ChatML format."""

    name: str = "qwen"
    default_base_url: str = "http://localhost:8033/v1"
    length_normalization_exponent: float = 0.12
    stop_tokens: List[str] = ["<|im_end|>", "\n"]

    def __init__(
        self,
        is_qwq: bool = False,
        discovered_yes_ids: Optional[Set[int]] = None,
        discovered_no_ids: Optional[Set[int]] = None,
    ):
        super().__init__(discovered_yes_ids=discovered_yes_ids, discovered_no_ids=discovered_no_ids)
        self.is_qwq = is_qwq

    def format_prompt(
        self,
        query: str,
        chunk_content: str,
        file_path: Optional[str] = None,
        symbol: Optional[str] = None,
        intent: Optional[Any] = None,
        is_test: bool = False,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        file_info = f"File: {file_path}\n" if file_path else ""
        if file_path and start_line is not None and end_line is not None:
            line_info = f"Lines: {start_line}-{end_line}\n"
        elif file_path:
            line_info = "Lines: 1-1\n"
        else:
            line_info = ""

        sym_info = f"Symbol: {symbol}\n" if symbol else ""
        test_guidance = (
            "[Context: This snippet is from an authoritative test/specification suite. "
            "If the query asks for behaviors, error cases, expectations, contracts, or functionality verified here, "
            "treat this test definition as directly relevant.]\n\n"
            if is_test
            else ""
        )

        think_suffix = "<think>\n</think>\n" if self.is_qwq else ""

        return (
            "<|im_start|>system\n"
            "You are a binary code retrieval evaluator. For the given search query and code snippet, "
            "evaluate if the snippet contains the relevant implementation, specification, definition, or answer requested.\n"
            "Respond with exactly \"yes\" if relevant, or \"no\" if not relevant.<|im_end|>\n"
            "<|im_start|>user\n"
            f"Query: {query}\n\n"
            f"{file_info}"
            f"{line_info}"
            f"{sym_info}"
            f"{test_guidance}"
            "Code:\n"
            f"{chunk_content}\n\n"
            "Does this snippet contain the relevant code for the query? Answer (yes/no):<|im_end|>\n"
            f"<|im_start|>assistant\n{think_suffix}"
        )


class GemmaProfile(ModelProfile):
    """Profile for Gemma (2 / 3) turn-based prompt format."""

    name: str = "gemma"
    default_base_url: str = "http://localhost:11434/v1"
    length_normalization_exponent: float = 0.12
    stop_tokens: List[str] = ["<end_of_turn>", "\n"]

    def format_prompt(
        self,
        query: str,
        chunk_content: str,
        file_path: Optional[str] = None,
        symbol: Optional[str] = None,
        intent: Optional[Any] = None,
        is_test: bool = False,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        file_info = f"File: {file_path}\n" if file_path else ""
        if file_path and start_line is not None and end_line is not None:
            line_info = f"Lines: {start_line}-{end_line}\n"
        elif file_path:
            line_info = "Lines: 1-1\n"
        else:
            line_info = ""

        sym_info = f"Symbol: {symbol}\n" if symbol else ""
        test_guidance = (
            "[Context: This snippet is from an authoritative test/specification suite. "
            "If the query asks for behaviors, error cases, expectations, contracts, or functionality verified here, "
            "treat this test definition as directly relevant.]\n\n"
            if is_test
            else ""
        )

        return (
            "<start_of_turn>user\n"
            "You are a binary code retrieval evaluator. For the given search query and code snippet, "
            "evaluate if the snippet contains the relevant implementation, specification, definition, or answer requested.\n"
            "Respond with exactly \"yes\" if relevant, or \"no\" if not relevant.\n\n"
            f"Query: {query}\n\n"
            f"{file_info}"
            f"{line_info}"
            f"{sym_info}"
            f"{test_guidance}"
            "Code:\n"
            f"{chunk_content}\n\n"
            "Does this snippet contain the relevant code for the query? Answer (yes/no):<end_of_turn>\n"
            "<start_of_turn>model\n"
        )


class RWKVProfile(ModelProfile):
    """Profile for RWKV (v5 / v6 / v7) raw completion format optimized for RNN/linear state attention."""

    name: str = "rwkv"
    default_base_url: str = "http://localhost:8000/v1"
    length_normalization_exponent: float = 0.10
    stop_tokens: List[str] = ["\n\n", "\n", "User:"]

    def format_prompt(
        self,
        query: str,
        chunk_content: str,
        file_path: Optional[str] = None,
        symbol: Optional[str] = None,
        intent: Optional[Any] = None,
        is_test: bool = False,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        file_info = f"File: {file_path}\n" if file_path else ""
        if file_path and start_line is not None and end_line is not None:
            line_info = f"Lines: {start_line}-{end_line}\n"
        elif file_path:
            line_info = "Lines: 1-1\n"
        else:
            line_info = ""

        sym_info = f"Symbol: {symbol}\n" if symbol else ""
        test_guidance = (
            "[Context: This snippet is from an authoritative test/specification suite. "
            "If the query asks for behaviors, error cases, expectations, contracts, or functionality verified here, "
            "treat this test definition as directly relevant.]\n\n"
            if is_test
            else ""
        )

        return (
            "User: You are a binary code retrieval evaluator. For the given search query and code snippet, "
            "evaluate if the snippet contains the relevant implementation, specification, definition, or answer requested.\n"
            "Respond with exactly \"yes\" if relevant, or \"no\" if not relevant.\n\n"
            f"Query: {query}\n\n"
            f"{file_info}"
            f"{line_info}"
            f"{sym_info}"
            f"{test_guidance}"
            "Code:\n"
            f"{chunk_content}\n\n"
            "Does this snippet contain the relevant code for the query? Answer (yes/no):\n\n"
            "Assistant: "
        )


class GenericOpenAIProfile(ModelProfile):
    """Generic OpenAI-compatible profile (vLLM, Ollama, llama.cpp, TabbyAPI)."""

    name: str = "openai"
    default_base_url: str = "http://localhost:11434/v1"
    length_normalization_exponent: float = 0.12
    stop_tokens: List[str] = ["<|im_end|>", "<end_of_turn>", "\n"]

    def format_prompt(
        self,
        query: str,
        chunk_content: str,
        file_path: Optional[str] = None,
        symbol: Optional[str] = None,
        intent: Optional[Any] = None,
        is_test: bool = False,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        file_info = f"File: {file_path}\n" if file_path else ""
        if file_path and start_line is not None and end_line is not None:
            line_info = f"Lines: {start_line}-{end_line}\n"
        elif file_path:
            line_info = "Lines: 1-1\n"
        else:
            line_info = ""

        sym_info = f"Symbol: {symbol}\n" if symbol else ""
        test_guidance = (
            "[Context: This snippet is from an authoritative test/specification suite. "
            "If the query asks for behaviors, error cases, expectations, contracts, or functionality verified here, "
            "treat this test definition as directly relevant.]\n\n"
            if is_test
            else ""
        )

        return (
            "<|im_start|>system\n"
            "You are a binary code retrieval evaluator. For the given search query and code snippet, "
            "evaluate if the snippet contains the relevant implementation, specification, definition, or answer requested.\n"
            "Respond with exactly \"yes\" if relevant, or \"no\" if not relevant.<|im_end|>\n"
            "<|im_start|>user\n"
            f"Query: {query}\n\n"
            f"{file_info}"
            f"{line_info}"
            f"{sym_info}"
            f"{test_guidance}"
            "Code:\n"
            f"{chunk_content}\n\n"
            "Does this snippet contain the relevant code for the query? Answer (yes/no):<|im_end|>\n"
            "<|im_start|>assistant\n"
        )


PROFILES: Dict[str, Type[ModelProfile]] = {
    "lfm": LFMProfile,
    "qwen": QwenProfile,
    "gemma": GemmaProfile,
    "rwkv": RWKVProfile,
    "openai": GenericOpenAIProfile,
}

PROFILE_ALIASES: Dict[str, str] = {
    "lfm2": "lfm",
    "lfm2.5": "lfm",
    "lfm-2.5": "lfm",
    "liquid": "lfm",
    "qwen2": "qwen",
    "qwen2.5": "qwen",
    "qwen3": "qwen",
    "qwen3.8": "qwen",
    "qwq": "qwen",
    "gemma2": "gemma",
    "gemma3": "gemma",
    "rwkv5": "rwkv",
    "rwkv6": "rwkv",
    "rwkv7": "rwkv",
    "generic": "openai",
    "ollama": "openai",
    "vllm": "openai",
    "tabbyapi": "openai",
}


def get_profile(name_or_alias: str, **kwargs: Any) -> ModelProfile:
    """Retrieve profile instance by model name or alias."""
    low = name_or_alias.strip().lower()
    resolved = PROFILE_ALIASES.get(low, low)
    cls = PROFILES.get(resolved)
    if cls is None:
        prof = GenericOpenAIProfile(**kwargs)
        prof.name = low
        return prof
    if resolved == "qwen" and ("qwq" in low):
        kwargs.setdefault("is_qwq", True)
    return cls(**kwargs)


def detect_profile_from_model_names(model_names: Sequence[str]) -> Optional[ModelProfile]:
    """Inspect model names or IDs returned by server and select matching profile."""
    for name in model_names:
        low = name.lower()
        if "qwen" in low or "qwq" in low:
            return QwenProfile(is_qwq="qwq" in low)
        if "lfm" in low or "liquid" in low:
            return LFMProfile()
        if "gemma" in low:
            return GemmaProfile()
        if "rwkv" in low:
            return RWKVProfile()
    if model_names:
        return GenericOpenAIProfile()
    return None


def fallback_profile_from_url(url: str) -> ModelProfile:
    """Heuristic fallback based on local port when server cannot be probed."""
    u = url.lower()
    if ":8034" in u:
        return LFMProfile()
    if ":8033" in u:
        return QwenProfile()
    if ":8000" in u:
        return RWKVProfile()
    if ":11434" in u:
        return GenericOpenAIProfile()
    return LFMProfile()


def probe_and_detect_profile_sync(
    base_url: str,
    timeout: float = 2.0,
) -> ModelProfile:
    """Probe GET /v1/models on base_url and auto-detect ModelProfile."""
    raw = base_url.rstrip("/")
    if raw.endswith("/models"):
        models_url = raw
    elif raw.endswith("/v1"):
        models_url = f"{raw}/models"
    else:
        models_url = f"{raw}/v1/models"

    model_names: List[str] = []
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(models_url)
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("data", []):
                    if isinstance(item, dict):
                        if "id" in item:
                            model_names.append(str(item["id"]))
                        if "name" in item:
                            model_names.append(str(item["name"]))
                for item in data.get("models", []):
                    if isinstance(item, dict):
                        if "name" in item:
                            model_names.append(str(item["name"]))
                        if "model" in item:
                            model_names.append(str(item["model"]))
    except Exception as e:
        logger.debug(f"Auto-detect probe failed on {models_url}: {e}")

    detected = detect_profile_from_model_names(model_names)
    if detected is not None:
        return detected

    return fallback_profile_from_url(base_url)


async def probe_and_detect_profile_async(
    base_url: str,
    timeout: float = 2.0,
    client: Optional[httpx.AsyncClient] = None,
) -> ModelProfile:
    """Asynchronous version of probe_and_detect_profile."""
    raw = base_url.rstrip("/")
    if raw.endswith("/models"):
        models_url = raw
    elif raw.endswith("/v1"):
        models_url = f"{raw}/models"
    else:
        models_url = f"{raw}/v1/models"

    model_names: List[str] = []
    try:
        if client is not None:
            resp = await client.get(models_url, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("data", []):
                    if isinstance(item, dict) and "id" in item:
                        model_names.append(str(item["id"]))
                for item in data.get("models", []):
                    if isinstance(item, dict) and "name" in item:
                        model_names.append(str(item["name"]))
        else:
            async with httpx.AsyncClient(timeout=timeout) as c:
                resp = await c.get(models_url)
                if resp.status_code == 200:
                    data = resp.json()
                    for item in data.get("data", []):
                        if isinstance(item, dict) and "id" in item:
                            model_names.append(str(item["id"]))
                    for item in data.get("models", []):
                        if isinstance(item, dict) and "name" in item:
                            model_names.append(str(item["name"]))
    except Exception as e:
        logger.debug(f"Async auto-detect probe failed on {models_url}: {e}")

    detected = detect_profile_from_model_names(model_names)
    if detected is not None:
        return detected

    return fallback_profile_from_url(base_url)


def probe_tokenizer_tokens_sync(
    base_url: str,
    timeout: float = 2.0,
) -> Tuple[Set[int], Set[int]]:
    """Probes /tokenize on base_url for Yes/No token IDs, with graceful fallback to empty sets."""
    raw = base_url.rstrip("/")
    if raw.endswith("/v1"):
        tokenize_url = f"{raw[:-3].rstrip('/')}/tokenize"
    elif raw.endswith("/tokenize"):
        tokenize_url = raw
    else:
        tokenize_url = f"{raw}/tokenize"

    yes_words = ["yes", "Yes", " yes", " Yes"]
    no_words = ["no", "No", " no", " No"]
    yes_ids: Set[int] = set()
    no_ids: Set[int] = set()

    try:
        with httpx.Client(timeout=timeout) as client:
            for w in yes_words:
                resp = client.post(tokenize_url, json={"content": w})
                if resp.status_code == 200:
                    data = resp.json()
                    toks = data.get("tokens", [])
                    if len(toks) == 1 and isinstance(toks[0], int):
                        yes_ids.add(toks[0])
            for w in no_words:
                resp = client.post(tokenize_url, json={"content": w})
                if resp.status_code == 200:
                    data = resp.json()
                    toks = data.get("tokens", [])
                    if len(toks) == 1 and isinstance(toks[0], int):
                        no_ids.add(toks[0])
    except Exception as e:
        logger.debug(f"Tokenizer probing failed on {tokenize_url}: {e}")

    return yes_ids, no_ids
