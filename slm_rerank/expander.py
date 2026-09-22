"""Optional SLM query expansion: turn a natural-language query into extra search terms.

Deterministic recall (``slm_rerank.discovery``) stems words -- it can reach "migrate"
from "migration", but never "bridge" from "harness". Only a model that knows the two
are the same idea can do that, which is the whole point of this module.

It is strictly additive and strictly optional. Every failure path returns an empty
list, so a missing, slow or confused model degrades recall back to the deterministic
terms rather than breaking it.

The on-disk cache format and key are shared with the Node implementation in
src/expander.mjs, so both sides reuse each other's expansions.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from .discovery import QUERY_STOP_WORDS, extract_query_terms

EXPANSION_CACHE_VERSION = 1
DEFAULT_TIMEOUT = 2.0
DEFAULT_MAX_TERMS = 12
CACHE_TTL_SECONDS = 30 * 24 * 60 * 60
MIN_TERM_LENGTH = 3

# Generic words a model reaches for that match half a repository.
USELESS_TERMS = {
    "code", "file", "files", "function", "functions", "method", "methods", "class", "classes",
    "module", "modules", "source", "implementation", "logic", "handler", "handlers", "util",
    "utils", "helper", "helpers", "value", "values", "data", "object", "objects", "type", "types",
    "string", "number", "boolean", "return", "const", "let", "var", "import", "export", "src",
    "api", "sdk", "app", "application", "framework", "library", "tool", "tools", "system",
    "service", "services", "server", "client", "integration", "documentation", "docs", "testing",
    "deployment", "configuration", "config", "environment", "development", "production",
    "feature", "features", "component", "components", "interface", "pattern", "patterns",
    "structure", "process", "processing", "operation", "operations", "management", "support",
}


def _cache_file() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "slm-rerank" / "expansions.json"


def _cache_key(query: str, model: str) -> str:
    raw = f"{EXPANSION_CACHE_VERSION}|{model}|{query}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _read_cache(key: str, now: float) -> Optional[List[str]]:
    try:
        store = json.loads(_cache_file().read_text(encoding="utf-8"))
        entry = store.get(key)
        if not entry or not isinstance(entry.get("terms"), list):
            return None
        # Node writes milliseconds; normalise so the two share one cache file.
        ts = float(entry.get("ts", 0)) / 1000.0
        if now - ts > CACHE_TTL_SECONDS:
            return None
        return [str(t) for t in entry["terms"]]
    except (OSError, ValueError, TypeError):
        return None  # no cache, unreadable cache, corrupt JSON -- all mean "ask the model"


def _write_cache(key: str, terms: List[str], now: float) -> None:
    try:
        path = _cache_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            store = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(store, dict):
                store = {}
        except (OSError, ValueError):
            store = {}
        store[key] = {"terms": terms, "ts": int(now * 1000)}
        path.write_text(json.dumps(store), encoding="utf-8")
    except (OSError, ValueError, TypeError):
        pass  # a cache that cannot be written must not fail the query


def format_expansion_prompt(query: str) -> str:
    """ChatML with the reasoning bypass, matching the scorer's prompt shape."""
    return (
        "<|startoftext|><|im_start|>system\n"
        "You expand code search queries for a codebase search tool.\n"
        "Given a query, list the identifiers, domain nouns and synonyms that would plausibly "
        "appear in the source file that answers it, especially words the query does not "
        "already contain.\n"
        "Reply with one line of lowercase comma-separated terms. No prose, no explanation.\n"
        "<|im_end|>\n"
        f"<|im_start|>user\nQuery: {query}\n<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n</think>\n"
    )


def parse_expansion_text(text: str, query: str, max_terms: int = DEFAULT_MAX_TERMS) -> List[str]:
    """Pull terms out of whatever shape the endpoint returned."""
    if not text or not isinstance(text, str) or not text.strip():
        return []

    already = set(extract_query_terms(query))
    terms: List[str] = []
    for piece in "".join(c if c.isalnum() or c == "_" else " " for c in text.lower()).split():
        term = piece.strip()
        if len(term) < MIN_TERM_LENGTH:
            continue
        if term in USELESS_TERMS or term in QUERY_STOP_WORDS:
            continue
        if term in already or term in terms:
            continue
        terms.append(term)
        if len(terms) >= max_terms:
            break
    return terms


def _extract_text(data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    if isinstance(data.get("content"), str):
        return data["content"]  # llama.cpp /completion
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice.get("text"), str):
            return choice["text"]
        message = choice.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]
    return ""


def _completion_url(base_url: str) -> str:
    """llama.cpp serves native completion at the server root, not under /v1."""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        return f"{root[:-3]}/completion"
    if root.endswith("/completion"):
        return root
    return f"{root}/completion"


async def expand_query(
    query: str,
    base_url: str,
    model: str = "lfm",
    timeout: float = DEFAULT_TIMEOUT,
    max_terms: int = DEFAULT_MAX_TERMS,
    use_cache: bool = True,
    client: Optional[httpx.AsyncClient] = None,
) -> List[str]:
    """Ask the local model for extra search terms. Returns [] on any failure."""
    if not query or not isinstance(query, str) or not base_url:
        return []

    now = time.time()
    key = _cache_key(query, model)
    if use_cache:
        cached = _read_cache(key, now)
        if cached:
            return cached[:max_terms]

    payload: Dict[str, Any] = {
        "prompt": format_expansion_prompt(query),
        "n_predict": 64,
        "temperature": 0,  # greedy: same query -> same terms, so the cache is meaningful
        "top_p": 0.9,
        # Unlike scoring, this call reads generated text rather than logprobs, so
        # newline stop tokens are safe here -- they are not in adapters.py.
        "stop": ["<|im_end|>", "\n"],
        "cache_prompt": True,
    }

    try:
        if client is not None:
            resp = await client.post(_completion_url(base_url), json=payload, timeout=timeout)
        else:
            async with httpx.AsyncClient(timeout=timeout) as owned:
                resp = await owned.post(_completion_url(base_url), json=payload)
        if resp.status_code != 200:
            return []
        terms = parse_expansion_text(_extract_text(resp.json()), query, max_terms)
    except (httpx.HTTPError, ValueError, TypeError):
        return []  # unreachable, timed out, malformed -- recall carries on without us

    if terms and use_cache:
        _write_cache(key, terms, now)
    return terms
