"""
Multi-Port Endpoint & Smart File Auto-Discovery for SLM Reranker.
Scans distinct model ports across the 8033-8040 range and discovers candidate files via ripgrep/git grep.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import urllib.request
import urllib.error

SLM_PORT_RANGE = [8033, 8034, 8035, 8036, 8037, 8038, 8039, 8040]

# Endpoint and host environment variables, in precedence order. SLM_* is the
# canonical spelling shared with the Node implementation; RERANKER_*/LFM_* are
# the historical Python names, kept so one variable configures either side.
ENDPOINT_ENV_VARS = ("SLM_ENDPOINT", "RERANKER_BASE_URL", "LFM_ENDPOINT")
HOST_ENV_VARS = ("SLM_HOST", "RERANKER_HOST")
DEFAULT_HOST = "127.0.0.1"


def _first_env_value(names, env: Optional[Dict[str, str]] = None) -> Optional[str]:
    source = os.environ if env is None else env
    for name in names:
        value = source.get(name)
        if value and value.strip():
            return value.strip()
    return None


def resolve_endpoint_env(env: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Full endpoint URL pinned via env, or None when none is set."""
    return _first_env_value(ENDPOINT_ENV_VARS, env)


def resolve_host_env(env: Optional[Dict[str, str]] = None) -> str:
    """Host to scan for model servers. Defaults to loopback."""
    return _first_env_value(HOST_ENV_VARS, env) or DEFAULT_HOST


def probe_port_sync(host: str, port: int, timeout: float = 0.25) -> Optional[Dict[str, Any]]:
    """Synchronously probe a single host:port/v1/models endpoint."""
    url = f"http://{host}:{port}/v1"
    req = urllib.request.Request(f"{url}/models", headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                body = json.loads(resp.read().decode("utf-8"))
                model_id = "unknown"
                if "data" in body and body["data"] and "id" in body["data"][0]:
                    model_id = body["data"][0]["id"]
                elif "models" in body and body["models"] and "name" in body["models"][0]:
                    model_id = body["models"][0]["name"]
                return {
                    "port": port,
                    "host": host,
                    "url": url,
                    "model_id": model_id,
                    "ok": True,
                }
    except Exception:
        pass
    return None


async def probe_port_async(host: str, port: int, timeout: float = 0.25) -> Optional[Dict[str, Any]]:
    """Asynchronously probe a single host:port/v1/models endpoint using asyncio."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, probe_port_sync, host, port, timeout)


async def auto_discover_endpoint(
    host: Optional[str] = None,
    requested_model: Optional[str] = None,
    ports: List[int] = SLM_PORT_RANGE,
    timeout: float = 0.30,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Concurrently scan ports 8033-8040 on a single host for active SLM servers.
    If requested_model is specified, matches against model ID (e.g. 'lfm', 'qwen').
    Otherwise prioritizes 8034 (LFM) and 8033 (Qwen), falling back to first responding port.

    A URL pinned via SLM_ENDPOINT/RERANKER_BASE_URL/LFM_ENDPOINT wins outright.
    When nothing answers, the result carries ok=False, url=None and a 'reason'.
    """
    host = host or resolve_host_env(env)

    # An explicitly pinned URL wins outright, including when a model was requested:
    # the caller named the server, so we do not second-guess it by scanning loopback.
    pinned = resolve_endpoint_env(env)
    if pinned:
        return {
            "port": None,
            "host": host,
            "url": pinned,
            "model_id": "pinned-via-env",
            "ok": True,
        }

    tasks = [probe_port_async(host, p, timeout) for p in ports]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    results = [r for r in raw_results if isinstance(r, dict) and r.get("ok")]

    if not results:
        # No phantom endpoint: handing back a URL nothing answered on turns
        # "no server here" into a connection error much later, which reads
        # like a transport bug rather than a configuration gap.
        port_range = f"{ports[0]}-{ports[-1]}" if ports else "none"
        return {
            "port": None,
            "host": host,
            "url": None,
            "model_id": None,
            "ok": False,
            "reason": (
                f"No SLM model server answered on {host} (ports {port_range}). "
                "Discovery only scans ports on a single host, never the network. "
                "If the model runs on another machine, set "
                "SLM_ENDPOINT=http://<host>:8034/v1 or SLM_HOST=<host>."
            ),
        }

    if requested_model:
        req_lower = requested_model.lower()
        for r in results:
            if req_lower in r["model_id"].lower():
                return r

    # Preference: 8034 (LFM) > 8033 (Qwen) > first responding
    for p in [8034, 8033]:
        for r in results:
            if r["port"] == p:
                return r

    return results[0]


CODE_EXTS = {".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".rs", ".go", ".c", ".cpp", ".h"}

QUERY_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "of", "for",
    "with", "from", "into", "by", "as", "is", "are", "was", "were", "be",
    "code", "file", "function", "method", "class", "find", "search", "how", "what", "where",
}

IGNORE_GLOBS = ["!node_modules", "!.git", "!dist", "!build", "!coverage", "!__pycache__", "!.venv"]
MAX_SEARCH_TERMS = 6    # one rg invocation each, so this is the cost knob
PATH_HIT_WEIGHT = 2     # a term in the path is a stronger signal than one body mention
TEST_FILE_FACTOR = 0.5  # tests restate domain vocabulary; rank them below implementations
MIN_STEM_LENGTH = 4
NO_MATCH_SAMPLE = 20

_WORD_RE = re.compile(r"\b[a-z0-9_]{2,}\b")
_PLURAL_ES_RE = re.compile(r"(?:s|x|z|ch|sh)es$")
_TEST_DIR_RE = re.compile(r"(^|/)(tests?|__tests__|specs?)(/|$)")
_TEST_FILE_RE = re.compile(r"\.(test|spec)\.[a-z0-9]+$")
_TEST_SUFFIX_RE = re.compile(r"_test\.[a-z0-9]+$")


def _glob_args() -> List[str]:
    args: List[str] = []
    for glob in IGNORE_GLOBS:
        args.extend(["--glob", glob])
    return args


def _stem_variants(word: str) -> List[str]:
    """Suffix stems, emitted next to their root so a term cap never severs them."""
    out: List[str] = []

    def add(stem: str) -> None:
        if len(stem) >= MIN_STEM_LENGTH and stem not in out:
            out.append(stem)

    if word.endswith("ing") and len(word) > 6:
        add(word[:-3])
    elif word.endswith("ion") and len(word) > 6:
        base = word[:-3]
        add(base)           # migration -> migrat, which substring-matches migrate too
        add(base + "e")     # migration -> migrate
    elif word.endswith("ate") and len(word) > 5:
        add(word[:-1])
    elif word.endswith("ed") and len(word) > 5:
        add(word[:-2])
    elif _PLURAL_ES_RE.search(word) and len(word) > 4:
        add(word[:-2])      # classes -> class, boxes -> box
    elif word.endswith("s") and not word.endswith("ss") and len(word) > 4:
        add(word[:-1])      # exports -> export, but harness stays harness
    return out


def extract_query_terms(query: str) -> List[str]:
    """Query -> ordered search terms, stop words dropped, each stem following its root."""
    if not query or not isinstance(query, str):
        return []

    raw_words = [w for w in _WORD_RE.findall(query.lower()) if w not in QUERY_STOP_WORDS]
    if not raw_words:
        first = query.strip().split()
        if first:
            raw_words.append(first[0].lower())

    terms: List[str] = []

    def push(term: str) -> None:
        if term and term not in terms:
            terms.append(term)

    for word in raw_words:
        push(word)
        for stem in _stem_variants(word):
            push(stem)
    return terms


def is_test_path(file_path: str) -> bool:
    """Tests mention domain vocabulary constantly; this keeps them from crowding the budget."""
    lower = file_path.lower()
    base = lower.rsplit("/", 1)[-1]
    return bool(
        _TEST_DIR_RE.search(lower)
        or _TEST_FILE_RE.search(base)
        or base.startswith("test_")
        or _TEST_SUFFIX_RE.search(base)
    )


def _split_file_list(stdout: str, target_dir: Path, check_exists: bool) -> List[str]:
    files: List[str] = []
    for line in stdout.split("\n"):
        trimmed = line.strip()
        if not trimmed or Path(trimmed).suffix not in CODE_EXTS:
            continue
        if check_exists and not (target_dir / trimmed).exists():
            continue
        files.append(trimmed)
    return files


def _list_repo_files(target_dir: Path) -> List[str]:
    """Every code file in the tree, for path matching. Empty when neither tool is available."""
    if shutil.which("rg"):
        try:
            res = subprocess.run(
                ["rg", "--files"] + _glob_args(),
                cwd=target_dir, capture_output=True, text=True, timeout=10.0,
            )
            if res.returncode == 0 and res.stdout:
                return _split_file_list(res.stdout, target_dir, False)
        except (subprocess.SubprocessError, OSError):
            pass

    if shutil.which("git"):
        try:
            res = subprocess.run(
                ["git", "ls-files"],
                cwd=target_dir, capture_output=True, text=True, timeout=10.0,
            )
            if res.returncode == 0 and res.stdout:
                return _split_file_list(res.stdout, target_dir, True)
        except (subprocess.SubprocessError, OSError):
            pass
    return []


def _find_body_matches(target_dir: Path, terms: List[str]) -> Dict[str, set]:
    """file -> set of terms matched in its contents."""
    hits: Dict[str, set] = {}

    def record(file_path: str, term: str) -> None:
        hits.setdefault(file_path, set()).add(term)

    if shutil.which("rg"):
        for term in terms:
            try:
                res = subprocess.run(
                    ["rg", "--files-with-matches", "--ignore-case", "--max-count", "1"]
                    + _glob_args() + [term],
                    cwd=target_dir, capture_output=True, text=True, timeout=10.0,
                )
                if res.returncode == 0 and res.stdout:
                    for f in _split_file_list(res.stdout, target_dir, False):
                        record(f, term)
            except (subprocess.SubprocessError, OSError):
                break
        return hits

    if shutil.which("git"):
        for term in terms:
            try:
                res = subprocess.run(
                    ["git", "grep", "-l", "-i", term],
                    cwd=target_dir, capture_output=True, text=True, timeout=10.0,
                )
                if res.returncode == 0 and res.stdout:
                    for f in _split_file_list(res.stdout, target_dir, True):
                        record(f, term)
            except (subprocess.SubprocessError, OSError):
                break
    return hits


def discover_candidate_files(query: str, cwd: Optional[str] = None, limit: int = 60) -> List[str]:
    """
    Smart candidate file discovery using ripgrep (rg) with git grep / git ls-files fallback.

    Recall is a union of two signals, always both consulted and then ranked: where a term
    appears in a file's *path* and where it appears in its *body*. Gating path matching
    behind "content search found nothing" used to make a file named after the thing you
    asked for unreachable as soon as any other file mentioned the words -- which is
    precisely what test files do.
    """
    terms = extract_query_terms(query)
    if not terms:
        return []

    target_dir = Path(cwd) if cwd else Path.cwd()
    repo_files = _list_repo_files(target_dir)
    # Path matching is in-memory, so every term is used; only content search is capped.
    body_hits = _find_body_matches(target_dir, terms[:MAX_SEARCH_TERMS])

    scored: Dict[str, Dict[str, set]] = {}

    def entry_for(file_path: str) -> Dict[str, set]:
        return scored.setdefault(file_path, {"path": set(), "body": set()})

    for file_path in repo_files:
        lower = file_path.lower()
        for term in terms:
            if term in lower:
                entry_for(file_path)["path"].add(term)

    for file_path, matched_terms in body_hits.items():
        entry_for(file_path)["body"].update(matched_terms)

    ranked = []
    for file_path, hit in scored.items():
        if not hit["path"] and not hit["body"]:
            continue
        score = PATH_HIT_WEIGHT * len(hit["path"]) + len(hit["body"])
        if is_test_path(file_path):
            score *= TEST_FILE_FACTOR
        ranked.append((score, file_path.count("/"), file_path))

    if not ranked:
        return repo_files[:NO_MATCH_SAMPLE]

    ranked.sort(key=lambda r: (-r[0], r[1], r[2]))
    return [file_path for _, _, file_path in ranked[:limit]]
