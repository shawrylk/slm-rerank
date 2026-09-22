"""
Multi-Port Endpoint & Smart File Auto-Discovery for SLM Reranker.
Scans distinct model ports across the 8033-8040 range and discovers candidate files via ripgrep.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import urllib.request
import urllib.error

SLM_PORT_RANGE = [8033, 8034, 8035, 8036, 8037, 8038, 8039, 8040]


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
    host: str = "127.0.0.1",
    requested_model: Optional[str] = None,
    ports: List[int] = SLM_PORT_RANGE,
    timeout: float = 0.30,
) -> Dict[str, Any]:
    """
    Concurrently scan ports 8033-8040 for active SLM servers.
    If requested_model is specified, matches against model ID (e.g. 'lfm', 'qwen').
    Otherwise prioritizes 8034 (LFM) and 8033 (Qwen), falling back to first responding port.
    """
    tasks = [probe_port_async(host, p, timeout) for p in ports]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    results = [r for r in raw_results if isinstance(r, dict) and r.get("ok")]

    if not results:
        # Default fallback
        return {
            "port": 8034,
            "host": host,
            "url": f"http://{host}:8034/v1",
            "model_id": "fallback-default-lfm",
            "ok": False,
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


def discover_candidate_files(query: str, cwd: Optional[str] = None, limit: int = 60) -> List[str]:
    """
    Smart candidate file discovery using ripgrep (rg) with git ls-files fallback.
    Extracts high-signal query terms to locate matching files across large codebases in milliseconds.
    """
    target_dir = cwd or os.getcwd()
    if not query or not query.strip():
        return []

    stop_words = {
        "a", "an", "the", "and", "or", "but", "not", "in", "on", "at", "to", "of", "for",
        "with", "from", "into", "by", "as", "is", "are", "was", "were", "be", "been",
        "code", "codes", "file", "files", "function", "method", "class", "find", "search", "how", "what", "where"
    }

    import re
    words = re.findall(r"\b[a-zA-Z0-9_]{2,}\b", query.lower())
    terms = [w for w in words if w not in stop_words]
    if not terms and words:
        terms = [words[0]]

    collected: set[str] = set()

    # 1. Try ripgrep
    if shutil.which("rg"):
        for term in terms[:3]:
            if len(collected) >= limit:
                break
            try:
                cmd = [
                    "rg",
                    "--files-with-matches",
                    "--ignore-case",
                    "--max-count", "1",
                    "--glob", "!node_modules",
                    "--glob", "!.git",
                    "--glob", "!dist",
                    "--glob", "!build",
                    "--glob", "!coverage",
                    "--glob", "!__pycache__",
                    "--glob", "!.venv",
                    term,
                ]
                res = subprocess.run(cmd, cwd=target_dir, capture_output=True, text=True, timeout=3.0)
                if res.returncode == 0 and res.stdout:
                    for line in res.stdout.strip().split("\n"):
                        f = line.strip()
                        if f and os.path.isfile(os.path.join(target_dir, f)):
                            collected.add(f)
                            if len(collected) >= limit:
                                break
            except Exception:
                pass

    # 2. Fallback to git ls-files if ripgrep found nothing
    if not collected and shutil.which("git"):
        try:
            res = subprocess.run(["git", "ls-files"], cwd=target_dir, capture_output=True, text=True, timeout=3.0)
            if res.returncode == 0 and res.stdout:
                code_exts = {".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".rs", ".go", ".c", ".cpp", ".h"}
                for line in res.stdout.strip().split("\n"):
                    f = line.strip()
                    if any(f.endswith(ext) for ext in code_exts):
                        f_lower = f.lower()
                        if any(t in f_lower for t in terms):
                            collected.add(f)
                            if len(collected) >= limit:
                                break
        except Exception:
            pass

    return sorted(list(collected))
