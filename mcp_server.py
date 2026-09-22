#!/usr/bin/env python3
"""Model Context Protocol (MCP) server for SLM Wide Reranker (slm-rerank).

Exposes fast AST-chunked semantic reranking using local small language models (LFM, Qwen, etc.)
to optimize token usage and context precision for frontier AI agents.
"""

from __future__ import annotations

import asyncio
import glob
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from mcp.server.mcpserver import MCPServer

from slm_rerank import LFMReranker, QueryIntent
from slm_rerank.discovery import resolve_endpoint_env, resolve_host_env

# Configure logging
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("slm_rerank_mcp")

IGNORE_DIRS = {
    ".git",
    "node_modules",
    "dist",
    "build",
    ".cache",
    ".next",
    "__pycache__",
    ".turbo",
    ".pytest_cache",
    "venv",
    ".venv",
}

mcp = MCPServer("slm-reranker")


def resolve_candidate_paths(patterns_or_paths: List[str], max_files: int = 200) -> List[str]:
    """Expand globs, directory walks, and file paths while ignoring build/dep directories."""
    resolved: List[str] = []
    seen = set()

    for pat in patterns_or_paths:
        pat_str = str(pat).strip()
        if not pat_str:
            continue

        p = Path(pat_str)

        # Direct file
        if p.is_file():
            resolved_p = str(p.resolve())
            if resolved_p not in seen:
                seen.add(resolved_p)
                resolved.append(resolved_p)
            continue

        # Directory: walk files
        if p.is_dir():
            for root, dirs, files in os.walk(p):
                dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
                for f in files:
                    file_p = Path(root) / f
                    resolved_p = str(file_p.resolve())
                    if resolved_p not in seen:
                        seen.add(resolved_p)
                        resolved.append(resolved_p)
                        if len(resolved) >= max_files:
                            return resolved
            continue

        # Glob pattern
        matches = glob.glob(pat_str, recursive=True)
        for m in matches:
            file_p = Path(m)
            if any(part in IGNORE_DIRS for part in file_p.parts):
                continue
            if file_p.is_file():
                resolved_p = str(file_p.resolve())
                if resolved_p not in seen:
                    seen.add(resolved_p)
                    resolved.append(resolved_p)
                    if len(resolved) >= max_files:
                        return resolved

    return resolved


def _default_base_url() -> str:
    """SLM_ENDPOINT/RERANKER_BASE_URL/LFM_ENDPOINT, else :8034 on SLM_HOST/RERANKER_HOST."""
    return resolve_endpoint_env() or f"http://{resolve_host_env()}:8034/v1"


DEFAULT_BASE_URL = _default_base_url()


@mcp.tool()
async def rerank_codebase(
    query: str,
    paths_or_globs: List[str],
    threshold: float = 0.65,
    top_k: int = 5,
    intent: str = "IMPLEMENTATION",
    model: str = "lfm",
    base_url: str = DEFAULT_BASE_URL,
) -> Dict[str, Any]:
    """Semantically rerank codebase files and AST chunks against a query using local SLM.

    Use this tool to find exact code symbols, definitions, and functions across many files
    without ingesting thousands of unnecessary tokens into the conversation context.

    Args:
        query: Natural language query or code task description (e.g. 'handle database pool connection').
        paths_or_globs: File paths, glob patterns (e.g. ['backend/src/**/*.ts']), or directory paths to search.
        threshold: Score threshold between 0.0 and 1.0 (default: 0.65).
        top_k: Maximum number of high-confidence results to return (default: 5).
        intent: Search intent: 'IMPLEMENTATION', 'SPECIFICATION', 'BUG_DIAGNOSIS', 'REFACTOR'.
        model: SLM model identifier ('lfm', 'qwen', 'gemma', 'rwkv', 'openai').
        base_url: Local model server endpoint (default: 'http://localhost:8034/v1').

    Returns:
        Structured rerank response containing top results with exact line citations, symbols,
        confidence scores, and token reduction statistics.
    """
    candidate_files = resolve_candidate_paths(paths_or_globs)
    if not candidate_files:
        return {
            "error": "No matching files found for candidate patterns.",
            "query": query,
            "patterns": paths_or_globs,
            "results": [],
        }

    try:
        resolved_intent = QueryIntent(intent.upper())
    except Exception:
        resolved_intent = QueryIntent.IMPLEMENTATION

    reranker = LFMReranker(model=model, base_url=base_url)

    resp = await reranker.rerank(
        query=query,
        candidates=candidate_files,
        threshold=threshold,
        top_k=top_k,
        intent=resolved_intent,
    )

    results_data = []
    for item in resp.results:
        start_l = item.citation.start_line
        end_l = item.citation.end_line
        results_data.append({
            "score": round(item.score, 3),
            "raw_score": round(item.raw_score, 3),
            "file_path": item.file_path or item.citation.file,
            "symbol": item.symbol,
            "start_line": start_l,
            "end_line": end_l,
            "line_count": (end_l - start_l + 1) if end_l >= start_l else 1,
            "snippet": item.snippet or "",
            "ground_truth": item.ground_truth_status.value if hasattr(item.ground_truth_status, "value") else str(item.ground_truth_status),
        })

    return {
        "query": query,
        "model": resp.telemetry.model_id or model,
        "files_evaluated": resp.telemetry.candidate_files_count,
        "chunks_evaluated": resp.telemetry.chunks_evaluated,
        "wall_time_seconds": round(resp.telemetry.total_wall_time_s, 3),
        "prefill_tokens_evaluated": resp.telemetry.total_prompt_tokens,
        "top_k_input_tokens": resp.telemetry.top_k_input_tokens,
        "token_reduction_ratio": f"{resp.telemetry.reduction_ratio:.1f}x",
        "token_reduction_percentage": f"{resp.telemetry.reduction_percentage:.1f}%",
        "top_results_count": len(results_data),
        "results": results_data,
    }


@mcp.tool()
async def reranker_status(base_url: str = DEFAULT_BASE_URL) -> Dict[str, Any]:
    """Check connectivity and model status of the local SLM inference server.

    Args:
        base_url: Base endpoint URL to check (default: 'http://localhost:8034/v1').
    """
    raw = base_url.rstrip("/")
    models_url = f"{raw}/models" if not raw.endswith("/models") else raw

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(models_url)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "status": "online",
                    "base_url": base_url,
                    "models_data": data,
                }
            return {
                "status": "error",
                "http_code": resp.status_code,
                "detail": resp.text,
            }
    except Exception as e:
        return {
            "status": "offline",
            "base_url": base_url,
            "error": str(e),
        }


if __name__ == "__main__":
    mcp.run("stdio")
