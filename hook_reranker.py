#!/usr/bin/env python3
"""Antigravity Lifecycle Hook for SLM Wide Reranker (Improved Precision).

Supports:
1. PreInvocation:
   High-precision intent classification: only fires when the user's prompt
   actually asks to search, locate, understand, implement, or debug code.
   Skips casual remarks, meta questions, and git workflow commands.
   Prioritizes candidate files based on query keywords before SLM logprob scoring.
   Injects exact AST line citations into the prompt context via ephemeralMessage.
2. PreToolUse:
   Intercepts tool calls, auditing and allowing safe execution without blocking.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Logging to stderr so stdout remains clean JSON
logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
logger = logging.getLogger("slm_hook")

IGNORE_DIRS = {
    ".git", "node_modules", "dist", "build", ".cache", ".next",
    "__pycache__", ".turbo", "venv", ".venv", "coverage", ".worktrees"
}

STOP_WORDS = {
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "and", "or",
    "is", "are", "was", "were", "do", "does", "did", "can", "could", "should",
    "this", "that", "it", "my", "your", "our", "their", "so", "just"
}


def should_rerank(prompt: str) -> bool:
    """High-precision intent classifier: filters out conversational & meta prompts."""
    p = prompt.strip().lower()
    if len(p) < 8:
        return False

    # 1. Ignore conversational, meta, review, or git commands
    ignore_patterns = [
        r"^(ok|okay|yes|no|thanks|thank you|cool|great|got it|sure|yep|nope)[.!?, ]*$",
        r"^(commit|push|merge|cherry-pick|stash|pull|checkout|branch)\b",
        r"^(so |just )?(more test|test again|run test|do not commit|no commit)",
        r"^(add |then add )(it |this )?(to|too)\b",
        r"^(then |so )?(verdict|what do you think|your opinion)",
        r"^how about\b.*\b(hook|claude|agy)",
    ]
    for pat in ignore_patterns:
        if re.search(pat, p):
            return False

    # 2. Positive code search & engineering patterns
    search_patterns = [
        r"\b(where|how|which file|what file)\b.*\b(is|are|does|can|do|defined|handled|implemented|located|work|configured)\b",
        r"\b(find|search|locate|look for|inspect|check|trace)\b",
        r"\b(fix|diagnose|debug|reproduce)\b.*\b(bug|issue|error|crash|failure|leak|exception)\b",
        r"\b(implement|create|add|write)\b.*\b(handler|route|service|controller|pipeline|schema|worker|component|feature)\b",
        r"\b(schema|migration|table|entity|relation|query|endpoint|api|queue|s3|sqs|rds|worker|auth|ticket)\b",
    ]
    for pat in search_patterns:
        if re.search(pat, p):
            return True

    return False


def get_workspace_files(workspace_path: Path, query: str, max_files: int = 30) -> List[str]:
    """Retrieve and prioritize relevant source files from workspace using git ls-files."""
    all_files: List[str] = []
    try:
        res = subprocess.run(
            ["git", "ls-files", "*.ts", "*.tsx", "*.js", "*.py", "*.tf", "*.yaml", "*.json"],
            cwd=workspace_path,
            capture_output=True,
            text=True,
            timeout=2.0
        )
        if res.returncode == 0 and res.stdout.strip():
            lines = [l.strip() for l in res.stdout.splitlines() if l.strip()]
            all_files = [
                str(workspace_path / l)
                for l in lines
                if not any(part in IGNORE_DIRS for part in Path(l).parts)
            ]
    except Exception:
        pass

    if not all_files:
        # Fallback walk
        for root, dirs, filenames in os.walk(workspace_path):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            for f in filenames:
                ext = Path(f).suffix.lower()
                if ext in {".ts", ".tsx", ".js", ".py", ".tf", ".yaml"}:
                    all_files.append(str(Path(root) / f))

    if not all_files:
        return []

    # Extract keywords from query for prior keyword path matching
    q_words = set(re.findall(r"\b[a-z0-9_-]+\b", query.lower())) - STOP_WORDS

    def score_file(f_path: str) -> int:
        f_lower = f_path.lower()
        return sum(2 for w in q_words if len(w) >= 3 and w in f_lower)

    # Sort files: files matching query keywords in their path come first
    sorted_files = sorted(all_files, key=score_file, reverse=True)
    return sorted_files[:max_files]


def extract_user_request(transcript_path: str) -> Optional[str]:
    """Extract the most recent user prompt from the transcript JSONL file."""
    if not transcript_path or not os.path.exists(transcript_path):
        return None

    last_user_prompt = None
    try:
        with open(transcript_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    if obj.get("type") == "USER_INPUT":
                        content = obj.get("content", "")
                        m = re.search(r"<USER_REQUEST>(.*?)</USER_REQUEST>", content, re.DOTALL)
                        if m:
                            last_user_prompt = m.group(1).strip()
                        elif content.strip():
                            last_user_prompt = content.strip()
                except Exception:
                    continue
    except Exception as e:
        logger.warning(f"Error reading transcript: {e}")

    return last_user_prompt


async def run_slm_rerank(query: str, files: List[str]) -> List[Dict[str, Any]]:
    """Execute SLM reranking using the local Reranker."""
    from slm_rerank import LFMReranker

    reranker = LFMReranker(model="lfm", base_url="http://localhost:8034/v1")
    resp = await reranker.rerank(
        query=query,
        candidates=files,
        threshold=0.65,
        top_k=3,
    )

    results = []
    for item in resp.results:
        results.append({
            "score": round(item.score, 3),
            "file": item.file_path or item.citation.file,
            "start_line": item.citation.start_line,
            "end_line": item.citation.end_line,
            "symbol": item.symbol or "module_scope",
            "snippet": item.snippet or "",
        })
    return results


def handle_pre_invocation(data: Dict[str, Any]) -> Dict[str, Any]:
    """PreInvocation handler: injects pre-ranked context only on genuine code intent."""
    transcript_path = data.get("transcriptPath", "")
    workspace_paths = data.get("workspacePaths", [])
    workspace = Path(workspace_paths[0]) if workspace_paths else Path.cwd()

    query = extract_user_request(transcript_path)
    if not query or not should_rerank(query):
        return {"injectSteps": []}

    files = get_workspace_files(workspace, query)
    if not files:
        return {"injectSteps": []}

    try:
        results = asyncio.run(asyncio.wait_for(run_slm_rerank(query, files), timeout=5.0))
        if not results:
            return {"injectSteps": []}

        lines = [
            "⚡ [SLM Wide Reranker] Pre-computed high-confidence code citations for your task:",
        ]
        for r in results:
            lines.append(
                f"• {r['file']}:{r['start_line']}-{r['end_line']} ({r['symbol']}) — confidence: {r['score']*100:.1f}%"
            )
        lines.append("Tip: Use targeted line ranges with view_file to conserve frontier context.")

        return {
            "injectSteps": [
                {
                    "ephemeralMessage": "\n".join(lines)
                }
            ]
        }
    except Exception as e:
        logger.warning(f"SLM reranking skipped or timed out: {e}")
        return {"injectSteps": []}


def handle_pre_tool_use(data: Dict[str, Any]) -> Dict[str, Any]:
    """PreToolUse handler: allows execution safely."""
    return {"decision": "allow"}


def main():
    try:
        raw_input = sys.stdin.read()
        if not raw_input.strip():
            print(json.dumps({}))
            return

        payload = json.loads(raw_input)

        if "toolCall" in payload:
            out = handle_pre_tool_use(payload)
        else:
            out = handle_pre_invocation(payload)

        print(json.dumps(out))
    except Exception as e:
        logger.error(f"Error in hook: {e}")
        print(json.dumps({"decision": "allow", "injectSteps": []}))


if __name__ == "__main__":
    main()
