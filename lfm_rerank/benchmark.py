"""Benchmarking suite for LFM Semantic Co-Processor Wide Reranker."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

from rich.console import Console
from rich.table import Table

from .cache import RerankCache
from .client import LFMReranker
from .models import RerankResponse

console = Console()


async def run_benchmark_trial(
    endpoint: str,
    query: str,
    candidates: Sequence[str],
    concurrency: int,
    threshold: float = 0.50,
    top_k: Optional[int] = None,
    cache: Optional[RerankCache] = None,
) -> RerankResponse:
    """Run a single benchmark trial with specified concurrency."""
    reranker = LFMReranker(endpoint=endpoint, concurrency=concurrency, cache=cache)
    return await reranker.rerank(query=query, candidates=candidates, threshold=threshold, top_k=top_k)


def render_benchmark_report(responses: List[RerankResponse], labels: List[str]) -> None:
    """Render a side-by-side comparison table of benchmark runs."""
    table = Table(
        title="⚡ LFM Wide Reranker Benchmark (1-Token Logprob Scoring & Caching)",
        header_style="bold magenta",
        show_lines=True,
    )
    table.add_column("Trial / Mode", justify="center", style="bold")
    table.add_column("Concurrency", justify="center")
    table.add_column("Chunks (Hits)", justify="center")
    table.add_column("Prefill Tok", justify="right", style="cyan")
    table.add_column("Decode Tok", justify="right", style="yellow")
    table.add_column("Prefill (tok/s)", justify="right", style="bold green")
    table.add_column("Wall Time (s)", justify="right", style="bold cyan")
    table.add_column("Frontier Reduction", justify="center", style="bold green")

    for label, resp in zip(labels, responses):
        t = resp.telemetry
        chunk_info = f"{t.chunks_evaluated} ({t.cache_hits} cached)"
        table.add_row(
            label,
            str(t.concurrency),
            chunk_info,
            f"{t.total_prompt_tokens:,}",
            f"{t.total_completion_tokens:,}",
            f"{t.prefill_tokens_per_sec:,.1f}",
            f"{t.total_wall_time_s:.3f}s",
            f"{t.reduction_ratio:.1f}x (-{t.reduction_percentage:.0f}%)",
        )

    console.print(table)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="lfm-rerank-bench",
        description="Benchmark LFM Wide Reranker 1-token logprob scoring and caching",
    )
    parser.add_argument(
        "--endpoint",
        "-e",
        type=str,
        default=os.environ.get("LFM_ENDPOINT", "http://localhost:8034/v1"),
        help="LFM llama-server base endpoint",
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default="kv cache memory allocation and init",
        help="Benchmark query",
    )
    parser.add_argument(
        "candidates",
        nargs="*",
        help="Candidate files to test against.",
    )
    parser.add_argument(
        "--concurrency-levels",
        type=str,
        default="1,2,4",
        help="Comma-separated concurrency levels to test",
    )

    args = parser.parse_args()

    candidates = args.candidates
    if not candidates:
        search_dirs = [
            Path("/home/shawry/llama.cpp/src"),
            Path("/home/shawry/qc-mono-snapshot-latest/backend/src"),
            Path("/home/shawry/lfm-kit/reranker/lfm_rerank"),
        ]
        candidate_paths = []
        for s_dir in search_dirs:
            if s_dir.exists() and s_dir.is_dir():
                found = list(s_dir.glob("*.cpp")) + list(s_dir.glob("*.ts")) + list(s_dir.glob("*.py"))
                for f in found[:8]:
                    candidate_paths.append(str(f))
                if candidate_paths:
                    break
        candidates = candidate_paths[:8]

    console.print(f"[bold cyan]Running LFM Logprob Reranker Benchmark[/bold cyan] with {len(candidates)} candidate files...")
    console.print(f"Endpoint: {args.endpoint}")
    console.print(f"Query: \"{args.query}\"")

    concurrency_list = [int(x.strip()) for x in args.concurrency_levels.split(",") if x.strip()]
    responses = []
    labels = []

    # Test fresh runs across concurrency levels (with temporary in-memory/isolated cache)
    for c in concurrency_list:
        trial_cache = RerankCache(db_path=Path("/tmp/lfm_bench_temp.db"))
        trial_cache.clear()
        console.print(f"\n[dim]Testing cold pass (concurrency={c})...[/dim]")
        resp = asyncio.run(
            run_benchmark_trial(
                endpoint=args.endpoint,
                query=args.query,
                candidates=candidates,
                concurrency=c,
                cache=trial_cache,
            )
        )
        responses.append(resp)
        labels.append(f"Cold (c={c})")

    # Test warm cached pass
    warm_cache = RerankCache(db_path=Path("/tmp/lfm_bench_temp.db"))
    console.print("\n[dim]Testing warm cached pass...[/dim]")
    warm_resp = asyncio.run(
        run_benchmark_trial(
            endpoint=args.endpoint,
            query=args.query,
            candidates=candidates,
            concurrency=4,
            cache=warm_cache,
        )
    )
    responses.append(warm_resp)
    labels.append("Warm (Cached)")

    render_benchmark_report(responses, labels)
    return 0


if __name__ == "__main__":
    sys.exit(main())
