"""Command-line interface for Model-Agnostic Wide Reranker."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

from .boundary import detect_slice, group_by_slice
from .cache import RerankCache
from .client import LFMReranker
from .config import load_config, resolve_endpoint_and_model
from rich.console import Console
from .discovery import auto_discover_endpoint, discover_candidate_files
from .display import console, print_results
from .stubber import generate_ghost_stub
from .verifier import GroundTruthVerifier


# rich writes to stdout by default; errors belong on stderr.
error_console = Console(stderr=True)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="lfm-rerank",
        description="Model-Agnostic Semantic Wide Reranker with 1-Token Logprob Scoring & AST Chunking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--query",
        "-q",
        type=str,
        required=True,
        help="Search query or task description to score candidates against",
    )
    parser.add_argument(
        "candidates",
        nargs="*",
        help="Candidate file paths or text. If omitted, smart ripgrep auto-discovery is used.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Target model profile or provider (e.g. lfm, qwen, gemma, rwkv, openai, ollama). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--base-url",
        "--endpoint",
        "-e",
        dest="base_url",
        type=str,
        default=None,
        help="Server base endpoint URL (probes ports 8033-8040 if omitted)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Target server host (default: 127.0.0.1, or SLM_HOST/RERANKER_HOST)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (defaults to ~/.config/reranker/config.yaml)",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=None,
        help="Score relevance threshold in [0.0, 1.0] (defaults to intent baseline: 0.65 or 0.60)",
    )
    parser.add_argument(
        "--intent",
        "-i",
        type=str,
        choices=["IMPLEMENTATION", "SPECIFICATION", "BUG_DIAGNOSIS", "REFACTOR"],
        default=None,
        help="Query intent guiding threshold and prior scoring adjustments (inferred if omitted)",
    )
    parser.add_argument(
        "--top",
        "-k",
        type=int,
        default=None,
        help="Optional ceiling on number of top candidates returned",
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=4,
        help="Number of concurrent requests to saturate server slots",
    )
    parser.add_argument(
        "--stub",
        "--slice",
        dest="stub",
        action="store_true",
        help="Generate AST Ghost Stubs for top results to minimize frontier model tokens",
    )
    parser.add_argument(
        "--dirty",
        action="store_true",
        help="Filter or heavily bias scoring by git uncommitted/modified files",
    )
    parser.add_argument(
        "--git-diff",
        dest="git_diff",
        action="store_true",
        help="Boost candidates recently touched in git history",
    )
    parser.add_argument(
        "--by-slice",
        dest="by_slice",
        action="store_true",
        help="Group results by architectural vertical slice",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass SQLite persistent cache",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear persistent cache before running",
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Hide candidate manifest table from terminal display",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Drop candidates that fail physical ground-truth verification",
    )
    parser.add_argument(
        "--allow-empty-if-low-confidence",
        action="store_true",
        help="Return empty context if top candidate score is below abstention threshold rather than forcing fallback floor",
    )
    parser.add_argument(
        "--abstain-below",
        type=float,
        default=0.20,
        help="Score threshold below which to return empty context when allow-empty is enabled (default: 0.20)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Force full Tier-2 GPU evaluation of every candidate, bypassing the Tier-1 lexical pre-filter",
    )
    parser.add_argument(
        "--with-context",
        dest="with_context",
        action="store_true",
        help="Stitch up to 150 tokens of 1-hop call-graph and type context into each scoring prompt",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of Rich table",
    )

    args = parser.parse_args(argv)
    args.endpoint = args.base_url
    return args


def collect_candidate_inputs(args: argparse.Namespace) -> List[str]:
    """Collect candidate strings from CLI arguments, stdin, or smart ripgrep discovery."""
    candidates: List[str] = []

    if args.candidates:
        candidates.extend(args.candidates)

    if not sys.stdin.isatty():
        stdin_content = sys.stdin.read()
        lines = [line.strip() for line in stdin_content.splitlines() if line.strip()]
        for line in lines:
            if ":" in line:
                parts = line.split(":", 1)
                possible_path = Path(parts[0])
                if possible_path.exists() and possible_path.is_file():
                    candidates.append(parts[0])
                    continue
            candidates.append(line)

    # Feature 2: Smart ripgrep candidate auto-discovery
    if not candidates and args.query:
        discovered = discover_candidate_files(args.query)
        if discovered:
            if not args.json:
                console.print(f"[dim]🔍 Auto-discovered {len(discovered)} candidate files via ripgrep...[/dim]")
            candidates.extend(discovered)

    return candidates


def main() -> int:
    args = parse_args()
    config_path = Path(args.config) if args.config else None

    # Feature 1: Multi-port auto-discovery across 8033-8040
    if not args.base_url:
        discovered_endpoint = asyncio.run(
            auto_discover_endpoint(host=args.host, requested_model=args.model)
        )
        if discovered_endpoint.get("url"):
            args.base_url = discovered_endpoint["url"]
            args.endpoint = args.base_url
            if not args.json and discovered_endpoint.get("ok"):
                console.print(f"[dim]🎯 Auto-discovered port :{discovered_endpoint['port']} ({discovered_endpoint['model_id']})[/dim]")
        else:
            # Nothing answered: stop here rather than scoring every chunk against
            # an unreachable host and hanging on connect timeouts.
            reason = discovered_endpoint.get("reason", "No SLM model server found.")
            if args.json:
                print(json.dumps({"error": reason}))
            else:
                error_console.print(f"[bold red]Error:[/bold red] {reason}")
            return 1

    candidate_inputs = collect_candidate_inputs(args)
    if not candidate_inputs:
        error_console.print("[bold red]Error:[/bold red] No candidates found or provided.")
        error_console.print("Usage: lfm-rerank --query \"auth\" src/*.py")
        error_console.print("Or piping: git ls-files | lfm-rerank --query \"auth\" --threshold 0.65")
        return 1

    verifier = GroundTruthVerifier(strict=args.strict)
    reranker = LFMReranker(
        model=args.model,
        base_url=args.base_url,
        concurrency=args.concurrency,
        verifier=verifier,
        no_cache=args.no_cache,
        config_path=config_path,
    )

    if args.clear_cache:
        reranker.cache.clear(model_id=reranker.profile.name)
        console.print(f"[dim]Persistent cache cleared for {reranker.profile.name}.[/dim]")

    try:
        response = asyncio.run(
            reranker.rerank(
                query=args.query,
                candidates=candidate_inputs,
                threshold=args.threshold,
                top_k=args.top,
                intent=args.intent,
                strict=args.strict,
                allow_empty_if_low_confidence=args.allow_empty_if_low_confidence,
                abstain_below=args.abstain_below,
                full=args.full,
                with_context=args.with_context,
            )
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Aborted by user.[/yellow]")
        return 130
    except Exception as e:
        error_console.print(f"[bold red]Reranking failed:[/bold red] {e}")
        return 1

    # Attach slice tags to results
    for item in response.results:
        item.slice = detect_slice(getattr(item, "file_path", None))
        if args.stub:
            item.ghost_stub = generate_ghost_stub(item.file_path, item)

    if args.by_slice and not args.json:
        grouped = group_by_slice(response.results)
        console.print(f"\n[bold cyan]📦 Architecture Slices ({len(grouped)} active slices):[/bold cyan]\n")
        for s_name, group in grouped.items():
            console.print(f"  [bold green][Slice: {s_name}][/bold green] ({len(group['items'])} items, max score: {group['max_score']*100:.1f}%)")
            for idx, it in enumerate(group["items"], 1):
                sym = f"({it.symbol})" if getattr(it, 'symbol', None) else ""
                console.print(f"    #{idx} | {it.score*100:.1f}% | {it.file_path}:{it.citation.start_line}-{it.citation.end_line} {sym}")
            console.print("")
    else:
        print_results(response, as_json=args.json, show_manifest=not args.no_manifest)

        if args.stub and not args.json:
            console.print("\n[bold cyan]👻 Top Candidate Ghost Stubs (Token-Saving Skeletons):[/bold cyan]\n")
            for it in response.results[:2]:
                if hasattr(it, "ghost_stub"):
                    console.print(f"[dim]--- {it.file_path}:{it.citation.start_line} (Folded {it.ghost_stub['folded_lines']} lines) ---[/dim]")
                    stub_preview = it.ghost_stub["stub"][:800]
                    console.print(stub_preview + ("\n..." if len(it.ghost_stub["stub"]) > 800 else ""))
                    console.print("[dim]------------------------------------------------[/dim]\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
