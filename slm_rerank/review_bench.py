"""Trust benchmark for the grounded review layer.

It answers one question: of the findings that reach the reader, how many point at code
that is actually there?

The adversarial corpus is not invented for the benchmark. It carries the fabricated
findings the local model produced during the quality-control-mono guide-tour review and
the earlier ad-hoc swarm, each paired with evidence that is absent from the target file.
A trustworthy pipeline rejects all of them. The supported corpus carries findings whose
evidence is a real line, and those must survive.

The offline run is deterministic and needs no model: it drives the verifier directly. The
optional live run sends the file through retrieval and generation and reports how many
model claims survived verification.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from rich.console import Console
from rich.table import Table

from .models import Citation
from .review import ReviewClaim, ReviewFinding, DroppedClaim, read_citation_lines, verify_claim

console = Console()

FIXTURE_NAME = "counter.tsx"
FIXTURE_SOURCE = """import { useEffect, useState } from "react";

export function Counter(): React.JSX.Element {
  const [count, setCount] = useState(0);

  useEffect(() => {
    setCount(count + 1);
  }, []);

  return <button onClick={() => setCount(count + 1)}>{count}</button>;
}
"""

# Findings whose evidence is a line of FIXTURE_SOURCE. They must be reported.
SUPPORTED: Sequence[Tuple[str, str]] = (
    ("State update in an empty-dependency effect reads a stale count", "setCount(count + 1);"),
    ("The click handler closes over count instead of using the updater form", "onClick={() => setCount(count + 1)}"),
)

# Findings the local model produced with evidence that is not in the file. They must be dropped.
FABRICATED: Sequence[Tuple[str, str]] = (
    ("onSkip is invoked twice, suppressing the flag twice", "onSkip(suppress); onSkip(suppress);"),
    ("The close button has no accessible name", "aria-label={closeLabel}"),
    ("A second error boundary swallows route errors", "<ShellErrorBoundary>"),
    ("The dialog never traps focus", 'role="dialog" aria-modal="true"'),
    ("The suppress checkbox lacks aria-hidden", '<input type="checkbox"'),
    ("The start modal renders before the router is ready", "const [started, setStarted] = useState(false);"),
)

# Findings with real evidence that nevertheless name a symbol the file does not contain.
# The identifier gate must drop them.
SYMBOL_INVENTED: Sequence[Tuple[str, str]] = (
    ("onSkip is invoked twice here", "setCount(count + 1);"),
    ("ShellErrorBoundary swallows a route error", "useEffect(() => {"),
    ("closeLabel has no accessible name", "const [count, setCount] = useState(0);"),
)


class BenchOutcome:
    def __init__(self, source_path: Path) -> None:
        self.source_path = source_path
        self.lines = read_citation_lines(Citation(file=str(source_path), start_line=1, end_line=len(FIXTURE_SOURCE.splitlines()))) or []
        self.findings: List[ReviewFinding] = []
        self.dropped: List[DroppedClaim] = []

    def run(self, claims: Sequence[Tuple[str, str]]) -> None:
        for index, (text, evidence) in enumerate(claims):
            claim = ReviewClaim(
                text=text,
                evidence=evidence,
                citation=Citation(file=str(self.source_path), start_line=1, end_line=len(self.lines), symbol="Counter"),
                candidate_id=f"bench:{index}",
            )
            finding, dropped = verify_claim(claim, self.lines)
            if finding is not None:
                self.findings.append(finding)
            elif dropped is not None:
                self.dropped.append(dropped)


def run_offline(source_path: Path) -> Tuple[BenchOutcome, BenchOutcome, BenchOutcome]:
    supported = BenchOutcome(source_path)
    supported.run(SUPPORTED)
    fabricated = BenchOutcome(source_path)
    fabricated.run(FABRICATED)
    symbol_invented = BenchOutcome(source_path)
    symbol_invented.run(SYMBOL_INVENTED)
    return supported, fabricated, symbol_invented


def render_offline(supported: BenchOutcome, fabricated: BenchOutcome, symbol_invented: BenchOutcome) -> int:
    accepted_supported = len(supported.findings)
    accepted_fabricated = len(fabricated.findings)
    rejected_fabricated = len(fabricated.dropped)
    accepted_symbol = len(symbol_invented.findings)
    rejected_symbol = len(symbol_invented.dropped)
    total_reported = accepted_supported + accepted_fabricated + accepted_symbol
    precision = accepted_supported / total_reported if total_reported else 1.0
    total_claims = len(SUPPORTED) + len(FABRICATED) + len(SYMBOL_INVENTED)
    unguarded_precision = len(SUPPORTED) / total_claims

    table = Table(title="Grounded review - trust benchmark (offline)", header_style="bold magenta", show_lines=True)
    table.add_column("Corpus", style="bold")
    table.add_column("Claims", justify="right")
    table.add_column("Reported", justify="right", style="green")
    table.add_column("Dropped", justify="right", style="red")
    table.add_row("Supported (evidence on disk)", str(len(SUPPORTED)), str(accepted_supported), str(len(SUPPORTED) - accepted_supported))
    table.add_row("Fabricated (evidence absent)", str(len(FABRICATED)), str(accepted_fabricated), str(rejected_fabricated))
    table.add_row("Symbol-invented (symbol absent)", str(len(SYMBOL_INVENTED)), str(accepted_symbol), str(rejected_symbol))
    console.print(table)

    summary = Table(title="Metrics", show_lines=False, header_style="bold")
    summary.add_column("Metric", style="bold")
    summary.add_column("Value", justify="right")
    summary.add_row("Supported findings recalled", f"{accepted_supported}/{len(SUPPORTED)}")
    summary.add_row("Fabricated findings rejected", f"{rejected_fabricated}/{len(FABRICATED)}")
    summary.add_row("Symbol-invented findings rejected", f"{rejected_symbol}/{len(SYMBOL_INVENTED)}")
    summary.add_row("Reported precision (grounded)", f"{precision:.2%}")
    summary.add_row("Reported precision (no verifier)", f"{unguarded_precision:.2%}")
    console.print(summary)

    failures: List[str] = []
    if accepted_supported != len(SUPPORTED):
        failures.append(f"lost {len(SUPPORTED) - accepted_supported} supported finding(s)")
    if accepted_fabricated:
        failures.append(f"reported {accepted_fabricated} fabricated finding(s)")
    if accepted_symbol:
        failures.append(f"reported {accepted_symbol} symbol-invented finding(s)")
    if failures:
        console.print(f"[bold red]FAIL:[/bold red] " + "; ".join(failures))
        return 1
    console.print("[bold green]PASS:[/bold green] every reported finding has evidence on disk and names only symbols in the chunk.")
    return 0


async def run_live(source_path: Path, endpoint: Optional[str], model: Optional[str], top_k: int) -> int:
    from .client import LFMReranker
    from .review import review

    engine = LFMReranker(endpoint=endpoint, model=model) if (endpoint or model) else LFMReranker()
    report = await review(
        query="review this file for real defects",
        candidates=[str(source_path)],
        reranker=engine,
        top_k=top_k,
    )
    table = Table(title="Grounded review - live run", header_style="bold magenta", show_lines=True)
    table.add_column("Chunks", justify="right")
    table.add_column("Claims", justify="right")
    table.add_column("Verified", justify="right", style="green")
    table.add_column("Dropped", justify="right", style="red")
    table.add_column("Model failures", justify="right")
    table.add_row(
        str(report.chunks_reviewed),
        str(report.claims_generated),
        str(report.claims_verified),
        str(report.claims_dropped),
        str(report.model_failures),
    )
    console.print(table)
    for finding in report.findings:
        console.print(f"[green]verified[/green] {finding.citation.file}:{finding.evidence_line} - {finding.text}")
    for dropped in report.dropped:
        console.print(f"[red]dropped[/red] {dropped.reason} - {dropped.text or dropped.citation.file}")
    if report.claims_generated and report.model_failures == 0:
        console.print("Every generated claim was checked against disk before it was reported.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="slm-rerank-review-bench", description="Trust benchmark for grounded code review.")
    parser.add_argument("--live", action="store_true", help="Also run retrieval and generation against the local model.")
    parser.add_argument("--endpoint", default=None, help="Model endpoint for the live run, e.g. http://localhost:8034/v1")
    parser.add_argument("--model", default=None, help="Model profile name for the live run, e.g. lfm")
    parser.add_argument("--top-k", type=int, default=1, help="Chunks to review in the live run.")
    parser.add_argument("--json", action="store_true", help="Print the offline result as JSON.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp:
        source_path = Path(tmp) / FIXTURE_NAME
        source_path.write_text(FIXTURE_SOURCE, encoding="utf-8")
        supported, fabricated, symbol_invented = run_offline(source_path)
        if args.json:
            print(
                json.dumps(
                    {
                        "supported_total": len(SUPPORTED),
                        "supported_reported": len(supported.findings),
                        "fabricated_total": len(FABRICATED),
                        "fabricated_reported": len(fabricated.findings),
                        "fabricated_rejected": len(fabricated.dropped),
                        "symbol_invented_total": len(SYMBOL_INVENTED),
                        "symbol_invented_reported": len(symbol_invented.findings),
                        "symbol_invented_rejected": len(symbol_invented.dropped),
                    }
                )
            )
        else:
            code = render_offline(supported, fabricated, symbol_invented)
            if code != 0:
                return code
        if args.live:
            import asyncio

            return asyncio.run(run_live(source_path, args.endpoint, args.model, args.top_k))
    return 0


if __name__ == "__main__":
    sys.exit(main())