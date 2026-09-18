"""Terminal output formatting and Rich visual rendering for lfm-rerank."""

from __future__ import annotations

import json
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .models import GroundTruthStatus, RerankResponse, RerankResultItem, Telemetry

console = Console()


def format_status_badge(status: GroundTruthStatus, below_threshold: bool = False) -> Text:
    """Return a color-coded Rich text badge for ground truth status."""
    if below_threshold:
        return Text("⚠ FLOOR (SUB-THR)", style="dim yellow")
    if status == GroundTruthStatus.VERIFIED:
        return Text("✔ VERIFIED", style="bold green")
    elif status == GroundTruthStatus.SYMBOL_NOT_FOUND:
        return Text("⚠ SYMBOL MISSING", style="bold yellow")
    elif status == GroundTruthStatus.INVALID_LINE_RANGE:
        return Text("✖ BAD LINES", style="bold red")
    elif status == GroundTruthStatus.FILE_NOT_FOUND:
        return Text("✖ NOT ON DISK", style="bold red")
    elif status == GroundTruthStatus.AMBIGUOUS_COMPLETION:
        return Text("✖ AMBIGUOUS", style="bold magenta")
    elif status == GroundTruthStatus.PARSE_ERROR:
        return Text("✖ ERROR", style="bold magenta")
    elif status == GroundTruthStatus.UNVERIFIED_RAW_TEXT:
        return Text("ℹ RAW TEXT", style="dim cyan")
    return Text(str(status), style="dim")


def render_results_table(response: RerankResponse) -> Table:
    """Generate a clean Rich Table for rerank results above threshold."""
    if response.telemetry.abstained:
        status_note = f" (Abstained: Top score < {response.telemetry.abstain_threshold:.2f})"
    elif response.telemetry.fallback_floor_triggered:
        status_note = " (Fallback Floor Active)"
    else:
        status_note = ""

    table = Table(
        title=f"LFM Semantic Rerank Results (Query: \"{response.query}\" | Threshold: ≥{response.threshold:.2f}{status_note})",
        header_style="bold cyan",
        show_lines=True,
    )

    table.add_column("#", justify="center", style="bold", width=4)
    table.add_column("Score (Adj/Raw)", justify="center", width=16)
    table.add_column("Citation (File : Lines)", style="cyan", overflow="fold")
    table.add_column("Symbol / AST Definition", style="green", overflow="ellipsis")
    table.add_column("Type", justify="center", width=8)
    table.add_column("Source", justify="center", width=12)
    table.add_column("Ground Truth", justify="center", width=18)

    if not response.results and response.telemetry.abstained:
        table.add_row("—", "[dim yellow]ABSTAINED[/dim yellow]", "[dim]No candidates cleared confidence threshold[/dim]", "—", "—", "—", "[dim yellow]ABSTAINED[/dim yellow]")
        return table

    for rank, item in enumerate(response.results, start=1):
        prob_pct = f"{item.adjusted_score * 100:.1f}%"
        if item.below_threshold:
            score_style = "dim yellow"
        elif item.adjusted_score >= 0.80:
            score_style = "bold green"
        elif item.adjusted_score >= 0.65:
            score_style = "bold yellow"
        else:
            score_style = "dim"

        if abs(item.delta) > 1e-4:
            delta_str = f" [dim]({item.delta:+.2f})[/dim]"
            score_line = f"{item.adjusted_score:.3f}{delta_str}\n[dim]raw: {item.raw_score:.3f}[/dim]"
        else:
            score_line = f"{item.adjusted_score:.3f}\n({prob_pct})"

        score_text = Text.from_markup(f"[{score_style}]{score_line}[/{score_style}]", justify="center")

        loc = f"{item.citation.file}:{item.citation.start_line}-{item.citation.end_line}"
        symbol_str = item.symbol or item.citation.symbol or "—"

        type_badge = Text("TEST", style="bold magenta") if item.is_test else Text("CODE", style="dim")

        if item.cached:
            src_text = Text("⚡ CACHED", style="bold cyan")
        else:
            src_text = Text(f"LIVE ({item.duration_ms:.0f}ms)", style="dim")

        status_badge = format_status_badge(item.ground_truth_status, below_threshold=item.below_threshold)
        table.add_row(str(rank), score_text, loc, symbol_str, type_badge, src_text, status_badge)

    return table


def render_manifest_table(response: RerankResponse) -> Table:
    """Render full candidate manifest table defending against silent omission."""
    table = Table(
        title="📋 Candidate Manifest (Full File Evaluation Roster)",
        header_style="bold magenta",
        show_lines=False,
    )
    table.add_column("Status", justify="center", width=10)
    table.add_column("Type", justify="center", width=8)
    table.add_column("Max P(Yes)", justify="center", width=12)
    table.add_column("Chunks", justify="center", width=8)
    table.add_column("File Path", style="dim cyan", overflow="fold")
    table.add_column("Identified Symbols", style="dim", overflow="ellipsis")

    for entry in response.candidate_manifest:
        if entry.status == "included":
            status_text = Text("INCLUDED", style="bold green")
            score_style = "bold green" if entry.max_score >= 0.80 else "yellow"
        else:
            status_text = Text("OMITTED", style="dim")
            score_style = "dim"

        type_text = Text("TEST", style="magenta") if entry.is_test else Text("CODE", style="dim")
        score_text = Text(f"{entry.max_score:.3f}", style=score_style)
        sym_preview = ", ".join(entry.symbols[:4])
        if len(entry.symbols) > 4:
            sym_preview += f" (+{len(entry.symbols) - 4} more)"

        table.add_row(
            status_text,
            type_text,
            score_text,
            str(entry.chunks_count),
            entry.file_path,
            sym_preview or "—",
        )

    return table


def render_telemetry_box(t: Telemetry) -> Panel:
    """Render telemetry in Claude CLI style bordered box."""
    cache_ratio = f"{t.cache_hits}/{t.chunks_evaluated}"
    floor_str = " (triggered)" if t.fallback_floor_triggered else " (none)"
    abstain_str = f" | Abstention: [bold red]Triggered (<{t.abstain_threshold:.2f})[/bold red]" if t.abstained else ""
    intent_str = f" | Intent: [bold yellow]{t.query_intent.value}[/bold yellow] (conf={t.intent_confidence:.1f})" if t.query_intent else ""
    p95_str = f"{t.score_latency_p95:.1f}ms" if t.score_latency_p95 > 0 else "0.0ms (cached)"
    content = (
        f"[bold white]Evaluated:[/bold white] {t.chunks_evaluated} semantic chunks across {t.candidate_files_count} files (threshold ≥ {t.threshold:.2f}, floor={floor_str}{intent_str}{abstain_str})\n"
        f"[bold white]Persistent Cache:[/bold white] [cyan]{t.cache_hits} hits[/cyan] ({t.cache_misses} misses) → [dim]hit rate: {t.cache_hit_rate * 100:.1f}% ({cache_ratio})[/dim] | [dim]p95 latency: {p95_str}[/dim]\n"
        f"[bold white]Operational Rates:[/bold white] Yes: [green]{t.yes_variant_rate * 100:.1f}%[/green] | No: [yellow]{t.no_variant_rate * 100:.1f}%[/yellow] | Ambiguous: [magenta]{t.ambiguous_completion_rate * 100:.1f}%[/magenta] ({t.ambiguous_completions})\n"
        f"[bold white]Token Ingestion:[/bold white] [cyan]{t.total_prompt_tokens:,}[/cyan] prefill tokens | [magenta]{t.total_completion_tokens:,}[/magenta] decode tokens (1 token/chunk!)\n"
        f"[bold white]Throughput:[/bold white] [green]{t.prefill_tokens_per_sec:,.1f} tok/s[/green] prefill | [yellow]{t.decode_tokens_per_sec:,.1f} tok/s[/yellow] decode\n"
        f"[bold white]Total Wall-Clock Time:[/bold white] [bold cyan]{t.total_wall_time_s:.3f}s[/bold cyan]\n"
        f"[bold white]Frontier Token Reduction:[/bold white] [bold green]{t.reduction_ratio:.1f}x[/bold green] "
        f"([green]{t.reduction_percentage:.1f}% reduction[/green] - from {t.total_prompt_tokens:,} to {t.top_k_input_tokens:,} tok)"
    )
    model_label = t.model_id.upper() if t.model_id else "LFM"
    return Panel(
        content,
        title=f"[bold cyan]⚡ {model_label} Telemetry & Throughput[/bold cyan]",
        border_style="dim cyan",
    )


def print_results(response: RerankResponse, as_json: bool = False, show_manifest: bool = True) -> None:
    """Print results either as formatted terminal table or JSON."""
    if as_json:
        print(response.model_dump_json(indent=2))
        return

    table = render_results_table(response)
    console.print(table)
    console.print("")

    if show_manifest and response.candidate_manifest:
        manifest_table = render_manifest_table(response)
        console.print(manifest_table)
        console.print("")

    telemetry_panel = render_telemetry_box(response.telemetry)
    console.print(telemetry_panel)
