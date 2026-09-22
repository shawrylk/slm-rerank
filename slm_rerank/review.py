"""Grounded code review: a finding is reported only when its evidence is on disk.

The reranker answers "which chunk is relevant" with a calibrated probability. It does
not write prose, so it cannot invent a finding. A generative model can, and a review
that reports an invented finding is worse than no review at all.

This module closes that gap. It takes the reranker's verified chunks, asks the model for
candidate findings through the native ``/completion`` endpoint only (``HANDOFF.md``:
``/v1/chat/completions`` applies a chat template and is not usable here), and then
checks each finding against the physical file. A finding survives only if the model
copied an evidence line that exists in the cited chunk. Everything else is dropped and
counted.

Every failure path returns fewer findings rather than raising, so a missing, slow or
confused model degrades the review; it never fabricates one.
"""
from __future__ import annotations

import argparse
import asyncio
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import httpx
from pydantic import BaseModel, Field

from .client import LFMReranker
from .expander import _completion_url, _extract_text
from .models import Citation, GroundTruthStatus, QueryIntent, RerankResultItem

DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_CLAIMS = 4
MIN_EVIDENCE_CHARS = 12
DEFAULT_SUPPORT_THRESHOLD = 0.5

# Reasons a candidate finding is not reported. Stable strings, so a caller can branch on them.
REASON_EMPTY_EVIDENCE = "EMPTY_EVIDENCE"
REASON_EVIDENCE_TOO_SHORT = "EVIDENCE_TOO_SHORT"
REASON_EVIDENCE_NOT_FOUND = "EVIDENCE_NOT_FOUND"
REASON_CHUNK_UNREADABLE = "CHUNK_UNREADABLE"
REASON_EMPTY_CLAIM = "EMPTY_CLAIM"
REASON_CLAIM_NOT_SUPPORTED = "CLAIM_NOT_SUPPORTED"
REASON_SUPPORT_UNVERIFIED = "SUPPORT_UNVERIFIED"
REASON_CLAIM_SYMBOL_NOT_IN_CHUNK = "CLAIM_SYMBOL_NOT_IN_CHUNK"

# Identifiers a claim names: a backticked token, a call, or a camelCase/PascalCase/snake_case name.
_IDENTIFIER_RE = re.compile(r"`([^`]{2,})`|\b([A-Za-z_][A-Za-z0-9_]*)\s*\(|\b([A-Za-z_][A-Za-z0-9_]*)\b")
# Words a bare match picks up that are not code identifiers.
_IDENTIFIER_STOP_WORDS = {"set", "get", "if", "for", "while", "return", "print", "the", "and", "not"}

_FINDING_RE = re.compile(r"^\s*FINDING:\s*(?P<claim>.+?)\s*\|\s*EVIDENCE:\s*(?P<evidence>.+?)\s*$", re.IGNORECASE)


class ReviewClaim(BaseModel):
    """A candidate finding the model produced, before any check."""

    text: str
    evidence: str
    citation: Citation
    candidate_id: str = ""


class ReviewFinding(BaseModel):
    """A finding whose evidence was found in the cited chunk on disk."""

    text: str
    evidence: str
    citation: Citation
    evidence_line: Optional[int] = None


class DroppedClaim(BaseModel):
    """A candidate the verifier did not report, with the reason."""

    text: str
    evidence: str
    citation: Citation
    reason: str


class ReviewReport(BaseModel):
    query: str
    findings: List[ReviewFinding] = Field(default_factory=list)
    dropped: List[DroppedClaim] = Field(default_factory=list)
    chunks_reviewed: int = 0
    claims_generated: int = 0
    claims_verified: int = 0
    claims_dropped: int = 0
    model_failures: int = 0
    semantic_gate: bool = False
    support_threshold: Optional[float] = None


def normalize_ws(value: str) -> str:
    """Collapse whitespace so a copied line matches regardless of indentation."""
    return " ".join(value.split())


def strip_evidence(value: str) -> str:
    """Drop the wrapping a model adds to a copied line: trailing separator, quotes, backticks, a leading dash."""
    text = value.strip().rstrip("|").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("`", '"', "'"):
        text = text[1:-1].strip()
    return text.lstrip("-* ").strip()


def format_review_prompt(query: str, file_path: str, start_line: int, end_line: int, content: str) -> str:
    """ChatML with the reasoning bypass, matching the scorer's and expander's prompt shape."""
    return (
        "<|startoftext|><|im_start|>system\n"
        "You review one code chunk. Report only a defect you can prove with a line copied "
        "from the chunk.\n"
        "Write each finding on its own line, exactly:\n"
        "FINDING: <one short sentence> | EVIDENCE: <a line copied verbatim from the chunk>\n"
        "If the chunk has no defect, reply exactly: NONE\n"
        "Never describe code that is not shown. Never guess.\n"
        "<|im_end|>\n"
        f"<|im_start|>user\nFile: {file_path} lines {start_line}-{end_line}\n"
        f"Review question: {query}\n"
        f"Chunk:\n{content}\n<|im_end|>\n"
        "<|im_start|>assistant\n thinking\n</think>\n"
    )


def parse_review_claims(text: str, citation: Citation, candidate_id: str = "") -> List[ReviewClaim]:
    """Pull ``FINDING: ... | EVIDENCE: ...`` lines out of whatever the endpoint returned."""
    if not text or not isinstance(text, str):
        return []
    claims: List[ReviewClaim] = []
    for line in text.splitlines():
        match = _FINDING_RE.match(line)
        if not match:
            continue
        claim_text = match.group("claim").strip()
        evidence = strip_evidence(match.group("evidence"))
        if not claim_text:
            continue
        claims.append(ReviewClaim(text=claim_text, evidence=evidence, citation=citation, candidate_id=candidate_id))
    return claims


def read_citation_lines(citation: Citation) -> Optional[List[str]]:
    """Return the physical lines a citation names, or ``None`` when it cannot be read."""
    if not citation.file:
        return None
    path = Path(citation.file)
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8", newline="", errors="replace") as handle:
            lines = handle.readlines()
    except OSError:
        return None
    if citation.start_line < 1 or citation.end_line < citation.start_line or citation.end_line > len(lines):
        return None
    return lines[citation.start_line - 1 : citation.end_line]


def locate_evidence(evidence: str, lines: Sequence[str], start_line: int) -> Optional[int]:
    """Find the line the evidence was copied from, or ``None`` when it is not there.

    A model often collapses a multi-line quote onto one line, so the evidence is matched
    against each line first and then against the whole chunk with whitespace collapsed.
    """
    needle = normalize_ws(evidence)
    if not needle:
        return None
    normalized_lines = [normalize_ws(line) for line in lines]
    for offset, line in enumerate(normalized_lines):
        if needle in line:
            return start_line + offset

    joined = ""
    ranges: List[Tuple[int, int, int]] = []
    for offset, line in enumerate(normalized_lines):
        begin = len(joined)
        joined += line
        ranges.append((begin, len(joined), offset))
        joined += " "
    position = joined.find(needle)
    if position == -1:
        return None
    for begin, end, offset in ranges:
        if begin <= position <= end:
            return start_line + offset
    return None


def _is_code_identifier(token: str, explicit: bool) -> bool:
    """A backticked or called token counts; a bare token counts only when it looks like code."""
    if explicit:
        return True
    if "_" in token:
        return True
    return any(char.isupper() for char in token[1:])


def extract_claim_identifiers(text: str) -> List[str]:
    """Pull the code identifiers a claim names, so they can be checked against the chunk."""
    found: List[str] = []
    for match in _IDENTIFIER_RE.finditer(text):
        backticked, called, bare = match.groups()
        token = (backticked or called or bare or "").strip().rstrip("()").strip()
        explicit = bool(backticked or called)
        if len(token) < 3 or token.lower() in _IDENTIFIER_STOP_WORDS or token in found:
            continue
        if not _is_code_identifier(token, explicit):
            continue
        found.append(token)
    return found


def verify_claim(claim: ReviewClaim, lines: Sequence[str]) -> Tuple[Optional[ReviewFinding], Optional[DroppedClaim]]:
    """Accept the claim only when its evidence is a line of the cited chunk and every identifier it names is there."""
    evidence = strip_evidence(claim.evidence)
    if not evidence:
        return None, DroppedClaim(text=claim.text, evidence=claim.evidence, citation=claim.citation, reason=REASON_EMPTY_EVIDENCE)
    if len(normalize_ws(evidence)) < MIN_EVIDENCE_CHARS:
        return None, DroppedClaim(text=claim.text, evidence=evidence, citation=claim.citation, reason=REASON_EVIDENCE_TOO_SHORT)
    line_number = locate_evidence(evidence, lines, claim.citation.start_line)
    if line_number is None:
        return None, DroppedClaim(text=claim.text, evidence=evidence, citation=claim.citation, reason=REASON_EVIDENCE_NOT_FOUND)
    chunk_text = "\n".join(lines)
    missing = [name for name in extract_claim_identifiers(claim.text) if name not in chunk_text]
    if missing:
        return None, DroppedClaim(
            text=claim.text,
            evidence=evidence,
            citation=claim.citation,
            reason=REASON_CLAIM_SYMBOL_NOT_IN_CHUNK,
        )
    finding = ReviewFinding(text=claim.text, evidence=evidence, citation=claim.citation, evidence_line=line_number)
    return finding, None


async def _generate(completion_url: str, prompt: str, timeout: float, client: httpx.AsyncClient) -> Optional[str]:
    """Ask the model for findings. ``None`` means the call failed, never an empty finding list."""
    payload: Dict[str, Any] = {
        "prompt": prompt,
        "n_predict": 256,
        "temperature": 0,
        "top_p": 0.9,
        "stop": ["<|im_end|>"],
        "cache_prompt": True,
    }
    try:
        response = await client.post(completion_url, json=payload, timeout=timeout)
        if response.status_code != 200:
            return None
        return _extract_text(response.json())
    except (httpx.HTTPError, ValueError, TypeError):
        return None


async def _judge_support(
    engine: LFMReranker,
    claim: ReviewClaim,
    evidence: str,
    timeout: float,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
) -> Tuple[str, Optional[float]]:
    """Ask the calibrated binary scorer whether the evidence supports the claim.

    Returns ``("skipped", None)`` when the engine has no prompt profile, ``("failed", None)``
    when the call could not be made, and ``("scored", probability)`` otherwise. A caller
    treats a failure as unverified, never as support.
    """
    profile = getattr(engine, "profile", None)
    if profile is None or not hasattr(profile, "format_prompt"):
        return "skipped", None
    prompt = profile.format_prompt(
        query=f"Claim: {claim.text}",
        chunk_content=evidence,
        file_path=claim.citation.file,
        symbol=claim.citation.symbol,
        is_test=False,
        start_line=claim.citation.start_line,
        end_line=claim.citation.end_line,
    )
    payload: Dict[str, Any] = {
        "prompt": prompt,
        "n_predict": 1,
        "n_probs": 10,
        "temperature": 0.0,
        "stop": list(getattr(profile, "stop_tokens", []) or []),
    }
    async with semaphore:
        try:
            response = await client.post(_completion_url(engine.raw_endpoint), json=payload, timeout=timeout)
            if response.status_code != 200:
                return "failed", None
            score, _yes, _no, _ambiguous = profile.extract_calibrated_logprobs(
                response.json(), chunk_token_est=max(1, len(evidence.split()))
            )
            return "scored", float(score)
        except (httpx.HTTPError, ValueError, TypeError):
            return "failed", None


async def _review_chunk(
    result: RerankResultItem,
    query: str,
    engine: LFMReranker,
    timeout: float,
    max_claims: int,
    support_threshold: Optional[float],
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
) -> Tuple[List[ReviewFinding], List[DroppedClaim], int, int]:
    """Generate and verify findings for one chunk. Returns findings, dropped, generated, failed."""
    lines = read_citation_lines(result.citation)
    if lines is None:
        dropped = [
            DroppedClaim(text="", evidence="", citation=result.citation, reason=REASON_CHUNK_UNREADABLE)
        ]
        return [], dropped, 0, 0

    citation = result.citation
    content = "\n".join(f"{citation.start_line + offset}: {line.rstrip(chr(10))}" for offset, line in enumerate(lines))
    prompt = format_review_prompt(query, citation.file, citation.start_line, citation.end_line, content)

    async with semaphore:
        text = await _generate(_completion_url(engine.raw_endpoint), prompt, timeout, client)
    if text is None:
        return [], [], 0, 1

    claims = parse_review_claims(text, citation, result.candidate_id)[:max_claims]
    findings: List[ReviewFinding] = []
    dropped: List[DroppedClaim] = []
    for claim in claims:
        finding, reject = verify_claim(claim, lines)
        if finding is None:
            if reject is not None:
                dropped.append(reject)
            continue
        if support_threshold is not None:
            status, score = await _judge_support(engine, claim, finding.evidence, timeout, client, semaphore)
            if status == "failed":
                dropped.append(
                    DroppedClaim(text=claim.text, evidence=finding.evidence, citation=citation, reason=REASON_SUPPORT_UNVERIFIED)
                )
                continue
            if status == "scored" and score is not None and score < support_threshold:
                dropped.append(
                    DroppedClaim(text=claim.text, evidence=finding.evidence, citation=citation, reason=REASON_CLAIM_NOT_SUPPORTED)
                )
                continue
        findings.append(finding)
    return findings, dropped, len(claims), 0


async def review(
    query: str,
    candidates: Sequence[Union[str, Path]],
    reranker: Optional[LFMReranker] = None,
    threshold: Optional[float] = None,
    top_k: Optional[int] = None,
    intent: Optional[Union[QueryIntent, str]] = None,
    timeout: float = DEFAULT_TIMEOUT,
    max_claims: int = DEFAULT_MAX_CLAIMS,
    concurrency: int = 4,
    support_threshold: Optional[float] = DEFAULT_SUPPORT_THRESHOLD,
) -> ReviewReport:
    """Retrieve, generate and verify. Only evidence-backed findings are returned.

    ``support_threshold`` enables the semantic support gate: a finding must also be judged
    supported by the calibrated binary scorer. Set it to ``None`` to run the evidence gate
    alone, for example in an offline test.
    """
    engine = reranker or LFMReranker()
    selection = await engine.rerank(
        query=query,
        candidates=candidates,
        threshold=threshold,
        top_k=top_k,
        intent=intent,
    )
    verified = [item for item in selection.results if item.ground_truth_status == GroundTruthStatus.VERIFIED]
    report = ReviewReport(
        query=query,
        chunks_reviewed=len(verified),
        semantic_gate=support_threshold is not None,
        support_threshold=support_threshold,
    )

    semaphore = asyncio.Semaphore(max(1, concurrency))
    async with httpx.AsyncClient(timeout=timeout) as client:
        outcomes = await asyncio.gather(
            *(
                _review_chunk(item, query, engine, timeout, max_claims, support_threshold, client, semaphore)
                for item in verified
            )
        )

    for findings, dropped, generated, failed in outcomes:
        report.findings.extend(findings)
        report.dropped.extend(dropped)
        report.claims_generated += generated
        report.model_failures += failed
    report.claims_verified = len(report.findings)
    report.claims_dropped = len(report.dropped)
    return report


def review_sync(
    query: str,
    candidates: Sequence[Union[str, Path]],
    reranker: Optional[LFMReranker] = None,
    threshold: Optional[float] = None,
    top_k: Optional[int] = None,
    intent: Optional[Union[QueryIntent, str]] = None,
    timeout: float = DEFAULT_TIMEOUT,
    max_claims: int = DEFAULT_MAX_CLAIMS,
    concurrency: int = 4,
    support_threshold: Optional[float] = DEFAULT_SUPPORT_THRESHOLD,
) -> ReviewReport:
    """Synchronous wrapper for :func:`review`."""
    return asyncio.run(
        review(
            query=query,
            candidates=candidates,
            reranker=reranker,
            threshold=threshold,
            top_k=top_k,
            intent=intent,
            timeout=timeout,
            max_claims=max_claims,
            concurrency=concurrency,
            support_threshold=support_threshold,
        )
    )


def render_review_report(report: ReviewReport) -> str:
    """A plain-text report, for a console or a log."""
    lines = [f"Query: {report.query}"]
    lines.append(
        f"Chunks reviewed {report.chunks_reviewed} | claims generated {report.claims_generated} | "
        f"verified {report.claims_verified} | dropped {report.claims_dropped} | model failures {report.model_failures}"
    )
    for finding in report.findings:
        location = f"{finding.citation.file}:{finding.evidence_line or finding.citation.start_line}"
        lines.append(f"  [verified] {location} - {finding.text}")
    for dropped in report.dropped:
        lines.append(f"  [dropped:{dropped.reason}] {dropped.text or dropped.citation.file}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="slm-rerank-review",
        description="Grounded code review: a finding is reported only when its evidence is on disk.",
    )
    parser.add_argument("paths", nargs="+", help="Files or globs to review.")
    parser.add_argument("-q", "--query", default="Review these files for real defects.", help="What to look for.")
    parser.add_argument("-e", "--endpoint", default=None, help="Model endpoint, e.g. http://localhost:8034/v1")
    parser.add_argument("-m", "--model", default=None, help="Model profile name, e.g. lfm")
    parser.add_argument("-t", "--threshold", type=float, default=None, help="Retrieval threshold.")
    parser.add_argument("-k", "--top", type=int, default=5, help="Chunks to review.")
    parser.add_argument(
        "--support-threshold",
        type=float,
        default=DEFAULT_SUPPORT_THRESHOLD,
        help="Minimum support probability from the binary judge.",
    )
    parser.add_argument("--no-support", action="store_true", help="Run the evidence gate only; skip the semantic judge.")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    engine = LFMReranker(endpoint=args.endpoint, model=args.model) if (args.endpoint or args.model) else LFMReranker()
    support = None if args.no_support else args.support_threshold
    report = review_sync(
        query=args.query,
        candidates=args.paths,
        reranker=engine,
        threshold=args.threshold,
        top_k=args.top,
        support_threshold=support,
    )
    if args.json:
        print(report.model_dump_json(indent=2))
    else:
        print(render_review_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())