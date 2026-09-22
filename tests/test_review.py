"""Tests for the grounded review layer: only evidence on disk is reported."""
from __future__ import annotations

import importlib

import pytest

# `slm_rerank.review` is the exported function, so the submodule is fetched by name.
review_mod = importlib.import_module("slm_rerank.review")
from slm_rerank.models import Citation, RerankResponse, RerankResultItem, Telemetry
from slm_rerank.review import (
    REASON_CLAIM_SYMBOL_NOT_IN_CHUNK,
    REASON_EMPTY_EVIDENCE,
    REASON_EVIDENCE_NOT_FOUND,
    REASON_EVIDENCE_TOO_SHORT,
    ReviewClaim,
    format_review_prompt,
    locate_evidence,
    normalize_ws,
    parse_review_claims,
    read_citation_lines,
    strip_evidence,
    verify_claim,
)


def test_normalize_ws_collapses_indentation():
    assert normalize_ws("    const x = 1;  \n") == "const x = 1;"


def test_strip_evidence_removes_wrapping():
    assert strip_evidence('`const x = 1;`') == "const x = 1;"
    assert strip_evidence('"setCount(count + 1)"') == "setCount(count + 1)"
    assert strip_evidence("- code line") == "code line"
    assert strip_evidence('"setCount(count + 1);" |') == "setCount(count + 1);"


def test_parse_review_claims_reads_finding_lines_only():
    text = (
        "Let me look at this.\n"
        "FINDING: State updates in an empty-dependency effect | EVIDENCE: setCount(count + 1);\n"
        "some trailing prose\n"
        "FINDING: Stale closure in the handler | EVIDENCE: onClick={() => setCount(count + 1)}\n"
    )
    claims = parse_review_claims(text, Citation(file="a.tsx", start_line=1, end_line=10))
    assert [c.text for c in claims] == [
        "State updates in an empty-dependency effect",
        "Stale closure in the handler",
    ]
    assert claims[0].evidence == "setCount(count + 1);"


def test_parse_review_claims_returns_empty_for_none():
    assert parse_review_claims("NONE", Citation(file="a.tsx", start_line=1, end_line=1)) == []


def test_locate_evidence_finds_physical_line():
    lines = ["export function f() {\n", "  const x = 1;\n", "}\n"]
    assert locate_evidence("const x = 1;", lines, 40) == 41


def test_locate_evidence_matches_a_collapsed_multi_line_quote():
    lines = ["  useEffect(() => {\n", "    setCount(count + 1);\n", "  }, []);\n", "}\n"]
    evidence = "useEffect(() => { setCount(count + 1); }, []);"
    assert locate_evidence(evidence, lines, 6) == 6


def test_locate_evidence_rejects_text_that_is_not_there():
    lines = ["export function f() {\n", "  const x = 1;\n", "}\n"]
    assert locate_evidence("onSkip(suppress); onSkip(suppress);", lines, 1) is None


def test_verify_claim_accepts_grounded_evidence():
    lines = ["export function f() {\n", "  const x = 1;\n", "}\n"]
    claim = ReviewClaim(text="x is set once", evidence="const x = 1;", citation=Citation(file="a.ts", start_line=10, end_line=12))
    finding, dropped = verify_claim(claim, lines)
    assert dropped is None
    assert finding is not None
    assert finding.evidence_line == 11


def test_verify_claim_rejects_fabricated_evidence():
    lines = ["export function f() {\n", "  const x = 1;\n", "}\n"]
    claim = ReviewClaim(text="invented", evidence="onSkip(suppress); onSkip(suppress);", citation=Citation(file="a.ts", start_line=10, end_line=12))
    finding, dropped = verify_claim(claim, lines)
    assert finding is None
    assert dropped is not None and dropped.reason == REASON_EVIDENCE_NOT_FOUND


def test_verify_claim_rejects_short_and_empty_evidence():
    lines = ["export function f() {\n"]
    short = ReviewClaim(text="x", evidence="fn", citation=Citation(file="a.ts", start_line=1, end_line=1))
    _, dropped_short = verify_claim(short, lines)
    assert dropped_short is not None and dropped_short.reason == REASON_EVIDENCE_TOO_SHORT

    empty = ReviewClaim(text="x", evidence="   ", citation=Citation(file="a.ts", start_line=1, end_line=1))
    _, dropped_empty = verify_claim(empty, lines)
    assert dropped_empty is not None and dropped_empty.reason == REASON_EMPTY_EVIDENCE


def test_verify_claim_rejects_a_claim_naming_an_absent_symbol():
    lines = ["export function f() {\n", "  setCount(count + 1);\n", "}\n"]
    claim = ReviewClaim(
        text="onSkip is invoked twice",
        evidence="setCount(count + 1);",
        citation=Citation(file="a.ts", start_line=1, end_line=3),
    )
    finding, dropped = verify_claim(claim, lines)
    assert finding is None
    assert dropped is not None and dropped.reason == REASON_CLAIM_SYMBOL_NOT_IN_CHUNK


def test_verify_claim_accepts_a_claim_naming_a_present_symbol():
    lines = ["export function f() {\n", "  setCount(count + 1);\n", "}\n"]
    claim = ReviewClaim(
        text="setCount is called with a stale count",
        evidence="setCount(count + 1);",
        citation=Citation(file="a.ts", start_line=1, end_line=3),
    )
    finding, dropped = verify_claim(claim, lines)
    assert dropped is None and finding is not None


def test_read_citation_lines_reads_the_named_range(tmp_path):
    path = tmp_path / "a.ts"
    path.write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")
    lines = read_citation_lines(Citation(file=str(path), start_line=2, end_line=3))
    assert lines == ["two\n", "three\n"]


def test_read_citation_lines_rejects_invalid_range(tmp_path):
    path = tmp_path / "a.ts"
    path.write_text("one\n", encoding="utf-8")
    assert read_citation_lines(Citation(file=str(path), start_line=5, end_line=9)) is None


def test_format_review_prompt_carries_file_and_chunk():
    prompt = format_review_prompt("find bugs", "/tmp/a.ts", 4, 6, "4: const x = 1;")
    assert "/tmp/a.ts lines 4-6" in prompt
    assert "const x = 1;" in prompt


class _FakeReranker:
    """Stands in for LFMReranker: one verified chunk and no network."""

    raw_endpoint = "http://localhost:8034/v1"

    def __init__(self, item: RerankResultItem):
        self._item = item

    async def rerank(self, query, candidates, threshold=None, top_k=None, intent=None):  # noqa: ANN001
        return RerankResponse(
            query=query,
            threshold=0.5,
            results=[self._item],
            candidate_manifest=[],
            telemetry=Telemetry(),
        )


def _item_for(path, start_line, end_line):
    return RerankResultItem(
        candidate_id=f"{path}:{start_line}-{end_line}:f",
        citation=Citation(file=str(path), start_line=start_line, end_line=end_line, symbol="f"),
    )


def test_review_reports_only_grounded_findings(tmp_path, monkeypatch):
    path = tmp_path / "a.tsx"
    path.write_text("export function f() {\n  setCount(count + 1);\n}\n", encoding="utf-8")
    item = _item_for(path, 1, 3)

    async def fake_generate(completion_url, prompt, timeout, client):  # noqa: ANN001
        return (
            "FINDING: State update in an empty effect | EVIDENCE: setCount(count + 1);\n"
            "FINDING: onSkip runs twice | EVIDENCE: onSkip(suppress); onSkip(suppress);\n"
        )

    monkeypatch.setattr(review_mod, "_generate", fake_generate)
    report = review_mod.review_sync("review", [str(path)], reranker=_FakeReranker(item))

    assert report.claims_generated == 2
    assert report.claims_verified == 1
    assert report.claims_dropped == 1
    assert report.findings[0].evidence == "setCount(count + 1);"
    assert report.dropped[0].reason == REASON_EVIDENCE_NOT_FOUND
    assert report.findings[0].evidence_line == 2


def test_review_returns_nothing_when_the_model_fails(tmp_path, monkeypatch):
    path = tmp_path / "a.tsx"
    path.write_text("export function f() {}\n", encoding="utf-8")
    item = _item_for(path, 1, 1)

    async def failing_generate(completion_url, prompt, timeout, client):  # noqa: ANN001
        return None

    monkeypatch.setattr(review_mod, "_generate", failing_generate)
    report = review_mod.review_sync("review", [str(path)], reranker=_FakeReranker(item))

    assert report.findings == []
    assert report.model_failures == 1
    assert report.claims_generated == 0


def test_review_drops_a_claim_the_support_judge_rejects(tmp_path, monkeypatch):
    path = tmp_path / "a.tsx"
    path.write_text("export function f() {\n  setCount(count + 1);\n}\n", encoding="utf-8")
    item = _item_for(path, 1, 3)

    async def fake_generate(completion_url, prompt, timeout, client):  # noqa: ANN001
        return "FINDING: plausible but wrong | EVIDENCE: setCount(count + 1);\n"

    async def reject_judge(engine, claim, evidence, timeout, client, semaphore):  # noqa: ANN001
        return "scored", 0.05

    monkeypatch.setattr(review_mod, "_generate", fake_generate)
    monkeypatch.setattr(review_mod, "_judge_support", reject_judge)
    report = review_mod.review_sync("review", [str(path)], reranker=_FakeReranker(item))

    assert report.claims_verified == 0
    assert report.dropped[0].reason == review_mod.REASON_CLAIM_NOT_SUPPORTED


def test_review_drops_a_claim_when_the_support_judge_cannot_run(tmp_path, monkeypatch):
    path = tmp_path / "a.tsx"
    path.write_text("export function f() {\n  setCount(count + 1);\n}\n", encoding="utf-8")
    item = _item_for(path, 1, 3)

    async def fake_generate(completion_url, prompt, timeout, client):  # noqa: ANN001
        return "FINDING: unverified | EVIDENCE: setCount(count + 1);\n"

    async def failed_judge(engine, claim, evidence, timeout, client, semaphore):  # noqa: ANN001
        return "failed", None

    monkeypatch.setattr(review_mod, "_generate", fake_generate)
    monkeypatch.setattr(review_mod, "_judge_support", failed_judge)
    report = review_mod.review_sync("review", [str(path)], reranker=_FakeReranker(item))

    assert report.claims_verified == 0
    assert report.dropped[0].reason == review_mod.REASON_SUPPORT_UNVERIFIED


def test_review_skips_the_support_judge_when_disabled(tmp_path, monkeypatch):
    path = tmp_path / "a.tsx"
    path.write_text("export function f() {\n  setCount(count + 1);\n}\n", encoding="utf-8")
    item = _item_for(path, 1, 3)

    async def fake_generate(completion_url, prompt, timeout, client):  # noqa: ANN001
        return "FINDING: grounded only | EVIDENCE: setCount(count + 1);\n"

    async def boom_judge(engine, claim, evidence, timeout, client, semaphore):  # noqa: ANN001
        raise AssertionError("the judge must not run when support_threshold is None")

    monkeypatch.setattr(review_mod, "_generate", fake_generate)
    monkeypatch.setattr(review_mod, "_judge_support", boom_judge)
    report = review_mod.review_sync("review", [str(path)], reranker=_FakeReranker(item), support_threshold=None)

    assert report.claims_verified == 1
    assert report.semantic_gate is False