"""Unit tests for the MCP server: candidate path resolution and the rerank_codebase tool."""

import asyncio
from types import SimpleNamespace

import pytest

import mcp_server
from mcp_server import resolve_candidate_paths


@pytest.fixture()
def sample_tree(tmp_path):
    """A small project tree with a nested package and an ignored dependency directory."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "nested").mkdir()
    (tmp_path / "node_modules").mkdir()

    (tmp_path / "src" / "pay.ts").write_text("export function charge() {}\n")
    (tmp_path / "src" / "nested" / "refund.ts").write_text("export function refund() {}\n")
    (tmp_path / "src" / "notes.md").write_text("# notes\n")
    (tmp_path / "node_modules" / "vendor.ts").write_text("export function vendor() {}\n")
    return tmp_path


def test_resolve_candidate_paths_direct_files(sample_tree):
    target = sample_tree / "src" / "pay.ts"
    resolved = resolve_candidate_paths([str(target)])

    assert resolved == [str(target.resolve())]


def test_resolve_candidate_paths_dedupes_and_skips_blanks(sample_tree):
    target = sample_tree / "src" / "pay.ts"
    resolved = resolve_candidate_paths([str(target), "  ", str(target)])

    assert resolved == [str(target.resolve())]


def test_resolve_candidate_paths_directory_walk_ignores_vendor_dirs(sample_tree):
    resolved = resolve_candidate_paths([str(sample_tree)])

    assert str((sample_tree / "src" / "pay.ts").resolve()) in resolved
    assert str((sample_tree / "src" / "nested" / "refund.ts").resolve()) in resolved
    assert str((sample_tree / "src" / "notes.md").resolve()) in resolved
    assert not any("node_modules" in path for path in resolved)


def test_resolve_candidate_paths_glob_patterns(sample_tree, monkeypatch):
    monkeypatch.chdir(sample_tree)
    resolved = resolve_candidate_paths(["src/**/*.ts"])

    assert str((sample_tree / "src" / "pay.ts").resolve()) in resolved
    assert str((sample_tree / "src" / "nested" / "refund.ts").resolve()) in resolved
    assert str((sample_tree / "src" / "notes.md").resolve()) not in resolved


def test_resolve_candidate_paths_glob_skips_ignored_dirs(sample_tree, monkeypatch):
    monkeypatch.chdir(sample_tree)
    resolved = resolve_candidate_paths(["**/*.ts"])

    assert resolved
    assert not any("node_modules" in path for path in resolved)


def test_resolve_candidate_paths_respects_max_files(sample_tree):
    resolved = resolve_candidate_paths([str(sample_tree)], max_files=2)

    assert len(resolved) == 2


def test_resolve_candidate_paths_no_matches_returns_empty_list(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert resolve_candidate_paths(["does/not/exist/**/*.ts"]) == []
    assert resolve_candidate_paths([]) == []


def _fake_response():
    """Minimal stand-in for slm_rerank.models.RerankResponse."""
    item = SimpleNamespace(
        score=0.9123,
        raw_score=0.8811,
        file_path="src/pay.ts",
        symbol="charge",
        snippet="export function charge() {}",
        citation=SimpleNamespace(file="src/pay.ts", start_line=1, end_line=4),
        ground_truth_status=SimpleNamespace(value="VERIFIED"),
    )
    telemetry = SimpleNamespace(
        model_id="lfm",
        candidate_files_count=1,
        chunks_evaluated=1,
        total_wall_time_s=0.12345,
        total_prompt_tokens=420,
        top_k_input_tokens=64,
        reduction_ratio=6.5625,
        reduction_percentage=84.76,
    )
    return SimpleNamespace(results=[item], telemetry=telemetry)


def test_rerank_codebase_returns_error_when_no_candidates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = asyncio.run(
        mcp_server.rerank_codebase(query="charge handler", paths_or_globs=["missing/**/*.ts"])
    )

    assert result["error"] == "No matching files found for candidate patterns."
    assert result["results"] == []
    assert result["query"] == "charge handler"


def test_rerank_codebase_maps_results_and_telemetry(sample_tree, monkeypatch):
    captured = {}

    class FakeReranker:
        def __init__(self, model, base_url):
            captured["model"] = model
            captured["base_url"] = base_url

        async def rerank(self, query, candidates, threshold, top_k, intent):
            captured["query"] = query
            captured["candidates"] = candidates
            captured["threshold"] = threshold
            captured["top_k"] = top_k
            captured["intent"] = intent
            return _fake_response()

    monkeypatch.setattr(mcp_server, "LFMReranker", FakeReranker)

    result = asyncio.run(
        mcp_server.rerank_codebase(
            query="charge handler",
            paths_or_globs=[str(sample_tree / "src" / "pay.ts")],
            threshold=0.7,
            top_k=3,
            intent="SPECIFICATION",
        )
    )

    assert captured["model"] == "lfm"
    assert captured["candidates"] == [str((sample_tree / "src" / "pay.ts").resolve())]
    assert captured["threshold"] == 0.7
    assert captured["top_k"] == 3
    assert captured["intent"] == mcp_server.QueryIntent.SPECIFICATION

    assert "error" not in result
    assert result["model"] == "lfm"
    assert result["files_evaluated"] == 1
    assert result["chunks_evaluated"] == 1
    assert result["wall_time_seconds"] == 0.123
    assert result["token_reduction_ratio"] == "6.6x"
    assert result["token_reduction_percentage"] == "84.8%"
    assert result["top_results_count"] == 1

    top = result["results"][0]
    assert top["score"] == 0.912
    assert top["raw_score"] == 0.881
    assert top["file_path"] == "src/pay.ts"
    assert top["symbol"] == "charge"
    assert top["start_line"] == 1
    assert top["end_line"] == 4
    assert top["line_count"] == 4
    assert top["ground_truth"] == "VERIFIED"


def test_rerank_codebase_falls_back_to_implementation_on_bad_intent(sample_tree, monkeypatch):
    captured = {}

    class FakeReranker:
        def __init__(self, model, base_url):
            pass

        async def rerank(self, query, candidates, threshold, top_k, intent):
            captured["intent"] = intent
            return _fake_response()

    monkeypatch.setattr(mcp_server, "LFMReranker", FakeReranker)

    asyncio.run(
        mcp_server.rerank_codebase(
            query="charge handler",
            paths_or_globs=[str(sample_tree / "src" / "pay.ts")],
            intent="NOT_A_REAL_INTENT",
        )
    )

    assert captured["intent"] == mcp_server.QueryIntent.IMPLEMENTATION
