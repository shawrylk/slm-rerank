"""Unit tests for v0.6.0 features: multi-port discovery, ghost stubs, boundaries, and ripgrep discovery."""

import pytest
from slm_rerank.boundary import detect_slice, group_by_slice
from slm_rerank.discovery import auto_discover_endpoint, discover_candidate_files, probe_port_sync
from slm_rerank.models import Citation, RerankResultItem
from slm_rerank.stubber import generate_ghost_stub


def test_detect_slice_patterns():
    assert detect_slice("features/billing/charge.ts") == "billing"
    assert detect_slice("modules/auth/tokens.py") == "auth"
    assert detect_slice("packages/qc-harness/index.mjs") == "qc-harness"
    assert detect_slice("workers/rasterizer/worker.ts") == "rasterizer"
    assert detect_slice("src/payment/service.ts") == "payment"


def test_group_by_slice():
    items = [
        RerankResultItem(
            candidate_id="c1",
            file_path="features/billing/charge.ts",
            score=0.95,
            citation=Citation(file="features/billing/charge.ts", start_line=10, end_line=20),
        ),
        RerankResultItem(
            candidate_id="c2",
            file_path="features/auth/login.ts",
            score=0.82,
            citation=Citation(file="features/auth/login.ts", start_line=5, end_line=15),
        ),
        RerankResultItem(
            candidate_id="c3",
            file_path="features/billing/invoice.ts",
            score=0.70,
            citation=Citation(file="features/billing/invoice.ts", start_line=1, end_line=30),
        ),
    ]

    grouped = group_by_slice(items)
    assert "billing" in grouped
    assert "auth" in grouped
    assert len(grouped["billing"]["items"]) == 2
    assert grouped["billing"]["max_score"] == 0.95
    assert len(grouped["auth"]["items"]) == 1


def test_generate_ghost_stub_python():
    code = """import os
from typing import List

def helper_one(x: int) -> int:
    y = x + 1
    z = y * 2
    return z

def target_calculation(a: int, b: int) -> int:
    return a + b

def helper_two():
    pass
"""
    class DummyChunk:
        start_line = 9
        end_line = 11

    res = generate_ghost_stub("dummy.py", DummyChunk(), content=code)
    stub = res["stub"]
    assert "import os" in stub
    assert "target_calculation" in stub
    assert "folded" in stub
    assert res["folded_lines"] > 0


def test_discover_candidate_files_with_ripgrep():
    files = discover_candidate_files("chunk and extract code symbols")
    assert isinstance(files, list)
    assert len(files) > 0
    assert any("chunker" in f for f in files)


def test_auto_discover_endpoint():
    """No server answering must report failure, not a phantom :8034 endpoint."""
    import asyncio
    ep = asyncio.run(auto_discover_endpoint(host="127.0.0.1", ports=[8099], env={}))
    assert ep["ok"] is False
    assert ep["url"] is None
    assert ep["port"] is None
    assert "SLM_ENDPOINT" in ep["reason"]
