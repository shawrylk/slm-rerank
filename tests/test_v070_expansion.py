"""v0.7.0: optional SLM query expansion.

Deterministic recall stems words; it cannot reach "bridge" from "harness". Expansion
asks the local model for that, and is strictly additive: every failure path returns no
terms so recall degrades to the deterministic list rather than breaking.

Mirrors the expander tests in src/index.test.mjs.
"""
import asyncio
import subprocess

import httpx
import pytest

from slm_rerank.discovery import discover_candidate_files
from slm_rerank.expander import expand_query, format_expansion_prompt, parse_expansion_text


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_prompt_carries_query_and_reasoning_bypass():
    prompt = format_expansion_prompt("interop harness")
    assert "Query: interop harness" in prompt
    assert prompt.endswith("<|im_start|>assistant\n<think>\n</think>\n")


def test_parsing_drops_prose_generic_nouns_and_query_terms():
    terms = parse_expansion_text(
        "communication, protocol, the, code, function, interop, bridge, adapter, bridge",
        "interop harness",
    )
    assert "bridge" in terms
    assert "adapter" in terms
    assert "the" not in terms, "stop word leaked through"
    assert "code" not in terms, "generic noun leaked through"
    assert "function" not in terms, "generic noun leaked through"
    assert "interop" not in terms, "term already in the query was re-added"
    assert terms.count("bridge") == 1

    assert parse_expansion_text("", "q") == []
    assert parse_expansion_text(None, "q") == []


def test_reads_llama_cpp_content_and_caps_term_count():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = __import__("json").loads(request.content)
        return httpx.Response(200, json={"content": "bridge, adapter, wrapper, connector, marshal"})

    async def main():
        async with _client(handler) as client:
            return await expand_query(
                "interop harness",
                base_url="http://127.0.0.1:8034/v1",
                use_cache=False,
                max_terms=3,
                client=client,
            )

    assert asyncio.run(main()) == ["bridge", "adapter", "wrapper"]
    assert seen["url"] == "http://127.0.0.1:8034/completion", "must use the native endpoint"
    assert seen["body"]["temperature"] == 0, "greedy decoding keeps the cache meaningful"
    assert seen["body"]["n_predict"] == 64


@pytest.mark.parametrize(
    "handler",
    [
        lambda request: httpx.Response(500, json={}),
        lambda request: httpx.Response(200, json={}),
        lambda request: httpx.Response(200, text="not json"),
    ],
)
def test_failures_degrade_to_no_terms(handler):
    async def main():
        async with _client(handler) as client:
            return await expand_query(
                "interop harness", base_url="http://x/v1", use_cache=False, client=client
            )

    assert asyncio.run(main()) == []


def test_transport_error_degrades_to_no_terms():
    def handler(request):
        raise httpx.ConnectError("unreachable")

    async def main():
        async with _client(handler) as client:
            return await expand_query("q", base_url="http://x/v1", use_cache=False, client=client)

    assert asyncio.run(main()) == []


def test_missing_inputs_short_circuit():
    assert asyncio.run(expand_query("", base_url="http://x/v1", use_cache=False)) == []
    assert asyncio.run(expand_query("q", base_url="", use_cache=False)) == []


def test_expanded_term_widens_recall_without_outranking_a_literal_hit(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "harness.ts").write_text("export const x = 1;\n")
    (tmp_path / "src" / "bridge.ts").write_text("export const y = 2;\n")
    subprocess.run(["git", "init", "-q", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=test", "commit", "-qm", "f"],
        cwd=tmp_path, check=True,
    )

    base = discover_candidate_files("harness", cwd=str(tmp_path))
    assert "src/bridge.ts" not in base

    expanded = discover_candidate_files("harness", cwd=str(tmp_path), extra_terms=["bridge"])
    assert "src/bridge.ts" in expanded, "expansion did not widen recall"
    assert expanded[0] == "src/harness.ts", "a synonym outranked the literal match"
