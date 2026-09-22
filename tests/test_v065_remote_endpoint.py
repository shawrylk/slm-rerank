"""v0.6.5: remote endpoint configuration and honest discovery failure.

Discovery only ever scans ports on a single host -- never the network -- so a
model server running on another machine is reachable only via an explicit
endpoint or host. These tests pin the env-var contract, which is shared
verbatim with the Node implementation in src/discovery.mjs.
"""
import asyncio

import mcp_server
from slm_rerank.config import resolve_endpoint_and_model
from slm_rerank.discovery import (
    DEFAULT_HOST,
    auto_discover_endpoint,
    resolve_endpoint_env,
    resolve_host_env,
)

ENDPOINT_VARS = ("SLM_ENDPOINT", "RERANKER_BASE_URL", "LFM_ENDPOINT")
HOST_VARS = ("SLM_HOST", "RERANKER_HOST")


def _clear_env(monkeypatch):
    for var in ENDPOINT_VARS + HOST_VARS:
        monkeypatch.delenv(var, raising=False)


def test_resolve_host_env_precedence():
    assert resolve_host_env({}) == DEFAULT_HOST
    assert resolve_host_env({"RERANKER_HOST": "10.2.99.1"}) == "10.2.99.1"
    assert resolve_host_env({"SLM_HOST": "a", "RERANKER_HOST": "b"}) == "a"
    assert resolve_host_env({"SLM_HOST": "   "}) == DEFAULT_HOST


def test_resolve_endpoint_env_precedence():
    assert resolve_endpoint_env({}) is None
    assert resolve_endpoint_env({"LFM_ENDPOINT": "http://x:8034/v1"}) == "http://x:8034/v1"
    assert (
        resolve_endpoint_env({"RERANKER_BASE_URL": "http://b/v1", "LFM_ENDPOINT": "http://c/v1"})
        == "http://b/v1"
    )
    assert (
        resolve_endpoint_env({"SLM_ENDPOINT": " http://a/v1 ", "RERANKER_BASE_URL": "http://b/v1"})
        == "http://a/v1"
    )


def test_pinned_endpoint_wins_over_requested_model():
    """The caller named the server, so discovery must not fall back to scanning loopback."""
    ep = asyncio.run(
        auto_discover_endpoint(
            requested_model="qwen",
            ports=[8099],
            env={"SLM_ENDPOINT": "http://192.168.1.220:8034/v1"},
        )
    )
    assert ep["url"] == "http://192.168.1.220:8034/v1"
    assert ep["ok"] is True
    assert ep["model_id"] == "pinned-via-env"


def test_host_env_selects_the_scan_target():
    ep = asyncio.run(
        auto_discover_endpoint(ports=[8099], timeout=0.05, env={"SLM_HOST": "192.0.2.1"})
    )
    assert ep["host"] == "192.0.2.1"
    assert ep["ok"] is False


def test_missing_server_reports_a_reason_not_a_phantom_url():
    ep = asyncio.run(
        auto_discover_endpoint(host="127.0.0.1", ports=[8099], timeout=0.05, env={})
    )
    assert ep["ok"] is False
    assert ep["url"] is None
    assert ep["port"] is None
    assert "SLM_ENDPOINT" in ep["reason"]
    assert "8099" in ep["reason"]


def test_config_honours_slm_endpoint(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("SLM_ENDPOINT", "http://192.168.1.220:8034/v1")
    _, base_url = resolve_endpoint_and_model(config={})
    assert base_url == "http://192.168.1.220:8034/v1"


def test_config_builds_endpoint_from_host_alone(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("SLM_HOST", "192.168.1.220")
    _, base_url = resolve_endpoint_and_model(config={})
    assert base_url == "http://192.168.1.220:8034/v1"


def test_config_leaves_loopback_unset(monkeypatch):
    """Loopback is the caller's own default; config must not invent an endpoint."""
    _clear_env(monkeypatch)
    _, base_url = resolve_endpoint_and_model(config={})
    assert base_url is None


def test_mcp_server_default_base_url_follows_env(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("RERANKER_BASE_URL", "http://10.2.99.1:8034/v1")
    assert mcp_server._default_base_url() == "http://10.2.99.1:8034/v1"

    _clear_env(monkeypatch)
    monkeypatch.setenv("SLM_HOST", "10.2.99.1")
    assert mcp_server._default_base_url() == "http://10.2.99.1:8034/v1"

    _clear_env(monkeypatch)
    assert mcp_server._default_base_url() == f"http://{DEFAULT_HOST}:8034/v1"
