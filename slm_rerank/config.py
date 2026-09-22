"""Configuration management for lfm-rerank (~/.config/reranker/config.yaml)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from .discovery import DEFAULT_HOST, resolve_endpoint_env, resolve_host_env

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def get_default_config_path() -> Path:
    """Return default configuration path according to XDG base directory specification."""
    base_dir = os.environ.get("XDG_CONFIG_HOME")
    if base_dir:
        config_dir = Path(base_dir) / "reranker"
    else:
        config_dir = Path.home() / ".config" / "reranker"
    return config_dir / "config.yaml"


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load and parse YAML config file if present, returning empty dict otherwise."""
    path = config_path or get_default_config_path()
    if not path.exists():
        return {}

    try:
        content = path.read_text(encoding="utf-8")
        if yaml is not None:
            data = yaml.safe_load(content)
            return data if isinstance(data, dict) else {}
        else:  # pragma: no cover
            # Minimal key-value parser fallback if PyYAML is somehow unavailable
            result: Dict[str, Any] = {}
            for line in content.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    k, v = line.split(":", 1)
                    result[k.strip()] = v.strip().strip("'\"")
            return result
    except Exception:
        return {}


def save_config(config: Dict[str, Any], config_path: Optional[Path] = None) -> Path:
    """Save configuration dictionary to YAML file."""
    path = config_path or get_default_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is not None:
        dumped = yaml.safe_dump(config, sort_keys=False)
    else:  # pragma: no cover
        dumped = "\n".join(f"{k}: {v}" for k, v in config.items()) + "\n"
    path.write_text(dumped, encoding="utf-8")
    return path


def resolve_endpoint_and_model(
    cli_model: Optional[str] = None,
    cli_base_url: Optional[str] = None,
    cli_endpoint: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve model and base URL using CLI args, env vars, and config file."""
    cfg = config if config is not None else load_config()

    # Model resolution: CLI -> ENV -> config
    resolved_model = cli_model or os.environ.get("RERANKER_MODEL") or cfg.get("model") or cfg.get("default_model")

    # Explicit base URL from CLI or env
    explicit_base_url = (
        cli_base_url
        or cli_endpoint
        or resolve_endpoint_env()  # SLM_ENDPOINT -> RERANKER_BASE_URL -> LFM_ENDPOINT
    )
    if explicit_base_url:
        return resolved_model, explicit_base_url

    # Check model-specific endpoint in config if model is resolved
    if resolved_model:
        endpoints = cfg.get("endpoints", {})
        if isinstance(endpoints, dict):
            m_low = resolved_model.lower()
            if m_low in endpoints:
                return resolved_model, endpoints[m_low]

    # Fallback to general base_url from config
    resolved_base_url = cfg.get("base_url")

    # A host on its own is enough to reach a remote server, so build the default
    # :8034 endpoint from it rather than letting the caller fall back to loopback.
    if not resolved_base_url:
        host = resolve_host_env()
        if host != DEFAULT_HOST:
            resolved_base_url = f"http://{host}:8034/v1"

    return resolved_model, resolved_base_url
