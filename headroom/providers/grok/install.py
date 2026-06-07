"""Grok install-time helpers."""

from __future__ import annotations

from .runtime import build_launch_env


def build_install_env(*, port: int, backend: str) -> dict[str, str]:
    """Build the persistent install environment for Grok Build."""
    # Accepted for the shared install-provider interface; Grok only needs the proxy URL.
    _ = backend
    env, _lines = build_launch_env(port=port, environ={})
    return {"GROK_CLI_CHAT_PROXY_BASE_URL": env["GROK_CLI_CHAT_PROXY_BASE_URL"]}
