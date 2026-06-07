"""Optional live Hermes + Headroom integration tests (OpenAI chat path).

Grok Build's ``wrap grok`` routes ``/v1/sessions/*`` traffic — mostly passthrough.
Hermes on spot-tech-ci exposes ``/v1/chat/completions``, which is the compressible
path Headroom optimizes.

Run on a host where Hermes llm-proxy is up (default ``:38765``):

    HEADROOM_LIVE_HERMES=1 UV_SKIP_WHEEL_FILENAME_CHECK=1 \\
        uv run --extra dev pytest tests/test_cli/test_hermes_grok_live.py -v

spot-tech-ci runbook (2026-06-07, ``falk@80.241.217.210``):

    curl -s http://127.0.0.1:38765/health          # => ok
    export PATH="$HOME/.local/bin:$PATH"
    uv tool install 'headroom-ai[proxy]'          # once, if headroom missing
    HEADROOM_LIVE_HERMES=1 uv run --extra dev \\
        pytest tests/test_cli/test_hermes_grok_live.py -v
    python scripts/bench_hermes_headroom.py

Captured benchmark (fat single-turn prompt, grok backend):

| Path | ``prompt_tokens`` | Headroom ``tokens.saved`` |
|------|-------------------|---------------------------|
| Plain Hermes | 3357 | — |
| Headroom → Hermes | 3357 | 0 |

Routing works end-to-end; token deltas depend on workload shape (tool dumps,
multi-turn agent loops). See ``scripts/bench_hermes_headroom.py --multi-turn``.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

_LIVE = os.environ.get("HEADROOM_LIVE_HERMES") == "1"
HERMES_BASE = os.environ.get("HEADROOM_HERMES_BASE_URL", "http://127.0.0.1:38765/v1").rstrip("/")
HERMES_MODEL = os.environ.get("HEADROOM_HERMES_MODEL", "grok-4.3")
REPLY_TOKEN = os.environ.get("HEADROOM_HERMES_REPLY_TOKEN", "HERMES_OK")
HERMES_HEALTH_URL = HERMES_BASE.replace("/v1", "") + "/health"

pytestmark = pytest.mark.skipif(not _LIVE, reason="set HEADROOM_LIVE_HERMES=1 to run live Hermes tests")


def _hermes_reachable() -> bool:
    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(HERMES_HEALTH_URL)
            return resp.status_code == 200
    except Exception:
        return False


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _headroom_argv(repo_root: Path) -> list[str]:
    if shutil.which("headroom"):
        return ["headroom"]
    return ["uv", "run", "headroom"]


def _wait_proxy_ready(port: int, timeout: float = 180.0) -> None:
    deadline = time.time() + timeout
    with httpx.Client(timeout=3.0) as client:
        while time.time() < deadline:
            for path in ("/readyz", "/health", "/livez"):
                try:
                    resp = client.get(f"http://127.0.0.1:{port}{path}")
                    if resp.status_code == 200:
                        return
                except Exception:
                    continue
            time.sleep(1.0)
    raise RuntimeError(f"headroom proxy on {port} did not become ready")


def _chat(client: httpx.Client, base_url: str, content: str) -> dict[str, Any]:
    payload = {
        "model": HERMES_MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 16,
        "temperature": 0,
        "stream": False,
    }
    resp = client.post(f"{base_url}/chat/completions", json=payload, timeout=180.0)
    resp.raise_for_status()
    body = resp.json()
    body["_backend"] = resp.headers.get("x-llm-backend")
    return body


@pytest.fixture(scope="module")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def hermes_client() -> Iterator[httpx.Client]:
    if not _hermes_reachable():
        pytest.skip(f"Hermes not reachable at {HERMES_HEALTH_URL}")
    with httpx.Client() as client:
        yield client


@pytest.fixture
def headroom_proxy(repo_root: Path) -> Iterator[int]:
    port = _pick_free_port()
    env = os.environ.copy()
    env["HEADROOM_TELEMETRY"] = "off"
    proc = subprocess.Popen(
        [
            *_headroom_argv(repo_root),
            "proxy",
            "--port",
            str(port),
            "--openai-api-url",
            HERMES_BASE,
            "--no-telemetry",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=repo_root,
        env=env,
    )
    try:
        _wait_proxy_ready(port)
        yield port
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.mark.skipif(not _hermes_reachable(), reason="Hermes llm-proxy not reachable")
def test_live_hermes_health(hermes_client: httpx.Client) -> None:
    resp = hermes_client.get(HERMES_HEALTH_URL)
    assert resp.status_code == 200
    assert resp.text.strip().lower() in {"ok", '"ok"'}


@pytest.mark.skipif(not _hermes_reachable(), reason="Hermes llm-proxy not reachable")
def test_live_plain_hermes_chat_completion(hermes_client: httpx.Client) -> None:
    body = _chat(
        hermes_client,
        HERMES_BASE,
        f"Reply with exactly one word: {REPLY_TOKEN}",
    )
    content = body["choices"][0]["message"]["content"]
    usage = body.get("usage") or {}
    assert REPLY_TOKEN in content
    assert int(usage.get("prompt_tokens") or 0) > 0


@pytest.mark.skipif(not _hermes_reachable(), reason="Hermes llm-proxy not reachable")
def test_live_headroom_proxy_routes_to_hermes(
    hermes_client: httpx.Client,
    headroom_proxy: int,
) -> None:
    body = _chat(
        hermes_client,
        f"http://127.0.0.1:{headroom_proxy}/v1",
        f"Reply with exactly one word: {REPLY_TOKEN}",
    )
    content = body["choices"][0]["message"]["content"]
    assert REPLY_TOKEN in content

    stats_resp = hermes_client.get(f"http://127.0.0.1:{headroom_proxy}/stats", timeout=30.0)
    stats_resp.raise_for_status()
    stats = stats_resp.json()
    requests_total = int((stats.get("requests") or {}).get("total") or 0)
    assert requests_total >= 1


@pytest.mark.skipif(not _hermes_reachable(), reason="Hermes llm-proxy not reachable")
def test_live_headroom_openai_upstream_points_at_hermes(
    hermes_client: httpx.Client,
    headroom_proxy: int,
) -> None:
    health_resp = hermes_client.get(f"http://127.0.0.1:{headroom_proxy}/health", timeout=30.0)
    health_resp.raise_for_status()
    health = health_resp.json()
    config = health.get("config") if isinstance(health.get("config"), dict) else health
    assert config.get("openai_api_url") == HERMES_BASE