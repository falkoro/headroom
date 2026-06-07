#!/usr/bin/env python3
"""Compare plain Hermes vs Headroom→Hermes token usage on /v1/chat/completions.

Uses agent-shaped workloads (large ``role: tool`` outputs) — the shape
Headroom SmartCrusher actually compresses. Plain user/system log filler does
not trigger savings.

Examples::

    python scripts/bench_hermes_headroom.py
    python scripts/bench_hermes_headroom.py --multi-turn
    HEADROOM_HERMES_BASE_URL=http://127.0.0.1:38765/v1 python scripts/bench_hermes_headroom.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Shared workloads live under tests/test_cli for pytest parity.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "test_cli"))
from hermes_workloads import agent_tool_messages, multi_turn_agent_messages  # noqa: E402

DEFAULT_HERMES_BASE = os.environ.get("HEADROOM_HERMES_BASE_URL", "http://127.0.0.1:38765/v1")
DEFAULT_MODEL = os.environ.get("HEADROOM_HERMES_MODEL", "grok-4.3")


def pick_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def hermes_health_ok(base_v1: str) -> bool:
    root = base_v1.rstrip("/").removesuffix("/v1")
    try:
        with urllib.request.urlopen(f"{root}/health", timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def chat(base_url: str, messages: list[dict], model: str, max_tokens: int = 24) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read().decode())
    body["_backend"] = resp.headers.get("X-LLM-Backend", "?")
    return body


def wait_proxy_ready(port: int, timeout: float = 180.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        for path in ("/readyz", "/health", "/livez"):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=3) as resp:
                    if resp.status == 200:
                        return
            except Exception:
                continue
        time.sleep(1.0)
    raise RuntimeError(f"headroom proxy on {port} did not become ready")


def fetch_stats(port: int) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/stats", timeout=30) as resp:
        return json.loads(resp.read().decode())


def headroom_argv() -> list[str]:
    if shutil.which("headroom"):
        return ["headroom"]
    return ["uv", "run", "headroom"]


def _delta_block(plain_usage: dict, wrapped_usage: dict, token_stats: dict, stats: dict) -> dict:
    plain_in = int(plain_usage.get("prompt_tokens") or 0)
    wrap_in = int(wrapped_usage.get("prompt_tokens") or 0)
    saved = int(token_stats.get("saved") or 0)
    return {
        "plain_prompt_tokens": plain_in,
        "headroom_prompt_tokens": wrap_in,
        "upstream_prompt_token_delta": plain_in - wrap_in,
        "headroom_tokens_saved": saved,
        "tokens_saved_by_strategy": stats.get("tokens_saved_by_strategy") or {},
        "compression_summary": stats.get("summary", {}).get("compression") or {},
    }


def run_agent_tool_turn(hermes_base: str, model: str) -> dict:
    messages = agent_tool_messages()
    plain = chat(hermes_base, messages, model)
    plain_usage = plain.get("usage") or {}

    port = pick_port()
    env = {**dict(os.environ), "HEADROOM_TELEMETRY": "off"}
    proc = subprocess.Popen(
        [
            *headroom_argv(),
            "proxy",
            "--port",
            str(port),
            "--openai-api-url",
            hermes_base,
            "--no-telemetry",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    try:
        wait_proxy_ready(port)
        wrapped = chat(f"http://127.0.0.1:{port}/v1", messages, model)
        wrapped_usage = wrapped.get("usage") or {}
        stats = fetch_stats(port)
        token_stats = stats.get("tokens") or {}
        delta = _delta_block(plain_usage, wrapped_usage, token_stats, stats)
        return {
            "mode": "agent-tool",
            "plain": {
                "content": plain["choices"][0]["message"]["content"].strip(),
                "usage": plain_usage,
                "backend": plain.get("_backend"),
            },
            "headroom": {
                "content": wrapped["choices"][0]["message"]["content"].strip(),
                "usage": wrapped_usage,
                "backend": wrapped.get("_backend"),
                "tokens_saved": token_stats.get("saved"),
                "proxy_compression_saved": token_stats.get("proxy_compression_saved"),
            },
            "delta": delta,
        }
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


def run_multi_turn(hermes_base: str, model: str, turns: int = 3) -> dict:
    messages = multi_turn_agent_messages(turns=turns)

    def final_turn(base_url: str) -> dict:
        body = chat(base_url, messages, model, max_tokens=16)
        return {
            "content": body["choices"][0]["message"]["content"].strip(),
            "usage": body.get("usage") or {},
            "backend": body.get("_backend"),
        }

    plain = final_turn(hermes_base)
    port = pick_port()
    env = {**dict(os.environ), "HEADROOM_TELEMETRY": "off"}
    proc = subprocess.Popen(
        [
            *headroom_argv(),
            "proxy",
            "--port",
            str(port),
            "--openai-api-url",
            hermes_base,
            "--no-telemetry",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    try:
        wait_proxy_ready(port)
        wrapped = final_turn(f"http://127.0.0.1:{port}/v1")
        stats = fetch_stats(port)
        token_stats = stats.get("tokens") or {}
        delta = _delta_block(plain["usage"], wrapped["usage"], token_stats, stats)
        return {
            "mode": "multi-turn-agent-tool",
            "turns": turns,
            "plain": plain,
            "headroom": {
                **wrapped,
                "tokens_saved": token_stats.get("saved"),
            },
            "delta": delta,
        }
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hermes-base",
        default=DEFAULT_HERMES_BASE,
        help="Hermes OpenAI base URL (default: %(default)s)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Chat model id")
    parser.add_argument(
        "--multi-turn",
        action="store_true",
        help="Run a multi-turn agent loop with tool outputs per turn",
    )
    args = parser.parse_args()
    hermes_base = args.hermes_base.rstrip("/")

    if not hermes_health_ok(hermes_base):
        print(f"Hermes health check failed for {hermes_base}", file=sys.stderr)
        return 1

    try:
        result = (
            run_multi_turn(hermes_base, args.model)
            if args.multi_turn
            else run_agent_tool_turn(hermes_base, args.model)
        )
    except urllib.error.URLError as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    delta = result.get("delta") or {}
    if int(delta.get("headroom_tokens_saved") or 0) <= 0 and int(
        delta.get("upstream_prompt_token_delta") or 0
    ) <= 0:
        print(
            "WARN: no token savings detected — ensure workload uses large role:tool outputs",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())