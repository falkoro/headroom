#!/usr/bin/env python3
"""Compare plain Hermes vs Headroom→Hermes token usage on /v1/chat/completions.

Designed for spot-tech-ci (Hermes llm-proxy on :38765) but works on any host
where Hermes is reachable.

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

DEFAULT_HERMES_BASE = os.environ.get("HEADROOM_HERMES_BASE_URL", "http://127.0.0.1:38765/v1")
DEFAULT_MODEL = os.environ.get("HEADROOM_HERMES_MODEL", "grok-4.3")
REPLY_TOKEN = "BENCH_OK"


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


def fat_system_prompt(repeats: int = 80) -> str:
    chunk = (
        "TOOL_OUTPUT chunk=42 status=ok lines=500 "
        "payload=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa "
        "stacktrace=Error at module.ts:128 in handleRequest "
        "notes=repeatable compressible log filler for headroom bench "
    )
    return (
        "You are a benchmark assistant. Ignore the filler below except to answer briefly.\n\n"
        + (chunk + "\n") * repeats
    )


def log_block(turn: int) -> str:
    return (
        f"TOOL_OUTPUT turn={turn} status=ok lines=200 "
        f"payload={'x' * 120} "
        f"trace=Error at svc.ts:{100 + turn} in handleBatch "
        f"notes=compressible repeated agent log block {turn}\n"
    )


def chat(base_url: str, messages: list[dict], model: str, max_tokens: int = 16) -> dict:
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


def run_single_turn(hermes_base: str, model: str) -> dict:
    messages = [
        {"role": "system", "content": fat_system_prompt()},
        {"role": "user", "content": f"Reply with exactly one word: {REPLY_TOKEN}"},
    ]
    plain = chat(hermes_base, messages, model)
    plain_usage = plain.get("usage") or {}

    port = pick_port()
    env = {
        **dict(os.environ),
        "HEADROOM_TELEMETRY": "off",
    }
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
        return {
            "mode": "single-turn",
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
            "delta": {
                "plain_prompt_tokens": int(plain_usage.get("prompt_tokens") or 0),
                "headroom_prompt_tokens": int(wrapped_usage.get("prompt_tokens") or 0),
                "headroom_tokens_saved": int(token_stats.get("saved") or 0),
            },
        }
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


def run_multi_turn(hermes_base: str, model: str, turns: int = 4, log_blocks: int = 60) -> dict:
    def conversation(base_url: str) -> dict:
        messages: list[dict] = [
            {
                "role": "system",
                "content": "Short answers only. You are running a multi-turn benchmark.",
            }
        ]
        usages: list[dict] = []
        for turn in range(1, turns + 1):
            messages.append(
                {
                    "role": "user",
                    "content": log_block(turn) * log_blocks + f"\nTurn {turn}: reply TURN_{turn}",
                }
            )
            body = chat(base_url, messages, model, max_tokens=12)
            reply = body["choices"][0]["message"]["content"].strip()
            messages.append({"role": "assistant", "content": reply})
            usages.append(body.get("usage") or {})
        return {
            "turn_usages": usages,
            "final_prompt_tokens": int(usages[-1].get("prompt_tokens") or 0),
            "sum_prompt_tokens": sum(int(u.get("prompt_tokens") or 0) for u in usages),
        }

    plain = conversation(hermes_base)
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
        wrapped = conversation(f"http://127.0.0.1:{port}/v1")
        stats = fetch_stats(port)
        token_stats = stats.get("tokens") or {}
        wrapped["headroom_tokens_saved"] = token_stats.get("saved")
        return {
            "mode": "multi-turn",
            "plain": plain,
            "headroom": wrapped,
            "delta": {
                "plain_final_prompt_tokens": plain["final_prompt_tokens"],
                "headroom_final_prompt_tokens": wrapped["final_prompt_tokens"],
                "plain_sum_prompt_tokens": plain["sum_prompt_tokens"],
                "headroom_sum_prompt_tokens": wrapped["sum_prompt_tokens"],
                "headroom_tokens_saved": int(token_stats.get("saved") or 0),
            },
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
        help="Run a 4-turn agent-style loop instead of a single fat prompt",
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
            else run_single_turn(hermes_base, args.model)
        )
    except urllib.error.URLError as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())