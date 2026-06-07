"""Tests for `headroom wrap grok` command.

Grok Build routes cli-chat-proxy traffic via ``GROK_CLI_CHAT_PROXY_BASE_URL``.
These tests mirror real ``grok`` CLI invocations (from ``grok --help``) and
verify Headroom only injects the proxy env — Grok's own flags pass through
unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from headroom.cli.main import main
from headroom.providers.grok.runtime import GROK_PROXY_ENV


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# Real Grok CLI shapes lifted from `grok --help` (2026-06).
REAL_GROK_INVOCATIONS: list[tuple[list[str], tuple[str, ...]]] = [
    # TUI — no args, proxy env only.
    ([], ()),
    # Headless single-turn (`-p` / `--single`).
    (
        ["-p", "fix the failing test in test_wrap_grok.py"],
        ("-p", "fix the failing test in test_wrap_grok.py"),
    ),
    (
        ["--single", "summarize the diff and suggest a commit message"],
        ("--single", "summarize the diff and suggest a commit message"),
    ),
    # Continue the cwd's most recent session.
    (["--continue"], ("--continue",)),
    (["-c"], ("-c",)),
    # Scoped working directory + agent override.
    (
        ["--cwd", "/tmp/myproject", "--agent", "reviewer"],
        ("--cwd", "/tmp/myproject", "--agent", "reviewer"),
    ),
    # Headless parallel best-of-n with auto-approve (common CI-style invocation).
    (
        ["-p", "add input validation", "--best-of-n", "3", "--always-approve"],
        ("-p", "add input validation", "--best-of-n", "3", "--always-approve"),
    ),
    # Prompt from file + structured output (headless).
    (
        [
            "--prompt-file",
            "tasks/refactor.md",
            "--output-format",
            "json",
            "--permission-mode",
            "bypassPermissions",
        ],
        (
            "--prompt-file",
            "tasks/refactor.md",
            "--output-format",
            "json",
            "--permission-mode",
            "bypassPermissions",
        ),
    ),
    # Force passthrough after `--` (escape hatch for future flag collisions).
    (
        ["--", "--port", "4242"],
        ("--port", "4242"),
    ),
]


@pytest.mark.parametrize("cli_args,expected_grok_args", REAL_GROK_INVOCATIONS)
def test_wrap_grok_forwards_real_cli_shapes(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cli_args: list[str],
    expected_grok_args: tuple[str, ...],
) -> None:
    """Each tuple mirrors a documented `grok` invocation; Headroom must not eat Grok flags."""
    monkeypatch.chdir(tmp_path)
    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with patch("headroom.cli.wrap.shutil.which", return_value="grok"):
        with patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch_tool):
            result = runner.invoke(main, ["wrap", "grok", *cli_args])

    assert result.exit_code == 0, result.output
    assert captured["args"] == expected_grok_args
    env = captured["env"]
    assert isinstance(env, dict)
    assert env[GROK_PROXY_ENV] == "http://127.0.0.1:8787/v1"


def test_wrap_grok_sets_proxy_env(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with patch("headroom.cli.wrap.shutil.which", return_value="grok"):
        with patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch_tool):
            result = runner.invoke(main, ["wrap", "grok", "-p", "fix the bug"])

    assert result.exit_code == 0, result.output
    env = captured["env"]
    assert isinstance(env, dict)
    assert env[GROK_PROXY_ENV] == "http://127.0.0.1:8787/v1"
    assert captured["tool_label"] == "GROK"
    assert captured["agent_type"] == "grok"
    assert captured["args"] == ("-p", "fix the bug")


def test_wrap_grok_keeps_headroom_port_long_option(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Grok owns `-p`; Headroom uses `--port` only so both can coexist on one command line."""
    monkeypatch.chdir(tmp_path)
    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with patch("headroom.cli.wrap.shutil.which", return_value="grok"):
        with patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch_tool):
            result = runner.invoke(
                main,
                ["wrap", "grok", "--port", "9999", "-p", "ship the feature"],
            )

    assert result.exit_code == 0, result.output
    assert captured["port"] == 9999
    assert captured["args"] == ("-p", "ship the feature")
    env = captured["env"]
    assert isinstance(env, dict)
    assert env[GROK_PROXY_ENV] == "http://127.0.0.1:9999/v1"


def test_wrap_grok_forwards_headroom_backend_options(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with patch("headroom.cli.wrap.shutil.which", return_value="grok"):
        with patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch_tool):
            result = runner.invoke(
                main,
                [
                    "wrap",
                    "grok",
                    "--backend",
                    "anyllm",
                    "--anyllm-provider",
                    "groq",
                    "--learn",
                    "--memory",
                    "-p",
                    "refactor auth module",
                ],
            )

    assert result.exit_code == 0, result.output
    assert captured["backend"] == "anyllm"
    assert captured["anyllm_provider"] == "groq"
    assert captured["learn"] is True
    assert captured["memory"] is True
    assert captured["args"] == ("-p", "refactor auth module")


def test_wrap_grok_forwards_no_proxy(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with patch("headroom.cli.wrap.shutil.which", return_value="grok"):
        with patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch_tool):
            result = runner.invoke(main, ["wrap", "grok", "--no-proxy"])

    assert result.exit_code == 0, result.output
    assert captured["no_proxy"] is True


def test_wrap_grok_missing_binary(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    with patch("headroom.cli.wrap.shutil.which", return_value=None):
        result = runner.invoke(main, ["wrap", "grok"])

    assert result.exit_code == 1
    assert "grok" in result.output.lower()


def test_wrap_grok_stub_binary_receives_proxy_env(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: a real subprocess child must see GROK_CLI_CHAT_PROXY_BASE_URL."""
    stub = tmp_path / "grok"
    stub.write_text('#!/bin/sh\nprintf "%s\\n" "$GROK_CLI_CHAT_PROXY_BASE_URL"\nexit 0\n')
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")

    with patch("headroom.cli.wrap._ensure_proxy", return_value=None):
        result = runner.invoke(main, ["wrap", "grok", "--no-proxy"])

    assert result.exit_code == 0, result.output
    assert "http://127.0.0.1:8787/v1" in result.output
    assert "HEADROOM WRAP: GROK" in result.output
