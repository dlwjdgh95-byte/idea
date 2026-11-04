"""Run council calls through the local Claude Code CLI (``claude -p``) instead of OpenRouter.

The CLI uses its own sign-in, so no API key is needed. Only Claude models are
available, which removes the cross-vendor diversity the OpenRouter line-up
has. The default line-up (``config.CLAUDE_MEMBERS``) mixes model sizes
instead, and the council rotates roles and hides model names from the
chairman. This is the default backend.

Each call is one headless session with:

* the stage prompt as the system prompt (replacing Claude Code's own),
* no tools at all, or only WebSearch and WebFetch for the fact-check gate,
* no MCP servers, no settings files, no slash commands, no saved session,
* an empty working directory, so no project CLAUDE.md is read.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
from typing import Any

from .llm import Completion

WEB_TOOLS = "WebSearch,WebFetch"
MAX_SYSTEM_PROMPT_CHARS = 100_000  # stays well under Linux's 128 KiB limit for one argument


def render_messages(messages: list[dict[str, Any]]) -> tuple[str, str]:
    """Split chat messages into (system prompt, stdin prompt).

    A single user turn is passed through unchanged; longer histories are
    written out as a labelled transcript.
    """
    system = "\n\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "system")
    turns = [m for m in messages if m.get("role") != "system"]
    if len(turns) == 1 and turns[0].get("role") == "user":
        return system, str(turns[0].get("content", ""))
    names = {"user": "사용자", "assistant": "어시스턴트"}
    prompt = "\n\n".join(
        f"### {names.get(str(m.get('role')), str(m.get('role')))}\n{m.get('content', '')}" for m in turns
    )
    return system, prompt


def parse_cli_output(model: str, stdout: str) -> Completion:
    """Read ``--output-format json`` output; anything unexpected becomes an error, never an exception."""
    try:
        data = json.loads(stdout)
    except ValueError:
        return Completion(model=model, text="", error=f"non-JSON CLI output: {stdout[:300]}")
    if not isinstance(data, dict):
        return Completion(model=model, text="", error="unexpected CLI output shape")
    result = data.get("result")
    if data.get("is_error") or data.get("subtype") not in (None, "success"):
        detail = result if isinstance(result, str) and result else data.get("subtype") or data.get("api_error_status")
        return Completion(model=model, text="", error=f"CLI error: {str(detail)[:300]}")
    if not isinstance(result, str):
        return Completion(model=model, text="", error="CLI output has no text result")
    return Completion(model=model, text=result)


class ClaudeCodeClient:
    """An :class:`idea_council.llm.LLM` backed by ``claude -p``."""

    def __init__(
        self,
        *,
        binary: str | None = None,
        timeout_s: float = 300.0,
        max_concurrency: int = 4,
        max_retries: int = 2,
        workdir: str | None = None,
    ) -> None:
        found = binary or os.environ.get("CLAUDE_BIN") or shutil.which("claude")
        if not found:
            raise RuntimeError("claude CLI not found (install Claude Code or set CLAUDE_BIN)")
        self.binary = found
        self.timeout_s = timeout_s
        self.max_retries = max(1, max_retries)
        self.workdir = workdir or tempfile.mkdtemp(prefix="idea-council-")
        self._semaphore = asyncio.Semaphore(max(1, max_concurrency))

    def command(self, model: str, system: str, *, web_search: bool) -> list[str]:
        cmd = [
            self.binary, "-p",
            "--model", model,
            "--output-format", "json",
            "--tools", WEB_TOOLS if web_search else "",
            "--strict-mcp-config",
            "--setting-sources", "",
            "--disable-slash-commands",
            "--no-session-persistence",
        ]
        if web_search:
            cmd += ["--allowedTools", WEB_TOOLS]
        if system:
            cmd += ["--system-prompt", system]
        return cmd

    async def _run_once(self, cmd: list[str], prompt: str) -> tuple[int | None, str, str]:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.workdir,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(prompt.encode("utf-8")), self.timeout_s)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise
        return proc.returncode, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")

    async def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        web_search: bool = False,
    ) -> Completion:
        system, prompt = render_messages(messages)
        if len(system) > MAX_SYSTEM_PROMPT_CHARS:
            return Completion(model=model, text="", error="system prompt too long for the CLI")
        cmd = self.command(model, system, web_search=web_search)
        delay = 2.0
        last_error = "unknown error"
        for attempt in range(self.max_retries):
            async with self._semaphore:
                try:
                    code, out, err = await self._run_once(cmd, prompt)
                except asyncio.TimeoutError:
                    last_error = f"timed out after {self.timeout_s:.0f}s"
                except OSError as exc:  # the binary cannot start; retrying will not help
                    return Completion(model=model, text="", error=f"{type(exc).__name__}: {exc}")
                else:
                    completion = parse_cli_output(model, out)
                    if completion.ok and code == 0:
                        return completion
                    last_error = completion.error or f"exit {code}: {err[:300]}"
            if attempt + 1 < self.max_retries:
                await asyncio.sleep(delay)
                delay *= 2
        return Completion(model=model, text="", error=last_error)
