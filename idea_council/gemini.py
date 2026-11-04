"""Call Gemini, so the council can seat a non-Claude member.

Two ways in, tried in this order:

* :class:`AgyClient` runs the Antigravity CLI (``agy -p``) signed in with the
  OAuth token in ``AGY_OAUTH_TOKEN_B64``, the same way the Miner repo's
  ``final_polish.py`` does. No API key needed. ``scripts/agy_setup.sh``
  installs ``agy`` and restores the token.
* :class:`GeminiClient` calls the REST ``generateContent`` endpoint with an
  API key from ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``). Web search uses
  Gemini's ``google_search`` grounding tool; the grounding sources are kept
  as citations, which the council treats as leads only.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import httpx

from .claude_code import render_messages
from .llm import Completion

GEMINI_API = "https://generativelanguage.googleapis.com/v1beta"
KEY_VARS: tuple[str, ...] = ("GEMINI_API_KEY", "GOOGLE_API_KEY")


def gemini_key() -> str | None:
    for var in KEY_VARS:
        if os.environ.get(var):
            return os.environ[var]
    return None


def is_gemini(model: str) -> bool:
    return model.startswith("gemini")


def to_gemini_request(messages: list[dict[str, Any]], *, web_search: bool) -> dict[str, Any]:
    """Chat messages -> a generateContent body (system turns become ``systemInstruction``)."""
    system = "\n\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "system")
    contents = [
        {"role": "model" if m.get("role") == "assistant" else "user", "parts": [{"text": str(m.get("content", ""))}]}
        for m in messages
        if m.get("role") != "system"
    ]
    body: dict[str, Any] = {"contents": contents}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    if web_search:
        body["tools"] = [{"google_search": {}}]
    return body


def parse_gemini_response(model: str, data: Any) -> Completion:
    """Turn a generateContent body into a Completion; anything unexpected becomes an error, never an exception."""
    if not isinstance(data, dict):
        return Completion(model=model, text="", error="unexpected response shape")
    if data.get("error"):
        return Completion(model=model, text="", error=f"provider error: {str(data['error'])[:300]}")
    feedback = data.get("promptFeedback")
    if isinstance(feedback, dict) and feedback.get("blockReason"):
        return Completion(model=model, text="", error=f"prompt blocked: {feedback['blockReason']}")
    candidates = data.get("candidates")
    cand = candidates[0] if isinstance(candidates, list) and candidates else None
    if not isinstance(cand, dict):
        return Completion(model=model, text="", error="no candidates in response")
    content = cand.get("content") if isinstance(cand.get("content"), dict) else {}
    parts = content.get("parts") if isinstance(content.get("parts"), list) else []
    text = "".join(
        p["text"] for p in parts if isinstance(p, dict) and isinstance(p.get("text"), str) and not p.get("thought")
    )
    if not text:
        reason = cand.get("finishReason") or "empty"
        return Completion(model=model, text="", error=f"no text in response (finishReason: {reason})")
    citations: list[dict[str, Any]] = []
    grounding = cand.get("groundingMetadata") if isinstance(cand.get("groundingMetadata"), dict) else {}
    for chunk in grounding.get("groundingChunks") or []:
        web = chunk.get("web") if isinstance(chunk, dict) else None
        if isinstance(web, dict) and web.get("uri"):
            citations.append({"url": web["uri"], "title": web.get("title", ""), "content": ""})
    return Completion(model=model, text=text, citations=citations)


async def missing_gemini_models(
    models: tuple[str, ...] | list[str], api_key: str | None = None, timeout_s: float = 30.0
) -> list[str]:
    """Gemini ids the API does not list for this key. Raises httpx.HTTPError if the list is unreachable."""
    key = api_key or gemini_key()
    listed: set[str] = set()
    page: str | None = None
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        while True:
            params = {"pageSize": "1000", **({"pageToken": page} if page else {})}
            resp = await client.get(f"{GEMINI_API}/models", params=params, headers={"x-goog-api-key": key or ""})
            resp.raise_for_status()
            try:
                data = resp.json()
            except ValueError as exc:
                raise httpx.DecodingError("model list is not JSON") from exc
            for m in data.get("models") or []:
                if isinstance(m, dict) and isinstance(m.get("name"), str):
                    listed.add(m["name"].removeprefix("models/"))
            page = data.get("nextPageToken")
            if not page:
                break
    return [m for m in models if is_gemini(m) and m not in listed]


class GeminiClient:
    """An :class:`idea_council.llm.LLM` backed by the Gemini API."""

    def __init__(self, api_key: str | None = None, *, timeout_s: float = 180.0, max_retries: int = 3) -> None:
        self.api_key = api_key or gemini_key()
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set")
        self.timeout_s = timeout_s
        self.max_retries = max(1, max_retries)

    async def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        web_search: bool = False,
    ) -> Completion:
        url = f"{GEMINI_API}/models/{model}:generateContent"
        body = to_gemini_request(messages, web_search=web_search)
        headers = {"x-goog-api-key": self.api_key}
        delay = 2.0
        last_error = "unknown error"
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            for attempt in range(self.max_retries):
                try:
                    resp = await client.post(url, json=body, headers=headers)
                except httpx.HTTPError as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                else:
                    if resp.status_code == 200:
                        try:
                            completion = parse_gemini_response(model, resp.json())
                        except ValueError:
                            completion = Completion(model=model, text="", error=f"non-JSON response: {resp.text[:300]}")
                        if completion.ok:
                            return completion
                        last_error = completion.error or last_error
                        if (completion.error or "").startswith("prompt blocked"):
                            break  # the same prompt will be blocked again
                    else:
                        last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
                        if resp.status_code not in (408, 429) and resp.status_code < 500:
                            break  # bad key, bad model id, bad request: not retryable
                if attempt + 1 < self.max_retries:
                    await asyncio.sleep(delay)
                    delay *= 2
        return Completion(model=model, text="", error=last_error)


class RoutedClient:
    """Send ``gemini*`` model ids to Gemini and everything else to the Claude backend."""

    def __init__(self, claude: Any, gemini: Any) -> None:
        self.claude = claude
        self.gemini = gemini

    async def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        web_search: bool = False,
    ) -> Completion:
        backend = self.gemini if is_gemini(model) else self.claude
        return await backend.complete(model, messages, web_search=web_search)


AGY_CANDIDATES: tuple[str, ...] = ("~/.agy-bin/agy", "~/.local/bin/agy")
AGY_TOKEN_FILE = Path("~/.gemini/antigravity-cli/antigravity-oauth-token")
AGY_SETUP = Path(__file__).resolve().parent.parent / "scripts" / "agy_setup.sh"
MAX_AGY_PROMPT_BYTES = 120_000  # the prompt goes in one argument; Linux caps one at 128 KiB
#: Only these reach agy: no other secrets from the environment, just what it needs to run behind the proxy.
_ENV_KEEP = ("HOME", "PATH", "LANG", "LC_ALL", "TMPDIR", "USER")
_ENV_KEEP_RE = re.compile(r"PROXY|CERT|CA_BUNDLE|CACERTS", re.I)


def agy_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k in _ENV_KEEP or _ENV_KEEP_RE.search(k)}


def _token_ok() -> bool:
    try:
        return AGY_TOKEN_FILE.expanduser().stat().st_size > 0
    except OSError:
        return False


def _locate_agy() -> str | None:
    for cand in (os.environ.get("AGY_BIN", ""), shutil.which("agy") or "", *AGY_CANDIDATES):
        if cand:
            path = Path(cand).expanduser()
            if path.is_file() and os.access(path, os.X_OK):
                return str(path)
    return None


def find_agy(*, setup: bool = True) -> str | None:
    """Path to a signed-in ``agy``; runs ``scripts/agy_setup.sh`` once if it or its token is missing."""
    found = _locate_agy()
    if (not found or not _token_ok()) and setup and os.environ.get("AGY_OAUTH_TOKEN_B64") and AGY_SETUP.is_file():
        env = {**agy_env(), "AGY_OAUTH_TOKEN_B64": os.environ["AGY_OAUTH_TOKEN_B64"]}
        try:
            subprocess.run(["bash", str(AGY_SETUP)], stdin=subprocess.DEVNULL, capture_output=True,
                           timeout=180, env=env, check=False)
        except (subprocess.TimeoutExpired, OSError):
            pass
        found = _locate_agy()
    return found if found and _token_ok() else None


def agy_prompt(messages: list[dict[str, Any]]) -> str:
    """agy has no system-prompt flag, so the stage instructions go first in the one prompt."""
    system, prompt = render_messages(messages)
    return f"{system}\n\n---\n\n{prompt}" if system else prompt


class AgyClient:
    """An :class:`idea_council.llm.LLM` backed by ``agy -p`` (Gemini through the Antigravity CLI).

    Each call runs in an empty temporary directory with ``--sandbox`` and a
    minimal environment. ``web_search`` is ignored: Gemini only sits on the
    council as a member, and the fact-check gate stays with the Claude verifier.
    """

    def __init__(self, binary: str | None = None, *, timeout_s: float = 300.0, max_concurrency: int = 2,
                 max_retries: int = 2) -> None:
        found = binary or find_agy()
        if not found:
            raise RuntimeError("agy is not available (set AGY_OAUTH_TOKEN_B64 and run scripts/agy_setup.sh)")
        self.binary = found
        self.timeout_s = timeout_s
        self.max_retries = max(1, max_retries)
        self._semaphore = asyncio.Semaphore(max(1, max_concurrency))

    def command(self, model: str, prompt: str) -> list[str]:
        return [self.binary, "-p", prompt, "--model", model, "--sandbox", "--disable-slash-commands"]

    async def _run_once(self, cmd: list[str]) -> tuple[int | None, str, str]:
        with tempfile.TemporaryDirectory(prefix="idea-council-agy-") as cwd:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE, cwd=cwd, env=agy_env(), start_new_session=True,
            )
            try:
                out, err = await asyncio.wait_for(proc.communicate(), self.timeout_s)
            except asyncio.TimeoutError:
                try:
                    os.killpg(proc.pid, 9)  # agy can start child processes; take them down too
                except OSError:
                    pass
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
        prompt = agy_prompt(messages)
        if len(prompt.encode("utf-8")) > MAX_AGY_PROMPT_BYTES:
            return Completion(model=model, text="", error="prompt too long for agy")
        cmd = self.command(model, prompt)
        delay = 2.0
        last_error = "unknown error"
        for attempt in range(self.max_retries):
            async with self._semaphore:
                try:
                    code, out, err = await self._run_once(cmd)
                except asyncio.TimeoutError:
                    last_error = f"timed out after {self.timeout_s:.0f}s"
                except OSError as exc:
                    return Completion(model=model, text="", error=f"{type(exc).__name__}: {exc}")
                else:
                    text = out.strip()
                    if code == 0 and text:
                        return Completion(model=model, text=text)
                    tail = (err or out).strip().splitlines()[-1:] or ["empty output"]
                    last_error = f"agy exit {code}: {tail[0][:300]}"
            if attempt + 1 < self.max_retries:
                await asyncio.sleep(delay)
                delay *= 2
        return Completion(model=model, text="", error=last_error)


def gemini_client(timeout_s: float = 300.0) -> Any:
    """agy if it can be set up, otherwise the API key client. Raises RuntimeError when neither works."""
    if os.environ.get("AGY_OAUTH_TOKEN_B64") or _locate_agy():
        binary = find_agy()
        if binary:
            return AgyClient(binary, timeout_s=timeout_s)
    if gemini_key():
        return GeminiClient(timeout_s=timeout_s)
    raise RuntimeError(
        "Gemini is not available: set AGY_OAUTH_TOKEN_B64 (agy sign-in, see scripts/agy_setup.sh) "
        "or GEMINI_API_KEY"
    )


def gemini_available() -> bool:
    """Whether a Gemini credential is configured (cheap check, no install)."""
    return bool(os.environ.get("AGY_OAUTH_TOKEN_B64") or gemini_key())
