"""Call Google's Gemini API directly, so the council can seat a non-Claude member.

Uses the REST ``generateContent`` endpoint with an API key from
``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``). Web search for the fact-check
gate uses Gemini's ``google_search`` grounding tool; the grounding sources it
returns are kept as citations, which the council treats as leads only.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx

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
