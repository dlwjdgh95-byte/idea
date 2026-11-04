"""Minimal async chat client for OpenRouter, plus the protocol the council uses.

Web search uses OpenRouter's ``openrouter:web_search`` server tool, which the
OpenRouter docs list as the recommended replacement for the deprecated
``:online`` suffix and ``web`` plugin.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


@dataclass
class Completion:
    model: str
    text: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class LLM(Protocol):
    async def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        web_search: bool = False,
    ) -> Completion: ...


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # some providers return content parts
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return ""


def _citations(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect url citations if the provider returned annotations.

    The council never trusts these alone; they are only kept as extra leads
    next to the evidence list the verifier is asked to return.
    """
    found: list[dict[str, Any]] = []
    for ann in message.get("annotations") or []:
        if not isinstance(ann, dict) or ann.get("type") != "url_citation":
            continue
        inner = ann.get("url_citation") if isinstance(ann.get("url_citation"), dict) else ann
        if inner.get("url"):
            found.append(
                {"url": inner["url"], "title": inner.get("title", ""), "content": inner.get("content", "")}
            )
    return found


def _parse_response(model: str, resp: httpx.Response) -> Completion:
    """Turn a 200 response into a Completion; bad bodies become errors, never exceptions."""
    try:
        data = resp.json()
    except ValueError:
        return Completion(model=model, text="", error=f"non-JSON response: {resp.text[:300]}")
    if not isinstance(data, dict):
        return Completion(model=model, text="", error="unexpected response shape")
    if data.get("error"):  # OpenRouter can report upstream failures inside a 200
        return Completion(model=model, text="", error=f"provider error: {str(data['error'])[:300]}")
    choice = (data.get("choices") or [{}])[0]
    if not isinstance(choice, dict):
        return Completion(model=model, text="", error="unexpected choice shape")
    if choice.get("error"):
        return Completion(model=model, text="", error=f"provider error: {str(choice['error'])[:300]}")
    message = choice.get("message") or {}
    return Completion(model=model, text=_content_text(message.get("content")), citations=_citations(message))


async def missing_models(models: tuple[str, ...] | list[str], timeout_s: float = 30.0) -> list[str]:
    """Return the model ids OpenRouter does not list. Raises httpx.HTTPError if the list is unreachable."""
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        resp = await client.get(OPENROUTER_MODELS_URL)
        resp.raise_for_status()
        try:
            data = resp.json()
        except ValueError as exc:
            raise httpx.DecodingError("model list is not JSON") from exc
        listed = {m.get("id") for m in (data.get("data") if isinstance(data, dict) else None) or [] if isinstance(m, dict)}
    return [m for m in models if m not in listed]


class OpenRouterClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        timeout_s: float = 180.0,
        max_retries: int = 3,
        web_search_max_results: int = 5,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.web_search_max_results = web_search_max_results
        self.extra_headers = extra_headers or {}

    async def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        web_search: bool = False,
    ) -> Completion:
        payload: dict[str, Any] = {"model": model, "messages": messages}
        if web_search:
            payload["tools"] = [
                {
                    "type": "openrouter:web_search",
                    "parameters": {"max_results": self.web_search_max_results},
                }
            ]
        headers = {"Authorization": f"Bearer {self.api_key}", **self.extra_headers}

        delay = 2.0
        last_error = "unknown error"
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            for attempt in range(self.max_retries):
                try:
                    resp = await client.post(OPENROUTER_URL, json=payload, headers=headers)
                except httpx.HTTPError as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                else:
                    if resp.status_code == 200:
                        completion = _parse_response(model, resp)
                        if completion.ok:
                            return completion
                        last_error = completion.error or last_error  # retry: often transient upstream errors
                        if attempt + 1 < self.max_retries:
                            await asyncio.sleep(delay)
                            delay *= 2
                        continue
                    last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
                    if resp.status_code not in (408, 429) and resp.status_code < 500:
                        break  # not retryable
                if attempt + 1 < self.max_retries:
                    await asyncio.sleep(delay)
                    delay *= 2
        return Completion(model=model, text="", error=last_error)
