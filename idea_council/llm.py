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
                        data = resp.json()
                        message = (data.get("choices") or [{}])[0].get("message") or {}
                        return Completion(
                            model=model,
                            text=_content_text(message.get("content")),
                            citations=_citations(message),
                        )
                    last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
                    if resp.status_code not in (408, 429) and resp.status_code < 500:
                        break  # not retryable
                if attempt + 1 < self.max_retries:
                    await asyncio.sleep(delay)
                    delay *= 2
        return Completion(model=model, text="", error=last_error)
