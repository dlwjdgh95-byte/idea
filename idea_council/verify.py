"""Stage 2.5: the fact-check gate.

The verifier model searches the web and returns evidence. Unreachable URLs
are dropped (a cheap guard against invented links), then the confidence is
computed by :func:`idea_council.confidence.assess`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import httpx

from .confidence import Evidence, Verdict, assess
from .llm import LLM
from .parsing import last_json_block
from .prompts import verify_messages


@dataclass
class ClaimCheck:
    claim: str
    verdict: Verdict
    evidence: list[Evidence] = field(default_factory=list)
    dropped_urls: list[str] = field(default_factory=list)
    counter_search: str = ""
    notes: str = ""
    error: str | None = None


def _to_evidence(item: dict[str, Any]) -> Evidence | None:
    url = str(item.get("url", "")).strip()
    grade = str(item.get("grade", "C")).strip().upper()[:1]
    stance = str(item.get("stance", "support")).strip().lower()
    if not url.startswith(("http://", "https://")):
        return None
    return Evidence(
        url=url,
        grade=grade if grade in ("A", "B", "C") else "C",  # type: ignore[arg-type]
        stance="contradict" if stance.startswith("contra") else "support",
        operator=str(item.get("operator", "")).strip(),
        origin=str(item.get("origin", "")).strip(),
        primary=bool(item.get("primary", False)),
        conflict_of_interest=bool(item.get("conflict_of_interest", False)),
        quote=str(item.get("quote", "")).strip(),
    )


async def _reachable(client: httpx.AsyncClient, url: str) -> bool:
    try:
        resp = await client.head(url, follow_redirects=True)
        if resp.status_code in (403, 405) or resp.status_code >= 500:
            # Many sites refuse HEAD or bots; a GET decides.
            resp = await client.get(url, follow_redirects=True)
        return resp.status_code < 400 or resp.status_code in (401, 403)
    except httpx.HTTPError:
        return False


async def filter_reachable(evidence: list[Evidence], timeout_s: float = 15.0) -> tuple[list[Evidence], list[str]]:
    if not evidence:
        return [], []
    headers = {"User-Agent": "idea-council/0.1 (source check)"}
    async with httpx.AsyncClient(timeout=timeout_s, headers=headers) as client:
        results = await asyncio.gather(*(_reachable(client, e.url) for e in evidence))
    kept = [e for e, ok in zip(evidence, results) if ok]
    dropped = [e.url for e, ok in zip(evidence, results) if not ok]
    return kept, dropped


class Verifier:
    def __init__(self, llm: LLM, model: str, *, check_urls: bool = True) -> None:
        self.llm = llm
        self.model = model
        self.check_urls = check_urls

    async def verify(self, claim: str, context: str = "") -> ClaimCheck:
        completion = await self.llm.complete(self.model, verify_messages(claim, context), web_search=True)
        if not completion.ok:
            return ClaimCheck(claim=claim, verdict=assess([]), error=completion.error)
        data = last_json_block(completion.text) or {}
        evidence = [e for e in (_to_evidence(x) for x in data.get("evidence") or [] if isinstance(x, dict)) if e]
        dropped: list[str] = []
        if self.check_urls:
            evidence, dropped = await filter_reachable(evidence)
        return ClaimCheck(
            claim=claim,
            verdict=assess(evidence),
            evidence=evidence,
            dropped_urls=dropped,
            counter_search=str(data.get("counter_search", "")).strip(),
            notes=str(data.get("notes", "")).strip(),
        )
