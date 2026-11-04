"""Stage 2.5: the fact-check gate.

The verifier model searches the web and returns evidence. Web search only
hands the model snippets, so the code itself opens every source: missing
pages, unknown or retracted DOIs (checked in OpenAlex) and quotes that are
not in the page are dropped before the confidence is computed by
:func:`idea_council.confidence.assess`.
"""

from __future__ import annotations

import asyncio
import html
import ipaddress
import re
from dataclasses import dataclass, field, replace
from typing import Any
from urllib.parse import unquote, urlparse

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
    dropped: list[str] = field(default_factory=list)
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


_DOI = re.compile(r"(10\.\d{4,9}/[^\s?#]+)", re.IGNORECASE)
_TAGS = re.compile(r"<(script|style)\b.*?</\1>|<[^>]+>", re.IGNORECASE | re.DOTALL)
_NON_WORD = re.compile(r"[\W_]+", re.UNICODE)
MIN_QUOTE_CHARS = 12  # after normalisation; shorter quotes prove nothing
MAX_PAGE_BYTES = 3_000_000
OPENALEX_WORK = "https://api.openalex.org/works/doi:{doi}"


def _squash(text: str) -> str:
    """Lowercase and drop everything but letters and digits, so markup and spacing do not matter."""
    return _NON_WORD.sub("", html.unescape(text).lower())


def page_text(body: str, content_type: str) -> str:
    if "html" in content_type or body.lstrip()[:1] == "<":
        body = _TAGS.sub(" ", body)
    return _squash(body)


def quote_in_page(quote: str, page: str) -> bool:
    """True if the quote, or each of its '...'-separated pieces, appears in the page text."""
    pieces = [_squash(p) for p in re.split(r"\.\.\.|…", quote)]
    pieces = [p for p in pieces if p]
    if not pieces or sum(len(p) for p in pieces) < MIN_QUOTE_CHARS:
        return False
    return all(p in page for p in pieces)


def _public_http_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    if not host or host == "localhost" or host.endswith((".local", ".internal", ".localhost")):
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return True
    return ip.is_global


def doi_of(url: str) -> str | None:
    match = _DOI.search(unquote(url))
    return match.group(1).rstrip(".,;)") if match else None


def _openalex_text(work: dict[str, Any]) -> str:
    """Title plus abstract (OpenAlex stores abstracts as an inverted index)."""
    words: dict[int, str] = {}
    for word, positions in (work.get("abstract_inverted_index") or {}).items():
        for pos in positions:
            words[pos] = word
    abstract = " ".join(words[i] for i in sorted(words))
    return f"{work.get('title') or ''} {abstract}"


async def _openalex_check(client: httpx.AsyncClient, doi: str) -> tuple[str | None, str]:
    """Return (reason to drop or None, title+abstract text). Network trouble is not a reason to drop."""
    try:
        resp = await client.get(OPENALEX_WORK.format(doi=doi))
    except httpx.HTTPError:
        return None, ""
    if resp.status_code == 404:
        return f"DOI {doi} not found in OpenAlex", ""
    if resp.status_code != 200:
        return None, ""
    try:
        work = resp.json()
    except ValueError:
        return None, ""
    if not isinstance(work, dict):
        return None, ""
    if work.get("is_retracted"):
        return f"DOI {doi} is retracted", ""
    return None, _openalex_text(work)


async def check_source(client: httpx.AsyncClient, ev: Evidence) -> tuple[Evidence | None, str]:
    """Keep, annotate or drop one piece of evidence.

    Dropped: non-public URL, unreachable or missing page, unknown or retracted DOI,
    no quote, or a quote that is not in the fetched page text.
    Kept but marked unchecked: pages that refuse bots (401/403/429) or are not text (PDF).
    """
    if not _public_http_url(ev.url):
        return None, f"{ev.url}: not a public URL"
    if not ev.quote.strip():
        return None, f"{ev.url}: no quote"
    doi = doi_of(ev.url)
    record = ""  # title + abstract from OpenAlex; publisher pages often hide both from scripts
    if doi:
        reason, record = await _openalex_check(client, doi)
        if reason:
            return None, f"{ev.url}: {reason}"
    if record and quote_in_page(ev.quote, _squash(record)):
        return replace(ev, checked="quote found (OpenAlex record)"), ""
    try:
        async with client.stream("GET", ev.url, follow_redirects=True) as resp:
            status = resp.status_code
            content_type = resp.headers.get("content-type", "").lower()
            chunks, size = [], 0
            if status < 400 and ("text" in content_type or "html" in content_type or "xml" in content_type):
                async for chunk in resp.aiter_bytes():
                    chunks.append(chunk)
                    size += len(chunk)
                    if size >= MAX_PAGE_BYTES:
                        break
            body = b"".join(chunks).decode(resp.encoding or "utf-8", errors="replace")
    except httpx.HTTPError as exc:
        return None, f"{ev.url}: unreachable ({type(exc).__name__})"
    if status in (401, 403, 429):
        return replace(ev, checked="unchecked: access refused"), ""
    if status >= 400:
        return None, f"{ev.url}: HTTP {status}"
    if not body:
        return replace(ev, checked="unchecked: not a text page"), ""
    if not quote_in_page(ev.quote, page_text(body, content_type)):
        return None, f"{ev.url}: quote not found in page"
    return replace(ev, checked="quote found"), ""


async def filter_sources(evidence: list[Evidence], timeout_s: float = 20.0) -> tuple[list[Evidence], list[str]]:
    if not evidence:
        return [], []
    headers = {"User-Agent": "idea-council/0.2 (source check)"}
    async with httpx.AsyncClient(timeout=timeout_s, headers=headers) as client:
        results = await asyncio.gather(*(check_source(client, e) for e in evidence))
    kept = [e for e, _ in results if e is not None]
    dropped = [reason for e, reason in results if e is None]
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
            evidence, dropped = await filter_sources(evidence)
        return ClaimCheck(
            claim=claim,
            verdict=assess(evidence),
            evidence=evidence,
            dropped=dropped,
            counter_search=str(data.get("counter_search", "")).strip(),
            notes=str(data.get("notes", "")).strip(),
        )
