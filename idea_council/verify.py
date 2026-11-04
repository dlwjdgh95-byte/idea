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
import os
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


_SUPPORT = ("support", "supports", "supporting", "지지")
_CONTRA = ("contra", "refut", "oppos", "against", "disput", "rebut", "반대", "반박")


def _stance(value: Any) -> str | None:
    """Map the verifier's stance to support/contradict; anything else (neutral, mixed, ...) is dropped."""
    text = str(value or "").strip().lower()
    if text in _SUPPORT:
        return "support"
    if text.startswith(_CONTRA):
        return "contradict"
    return None


def _flag(value: Any) -> bool:
    return value is True or str(value).strip().lower() == "true"


def _to_evidence(item: dict[str, Any]) -> Evidence | None:
    url = str(item.get("url", "")).strip()
    grade = str(item.get("grade", "C")).strip().upper()[:1]
    stance = _stance(item.get("stance"))
    if not url.startswith(("http://", "https://")) or stance is None:
        return None
    return Evidence(
        url=url,
        grade=grade if grade in ("A", "B", "C") else "C",  # type: ignore[arg-type]
        stance=stance,  # type: ignore[arg-type]
        operator=str(item.get("operator", "")).strip(),
        origin=str(item.get("origin", "")).strip(),
        primary=_flag(item.get("primary")),
        conflict_of_interest=_flag(item.get("conflict_of_interest")),
        quote=str(item.get("quote", "")).strip(),
    )


def counter_searched(text: str) -> bool:
    """The rule's path 2 needs a counter-evidence search; an empty or 'not done' note does not count."""
    t = text.strip().lower()
    if not t:
        return False
    return not any(m in t for m in ("not performed", "not done", "none performed", "안 함", "하지 않", "생략", "n/a"))


_DOI = re.compile(r"(10\.\d{4,9}/[^\s?#]+)", re.IGNORECASE)
_TAGS = re.compile(r"<(script|style)\b.*?</\1>|<[^>]+>", re.IGNORECASE | re.DOTALL)
_NON_WORD = re.compile(r"[\W_]+", re.UNICODE)
MIN_QUOTE_CHARS = 15  # per piece, after normalisation (letters and digits only); shorter proves nothing
MAX_PAGE_BYTES = 3_000_000
OPENALEX_WORK = "https://api.openalex.org/works/doi:{doi}"


def _squash(text: str) -> str:
    """Lowercase and drop everything but letters and digits, so markup and spacing do not matter."""
    return _NON_WORD.sub("", html.unescape(text).lower())


def page_text(body: str, content_type: str) -> str:
    if "html" in content_type or body.lstrip()[:1] == "<":
        body = _TAGS.sub(" ", body)
    return _squash(body)


def quote_in_page(quote: str, page: str, max_gap: int = 2000) -> bool:
    """True if the quote's '...'-separated pieces all appear in the page, in order and close together."""
    pieces = [_squash(p) for p in re.split(r"\.\.\.|…", quote)]
    pieces = [p for p in pieces if p]
    if not pieces or any(len(p) < MIN_QUOTE_CHARS for p in pieces):
        return False
    start = page.find(pieces[0])
    while start != -1:
        pos, ok = start + len(pieces[0]), True
        for piece in pieces[1:]:
            nxt = page.find(piece, pos)
            if nxt == -1 or nxt - pos > max_gap:
                ok = False
                break
            pos = nxt + len(piece)
        if ok:
            return True
        start = page.find(pieces[0], start + 1)
    return False


def _global_ip(text: str) -> bool:
    try:
        return ipaddress.ip_address(text.split("%", 1)[0]).is_global
    except ValueError:
        return False


async def _public_http_url(url: str) -> bool:
    """True only if every address the host resolves to is public (blocks localhost, private, link-local)."""
    try:
        parts = urlparse(url)
        host = (parts.hostname or "").lower()
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError:
        return False
    if parts.scheme not in ("http", "https") or not host:
        return False
    try:
        addresses = await _resolve(host, port)
    except OSError:
        return False
    return bool(addresses) and all(_global_ip(a) for a in addresses)


async def _resolve(host: str, port: int) -> list[str]:
    infos = await asyncio.get_running_loop().getaddrinfo(host, port)
    return [info[4][0] for info in infos]


_DOI_TAIL = re.compile(
    r"(?:v\d+)?(?:[./](?:full|abstract|pdf|epdf|html|fulltext|full-text|supplemental|references)(?:\.pdf)?)*$",
    re.IGNORECASE,
)


def doi_of(url: str) -> str | None:
    """DOI in a URL, without version/view suffixes that publishers append (v3, .full, /abstract, /pdf)."""
    match = _DOI.search(unquote(url))
    if not match:
        return None
    doi = match.group(1).rstrip(".,;)/")
    trimmed = _DOI_TAIL.sub("", doi).rstrip(".,;)/")
    return trimmed if "/" in trimmed and trimmed.split("/", 1)[1] else doi


def _openalex_text(work: dict[str, Any]) -> str:
    """Title plus abstract (OpenAlex stores abstracts as an inverted index)."""
    words: dict[int, str] = {}
    for word, positions in (work.get("abstract_inverted_index") or {}).items():
        for pos in positions:
            words[pos] = word
    abstract = " ".join(words[i] for i in sorted(words))
    return f"{work.get('title') or ''} {abstract}"


async def _openalex_check(client: httpx.AsyncClient, doi: str) -> tuple[str | None, str, bool]:
    """Return (reason to drop or None, title+abstract text, whether OpenAlex answered)."""
    params = {"api_key": os.environ["OPENALEX_API_KEY"]} if os.environ.get("OPENALEX_API_KEY") else None
    try:
        resp = await client.get(OPENALEX_WORK.format(doi=doi), params=params)
    except httpx.HTTPError:
        return None, "", False
    if resp.status_code == 404:
        return f"DOI {doi} not found in OpenAlex", "", True
    if resp.status_code != 200:
        return None, "", False
    try:
        work = resp.json()
    except ValueError:
        return None, "", False
    if not isinstance(work, dict):
        return None, "", False
    if work.get("is_retracted"):
        return f"DOI {doi} is retracted", "", True
    return None, _openalex_text(work), True


MAX_REDIRECTS = 5


async def check_source(client: httpx.AsyncClient, ev: Evidence) -> tuple[Evidence | None, str]:
    """Keep, annotate or drop one piece of evidence. Never raises.

    Dropped: non-public URL (checked on every redirect hop), unreachable or missing page,
    unknown or retracted DOI, no quote, or a quote that is not in the page text.
    Kept but marked "unchecked: ...": pages that refuse bots (401/403/429) or are not text (PDF).
    """
    try:
        return await _check_source(client, ev)
    except Exception as exc:  # noqa: BLE001  a bad URL from the model must not end the whole run
        return None, f"{ev.url}: check failed ({type(exc).__name__})"


async def _check_source(client: httpx.AsyncClient, ev: Evidence) -> tuple[Evidence | None, str]:
    if not ev.quote.strip():
        return None, f"{ev.url}: no quote"
    doi = doi_of(ev.url)
    record, openalex_ok = "", True  # title + abstract; publisher pages often hide both from scripts
    if doi:
        reason, record, openalex_ok = await _openalex_check(client, doi)
        if reason:
            return None, f"{ev.url}: {reason}"
    note = "" if openalex_ok else "; OpenAlex unavailable, retraction not checked"
    if record and quote_in_page(ev.quote, _squash(record)):
        return replace(ev, checked="quote found (OpenAlex record)" + note), ""

    url = ev.url
    for _ in range(MAX_REDIRECTS + 1):
        if not await _public_http_url(url):
            return None, f"{ev.url}: not a public URL ({url})"
        async with client.stream("GET", url, follow_redirects=False) as resp:
            if resp.is_redirect and resp.headers.get("location"):
                url = str(resp.url.join(resp.headers["location"]))
                continue
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
            break
    else:
        return None, f"{ev.url}: too many redirects"
    if status in (401, 403, 429):
        return replace(ev, checked="unchecked: access refused" + note), ""
    if status >= 400:
        return None, f"{ev.url}: HTTP {status}"
    if not body:
        return replace(ev, checked="unchecked: not a text page" + note), ""
    if not quote_in_page(ev.quote, page_text(body, content_type)):
        return None, f"{ev.url}: quote not found in page"
    return replace(ev, checked="quote found" + note), ""


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
        counter = str(data.get("counter_search", "")).strip()
        return ClaimCheck(
            claim=claim,
            verdict=assess(evidence, counter_searched=counter_searched(counter)),
            evidence=evidence,
            dropped=dropped,
            counter_search=counter,
            notes=str(data.get("notes", "")).strip(),
        )
