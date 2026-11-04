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
import json
import os
import re
from dataclasses import dataclass, field, replace
from typing import Any
from urllib.parse import urlparse

import httpx

from .confidence import Evidence, Verdict, assess, doi_of, work_id
from .llm import LLM
from .parsing import as_list, last_json_block
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
    key: bool = False  # the claim was marked key by a member (중요도 핵심)


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


def _text(value: Any) -> str:
    if isinstance(value, (dict, list)) and value:  # a model may give {"query": ..., "result": ...}
        return json.dumps(value, ensure_ascii=False)
    return value.strip() if isinstance(value, str) else ""


_NO_SEARCH = (  # phrases that only make sense about the search itself
    "no search was", "no search performed", "no counter search", "did not search", "didn't search",
    "search skipped", "skipped the search", "search not performed", "search was not performed",
    "검색 안 함", "검색 안함", "검색하지 않", "검색을 하지 않", "검색을 안 ", "검색 생략", "검색을 생략",
    "검색은 수행하지", "검색을 수행하지",
)
_NO_SEARCH_WHOLE = ("not performed", "not done", "skipped", "생략", "수행하지 않음", "안 함")
_EMPTY = frozenset({"n/a", "na", "none", "null", "-", "없음", "해당 없음"})


def counter_searched(text: str) -> bool:
    """The rule's path 2 needs a counter-evidence search. Only an explicit 'no search' phrase, or an
    empty/placeholder note, means none was done; '반박 근거는 존재하지 않음' is a search result."""
    t = " ".join(text.lower().split())
    if not t or t in _EMPTY:
        return False
    if any(m in t for m in _NO_SEARCH):
        return False
    # A bare "not performed" is about the search; inside a longer note it may be about something else.
    return not (len(t) <= 30 and any(m in t for m in _NO_SEARCH_WHOLE))


_TAGS = re.compile(r"<(script|style)\b.*?</\1>|<[^>]+>", re.IGNORECASE | re.DOTALL)
_NON_WORD = re.compile(r"[\W_]+", re.UNICODE)
MIN_QUOTE_CHARS = 15  # per piece, after normalisation (letters and digits only); shorter proves nothing
MAX_PAGE_BYTES = 3_000_000
OPENALEX_WORK = "https://api.openalex.org/works/{kind}:{value}"


_FOOTNOTE = re.compile(r"\[(?:\d+|[a-z]|citation needed|주\s*\d+)\]", re.IGNORECASE)


def _squash(text: str) -> str:
    """Lowercase and drop footnote markers ([1]) and everything but letters and digits,
    so markup, citations and spacing do not matter."""
    return _NON_WORD.sub("", _FOOTNOTE.sub("", html.unescape(text)).lower())


_SUP_REF = re.compile(r"<sup\b[^>]*class=[\"'][^\"']*reference[^\"']*[\"'][^>]*>.*?</sup>", re.IGNORECASE | re.DOTALL)


def page_text(body: str, content_type: str) -> str:
    if "html" in content_type or body.lstrip()[:1] == "<":
        body = _SUP_REF.sub(" ", body)
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


def _work_keys(work: dict[str, Any]) -> str:
    """OpenAlex id, DOI and a normalised-title key, so a preprint and its published version match."""
    keys = [str(work.get("id") or "")]
    doi = str(work.get("doi") or "").lower().removeprefix("https://doi.org/")
    if doi:
        keys.append(doi)
    title = _squash(str(work.get("title") or ""))
    if len(title) >= 30:  # short titles ("Introduction") are not distinctive
        keys.append("title:" + title[:200])
    return " ".join(k for k in keys if k)


def _openalex_abstract(work: dict[str, Any]) -> str:
    """Abstract text (OpenAlex stores abstracts as an inverted index). The title alone is not used:
    a quote that is only the title shows the paper exists, not that it supports the claim."""
    words: dict[int, str] = {}
    index = work.get("abstract_inverted_index")
    for word, positions in (index if isinstance(index, dict) else {}).items():
        for pos in positions if isinstance(positions, list) else []:
            if isinstance(pos, int):
                words[pos] = word
    return " ".join(words[i] for i in sorted(words))


_ARXIV_URL = re.compile(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})", re.IGNORECASE)
_PMID = re.compile(r"(?:pubmed\.ncbi\.nlm\.nih\.gov|ncbi\.nlm\.nih\.gov/pubmed)/(\d+)", re.IGNORECASE)
_PMCID = re.compile(r"/(PMC\d+)", re.IGNORECASE)


def work_lookup(url: str) -> tuple[str, str] | None:
    """(OpenAlex id kind, value) for a paper URL: DOI, PubMed id or PMC id."""
    arxiv = _ARXIV_URL.search(url)
    if arxiv:  # OpenAlex knows arXiv papers by their DataCite DOI, which joins them to the journal copy
        return "doi", "10.48550/arxiv." + arxiv.group(1)
    doi = doi_of(url)
    if doi:
        return "doi", doi
    pmid = _PMID.search(url)
    if pmid:
        return "pmid", pmid.group(1)
    if "ncbi.nlm.nih.gov" in url.lower():
        pmc = _PMCID.search(url)
        if pmc:
            return "pmcid", pmc.group(1).upper()
    return None


async def _openalex_check(client: httpx.AsyncClient, kind: str, value: str) -> tuple[str | None, str, str, bool]:
    """Return (reason to drop or None, abstract text, OpenAlex work id, whether OpenAlex answered)."""
    params = {"api_key": os.environ["OPENALEX_API_KEY"]} if os.environ.get("OPENALEX_API_KEY") else None
    try:
        resp = await client.get(OPENALEX_WORK.format(kind=kind, value=value), params=params)
    except httpx.HTTPError:
        return None, "", "", False
    if resp.status_code == 404:  # not every real DOI is indexed (OECD, some datasets): open the page instead
        return None, "", "", True
    if resp.status_code != 200:
        return None, "", "", False
    try:
        work = resp.json()
    except ValueError:
        return None, "", "", False
    if not isinstance(work, dict):
        return None, "", "", False
    if work.get("is_retracted"):
        return f"{kind} {value} is retracted", "", "", True
    return None, _openalex_abstract(work), _work_keys(work), True


MAX_REDIRECTS = 5


_META_CHARSET = re.compile(rb"""<meta[^>]+charset=["']?([A-Za-z0-9_\-]+)""", re.IGNORECASE)


def _charset(resp: httpx.Response, raw: bytes) -> str:
    """Header charset, else a <meta charset> in the first 4 KB (EUC-KR pages often only declare it there)."""
    if resp.charset_encoding:
        try:
            "".encode(resp.charset_encoding)
            return resp.charset_encoding
        except LookupError:
            pass
    match = _META_CHARSET.search(raw[:4096])
    if match:
        name = match.group(1).decode("ascii", "ignore")
        try:
            "".encode(name)
            return name
        except LookupError:
            pass
    return "utf-8"


async def _fetch(client: httpx.AsyncClient, url: str) -> tuple[int, str, str, str]:
    """GET one hop without following redirects: (status, content type, text body, redirect target)."""
    async with client.stream("GET", url, follow_redirects=False) as resp:
        if resp.is_redirect and resp.headers.get("location"):
            return resp.status_code, "", "", str(resp.url.join(resp.headers["location"]))
        content_type = resp.headers.get("content-type", "").lower()
        chunks, size = [], 0
        if resp.status_code < 400 and ("text" in content_type or "html" in content_type or "xml" in content_type):
            async for chunk in resp.aiter_bytes():
                chunks.append(chunk)
                size += len(chunk)
                if size >= MAX_PAGE_BYTES:
                    break
        raw = b"".join(chunks)
        return resp.status_code, content_type, raw.decode(_charset(resp, raw), errors="replace"), ""


async def check_source(client: httpx.AsyncClient, ev: Evidence) -> tuple[Evidence | None, str]:
    """Keep, annotate or drop one piece of evidence. Never raises.

    Dropped: non-public URL (checked on every redirect hop), unreachable or missing page,
    unknown or retracted DOI, no quote, or a quote that is not in the page text.
    Kept but marked "unchecked: ...": pages that refuse bots (401/403/429) or are not text (PDF).
    """
    if ev.checked.startswith("drop: "):
        return None, f"{ev.url}: {ev.checked.removeprefix('drop: ')}"
    try:
        kept, reason = await _check_source(client, ev)
    except Exception as exc:  # noqa: BLE001  a bad URL from the model must not end the whole run
        kept, reason = None, f"{ev.url}: check failed ({type(exc).__name__})"
    if kept is None and _keep_unconfirmed_rebuttal(ev, reason):
        # Dropping a rebuttal moves a claim toward a pass, so a credible one that could not be read
        # (paywall, 404, protocol error) stays and counts once on the contra side.
        return replace(ev, checked=f"unchecked: rebuttal not confirmed ({reason.split(': ', 1)[-1]})"), ""
    return kept, reason


def _keep_unconfirmed_rebuttal(ev: Evidence, reason: str) -> bool:
    if ev.stance != "contradict" or ev.grade not in ("A", "B") or not ev.quote.strip():
        return False
    return not any(m in reason for m in ("not a public URL", "retracted", "too many redirects"))


_CHALLENGE = ("captcha", "cookies must be enabled", "enable cookies", "enable javascript", "just a moment",
              "checking your browser", "are you a robot", "access denied", "verify you are human")


def _is_shell(body: str) -> bool:
    """A tiny page that only frames or scripts in its content (law.go.kr, some portals)."""
    low = body.lower()
    framed = "<iframe" in low or "<frame" in low
    js_redirect = "location.href" in low or "location.replace" in low or 'http-equiv="refresh"' in low
    return len(_TAGS.sub(" ", body).split()) < 60 and (framed or js_redirect)


# Statistics portals that draw their tables with scripts behind a large menu, so the size test misses them.
_SCRIPT_PORTALS = ("index.go.kr", "kosis.kr", "data-explorer.oecd.org", "data.oecd.org", "ecos.bok.or.kr",
                   "data.worldbank.org", "data.imf.org", "ourworldindata.org/grapher")


def _script_rendered(body: str, url: str = "") -> bool:
    """A page whose numbers are drawn by scripts (World Bank data, e-나라지표, KOSIS): a missing quote
    says nothing about the real page. Either a known portal, or visible text is a sliver of the HTML."""
    low = url.lower()
    if any(p in low for p in _SCRIPT_PORTALS):
        return True
    return len(body) > 20000 and len(_TAGS.sub(" ", body).strip()) < 0.03 * len(body)


def _looks_like_challenge(body: str) -> bool:
    """A bot wall is a short page; an article that merely mentions 'captcha' or 'enable javascript' is not."""
    if len(body) >= 30000 or len(_TAGS.sub(" ", body).split()) >= 200:
        return False
    low = body.lower()
    return any(m in low for m in _CHALLENGE)


async def _check_source(client: httpx.AsyncClient, ev: Evidence) -> tuple[Evidence | None, str]:
    if not ev.quote.strip():
        return None, f"{ev.url}: no quote"
    lookup = work_lookup(ev.url)
    abstract, openalex_ok = "", True
    if lookup:
        reason, abstract, work, openalex_ok = await _openalex_check(client, *lookup)
        if reason:
            return None, f"{ev.url}: {reason}"
        if work:
            ev = replace(ev, work=f"{ev.work} {work}".strip())  # ties every mirror of the paper together
    note = "" if openalex_ok else "; OpenAlex unavailable, retraction not checked"
    # A resolved paper's abstract can stand in for a publisher page that blocks scripts. This is safe
    # on any host: the work id makes every URL of the same paper count as one source.
    if abstract and quote_in_page(ev.quote, _squash(abstract)):
        return replace(ev, checked="quote found (OpenAlex abstract)" + note), ""

    url = ev.url
    for _ in range(MAX_REDIRECTS + 1):
        if not await _public_http_url(url):
            return None, f"{ev.url}: not a public URL ({url})"
        try:
            status, content_type, body, location = await _fetch(client, url)
        except httpx.TimeoutException:
            return replace(ev, checked="unchecked: timed out" + note), ""
        except httpx.HTTPError as exc:
            return None, f"{ev.url}: unreachable ({type(exc).__name__})"
        if location:
            url = location
            continue
        break
    else:
        return None, f"{ev.url}: too many redirects"
    if status in (401, 403, 429) or status >= 500:  # refused, rate-limited, bot wall or server trouble
        return replace(ev, checked="unchecked: access refused" + note), ""
    if status >= 400:
        return None, f"{ev.url}: HTTP {status}"
    if not body:
        return replace(ev, checked="unchecked: not a text page" + note), ""
    found = quote_in_page(ev.quote, page_text(body, content_type))
    if found:
        return replace(ev, checked="quote found" + note), ""
    if status != 200 or _looks_like_challenge(body) or _is_shell(body):  # we did not see the real text
        return replace(ev, checked="unchecked: bot check page" + note), ""
    if _script_rendered(body, url):
        return replace(ev, checked="unchecked: script-rendered page" + note), ""
    return None, f"{ev.url}: quote not found in page"


def _origin_lookup(origin: str) -> tuple[str, str] | None:
    wid = work_id(origin)
    return ("doi", wid) if wid else None


async def _with_origin_work(client: httpx.AsyncClient, ev: Evidence) -> Evidence:
    """When the origin names a paper, add that paper's keys, so a story about a study joins the study
    (and its preprint or published twin) instead of counting beside it."""
    lookup = _origin_lookup(ev.origin)
    if not lookup:
        return ev
    try:
        reason, _, keys, _ = await _openalex_check(client, *lookup)
    except Exception:  # noqa: BLE001  enrichment only; the doi key from the origin still applies
        return ev
    if reason and "retracted" in reason:  # a story about a retracted paper is not evidence
        return replace(ev, checked=f"drop: origin {reason}")
    return replace(ev, work=f"{ev.work} {keys}".strip()) if keys else ev


async def filter_sources(evidence: list[Evidence], timeout_s: float = 20.0) -> tuple[list[Evidence], list[str]]:
    if not evidence:
        return [], []
    headers = {"User-Agent": "idea-council/0.2 (source check)"}
    async with httpx.AsyncClient(timeout=timeout_s, headers=headers) as client:
        evidence = list(await asyncio.gather(*(_with_origin_work(client, e) for e in evidence)))
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
        data = last_json_block(completion.text)
        if data is None:
            return ClaimCheck(claim=claim, verdict=assess([]), error="verifier reply had no JSON block")
        evidence = [e for e in (_to_evidence(x) for x in as_list(data.get("evidence")) if isinstance(x, dict)) if e]
        # The flag counts only if the verifier also listed a credible rebuttal (grade A/B, no conflict of
        # interest), judged before the source check so a rebuttal page that could not be fetched still counts.
        rebuttal = _flag(data.get("rebuttal_found")) and any(
            e.stance == "contradict" and e.grade in ("A", "B") and not e.conflict_of_interest for e in evidence
        )
        dropped: list[str] = []
        if self.check_urls:
            evidence, dropped = await filter_sources(evidence)
        counter = _text(data.get("counter_search"))
        return ClaimCheck(
            claim=claim,
            verdict=assess(
                evidence,
                counter_searched=counter_searched(counter),
                rebuttal_found=rebuttal,
            ),
            evidence=evidence,
            dropped=dropped,
            counter_search=counter,
            notes=_text(data.get("notes")),
        )
