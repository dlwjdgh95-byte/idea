"""Deterministic confidence rubric (사실 검증 규칙).

The verifier model only gathers evidence. The confidence percentage is
computed here from that evidence, so the number follows the written rule
instead of a model's mood.

Independence: two sources count as one when they share an operator
(운영 주체) or an origin (원출처). The rule requires both to be shown to
differ, so a source missing either one cannot be shown to be independent;
all such sources collapse into a single group. Grade C sources and sources
with a conflict of interest are never counted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Literal
from urllib.parse import unquote, urlparse

Grade = Literal["A", "B", "C"]
Stance = Literal["support", "contradict"]
Path = Literal["path1", "path2", "none"]

PASS_THRESHOLD = 80

# Repositories and publishing platforms host many independent authors, so
# sharing one of these domains says nothing about shared origin.
HOSTING_DOMAINS: frozenset[str] = frozenset(
    {
        "arxiv.org",
        "doi.org",
        "ncbi.nlm.nih.gov",
        "pubmed.ncbi.nlm.nih.gov",
        "openalex.org",
        "semanticscholar.org",
        "researchgate.net",
        "ssrn.com",
        "osf.io",
        "zenodo.org",
        "github.com",
        "medium.com",
        "substack.com",
        "youtube.com",
        "wordpress.com",
        "blogspot.com",
        # large publishers: unrelated papers share the domain
        "sciencedirect.com",
        "springer.com",
        "link.springer.com",
        "nature.com",
        "wiley.com",
        "onlinelibrary.wiley.com",
        "tandfonline.com",
        "mdpi.com",
        "frontiersin.org",
        "plos.org",
        "journals.plos.org",
        "sagepub.com",
        "oup.com",
        "academic.oup.com",
        "cambridge.org",
        "ieeexplore.ieee.org",
        "dl.acm.org",
        "aclanthology.org",
        "biorxiv.org",
        "medrxiv.org",
        "jstor.org",
        "pmc.ncbi.nlm.nih.gov",
        "science.org",
        "pnas.org",
        "nejm.org",
        "thelancet.com",
        "cell.com",
        "bmj.com",
        "jamanetwork.com",
        "pubs.acs.org",
        "annualreviews.org",
        "royalsocietypublishing.org",
        "iop.org",
        "aps.org",
        "acpjournals.org",
        "dbpia.co.kr",
        "kci.go.kr",
        "riss.kr",
        "scienceon.kisti.re.kr",
        "koreascience.kr",
        "korea.kr",
        "europepmc.org",
        "core.ac.uk",
        "hal.science",
        "scholar.archive.org",
        # news aggregators: many outlets under one domain
        "n.news.naver.com",
        "news.naver.com",
        "v.daum.net",
        "news.yahoo.com",
        "msn.com",
        "news.google.com",
        # venues and blogs with many unrelated authors
        "proceedings.neurips.cc",
        "papers.nips.cc",
        "openreview.net",
        "proceedings.mlr.press",
        "jmlr.org",
        "nber.org",
        "blog.naver.com",
        "brunch.co.kr",
        "velog.io",
        "tistory.com",
        "gov.uk",
    }
)


@dataclass(frozen=True)
class Evidence:
    url: str
    grade: Grade
    stance: Stance = "support"
    operator: str = ""
    origin: str = ""
    primary: bool = False
    conflict_of_interest: bool = False
    quote: str = ""
    checked: str = ""  # set by the source check: "quote found" or "unchecked: ..."
    # Keys the source check found for the paper behind this source or behind its origin
    # (OpenAlex ids, DOIs, "title:<normalised title>"), space separated. A preprint and its
    # published version share the title key even when OpenAlex lists them as two works.
    work: str = ""


@dataclass(frozen=True)
class Group:
    best_grade: Grade
    primary: bool
    urls: tuple[str, ...]
    # False only when every member was opened and its quote could not be confirmed
    # (checked starts with "unchecked"). "" means no source check ran, e.g. in tests.
    confirmed: bool = True
    # False when no member names both an operator and an origin: such a group cannot be
    # shown to be independent and only counts when nothing better exists (40–59%).
    shown: bool = True


@dataclass(frozen=True)
class Verdict:
    confidence: int
    band: tuple[int, int]
    path: Path
    label: str
    independent_support: int
    independent_contra: int
    best_grade: Grade = "C"  # best grade among the groups counted on the side the verdict follows

    @property
    def passed(self) -> bool:
        return self.confidence >= PASS_THRESHOLD


def _norm(value: str) -> str:
    return " ".join(value.lower().split())


def _domain(url: str) -> str:
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""
    return host.removeprefix("www.")


_DOI = re.compile(r"(10\.\d{4,9}/[\x21-\x7e]+?)(?=$|[\s?#&\"'<>]|[^\x00-\x7f])", re.IGNORECASE)
_DOI_TAIL = re.compile(
    r"(?:[./](?:full|abstract|pdf|epdf|html|fulltext|full-text|supplemental|references)(?:\.pdf)?)+$",
    re.IGNORECASE,
)
# bioRxiv/medRxiv append a version (…123456v2) to the URL but not to the DOI. Other publishers can have a
# real DOI that ends in v<digits>, so the version is only stripped for 10.1101.
_BIORXIV_VERSION = re.compile(r"^(10\.1101/[\d.]*\d)v\d+$", re.IGNORECASE)
_ARXIV = re.compile(r"arxiv(?:\.org/(?:abs|pdf|html)/|[:\s]\s*(?:id\s*)?)(\d{4}\.\d{4,5})", re.IGNORECASE)
_BARE_ARXIV = re.compile(r"^\s*(\d{4}\.\d{4,5})(?:v\d+)?\s*$")


_NATURE = re.compile(r"nature\.com/articles/([a-z0-9.\-]+)", re.IGNORECASE)


def doi_of(url: str) -> str | None:
    """DOI in a URL, without version/view suffixes that publishers append (v3, .full, /abstract, /pdf)."""
    nature = _NATURE.search(url)
    if nature and not _DOI.search(unquote(url)):
        return "10.1038/" + _DOI_TAIL.sub("", nature.group(1)).rstrip(".")
    match = _DOI.search(unquote(url))
    if not match:
        return None
    doi = match.group(1).rstrip(".,;)/")
    trimmed = _DOI_TAIL.sub("", doi).rstrip(".,;)/")
    trimmed = _BIORXIV_VERSION.sub(r"\1", trimmed)
    return trimmed if "/" in trimmed and trimmed.split("/", 1)[1] else doi


def work_id(url: str) -> str | None:
    """A key for 'the same paper' across mirrors: DOI (arXiv ids mapped to their DataCite DOI).

    Accepts URLs and origin text such as 'arXiv:2306.05685', 'arXiv 2306.05685' or a bare id."""
    arxiv = _ARXIV.search(url) or _BARE_ARXIV.match(url)
    if arxiv:
        return "10.48550/arxiv." + arxiv.group(1)
    doi = doi_of(url)
    return doi.lower() if doi else None


# Hosting platforms and publishers are not the operator of a paper; its authors are.
PLATFORM_OPERATORS: tuple[str, ...] = (
    "pubmed", "ncbi", "nlm", "arxiv", "biorxiv", "medrxiv", "semantic scholar", "google scholar", "openalex",
    "researchgate", "ssrn", "zenodo", "osf", "elsevier", "sciencedirect", "springer", "wiley", "nature",
    "taylor", "sage", "mdpi", "frontiers", "plos", "ieee", "acm", "oxford university press",
    "cambridge university press", "jstor", "crossref", "doi.org", "acl anthology",
)


_PLACEHOLDERS = frozenset({"", "?", "unknown", "n/a", "na", "none", "null", "-", "없음", "모름", "미상", "알 수 없음"})


def _field(value: str) -> str:
    """Operator/origin text, with placeholders the model writes for 'don't know' treated as missing."""
    text = value.strip()
    return "" if _norm(text) in _PLACEHOLDERS else text


def _is_platform(operator: str) -> bool:
    """Exact platform names only ("PubMed", "PubMed (NLM)"); "IEEE Standards Association" is a real operator."""
    op = _norm(operator)
    return any(op == p or op.startswith((p + " (", p + "(")) for p in PLATFORM_OPERATORS)


def independent_groups(evidence: list[Evidence]) -> list[Group]:
    """Union-find over sources that share an operator or an origin."""
    items = [e for e in evidence if e.grade != "C" and not e.conflict_of_interest]
    parent = list(range(len(items)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    seen: dict[str, int] = {}
    for idx, item in enumerate(items):
        op, origin = _field(item.operator), _field(item.origin)
        keys = []
        if origin:
            keys.append("origin:" + _norm(origin))
            # An origin that names a paper (DOI, arXiv id, doi.org link) is that paper, whatever
            # the wording: news stories about a study share its origin with the study itself.
            origin_work = work_id(origin)
            if origin_work:
                keys.append("work:" + origin_work)
        url_work = work_id(item.url) if item.url else None
        paper = bool(item.work or url_work)
        if op and not _is_platform(op):
            keys.append("op:" + _norm(op))
        # Two pages of the same site share an operator, except on hosting platforms where
        # unrelated authors publish side by side, and except for papers (the work key covers them).
        domain = _domain(item.url) if item.url else ""
        if domain and not paper and not any(domain == h or domain.endswith("." + h) for h in HOSTING_DOMAINS):
            keys.append("domain:" + domain)
        # One paper reached through several mirrors (doi.org, publisher, PubMed, arXiv, ...) is one
        # source: the OpenAlex work id and the DOI/arXiv id from the URL both tie them together.
        for wid in (*item.work.split(), url_work):
            if wid:
                keys.append("work:" + wid.lower())
        if item.url:
            keys.append("url:" + item.url.strip().rstrip("/").lower())  # the same page listed twice
        for key in keys:
            if key in seen:
                union(seen[key], idx)
            else:
                seen[key] = idx

    groups: dict[int, list[Evidence]] = {}
    for idx, item in enumerate(items):
        groups.setdefault(find(idx), []).append(item)
    return [
        Group(
            best_grade="A" if any(e.grade == "A" for e in members) else "B",
            primary=any(e.primary and e.grade == "A" for e in members),
            urls=tuple(e.url for e in members),
            confirmed=not all(e.checked.startswith("unchecked") for e in members),
            shown=any(_shown(e) for e in members),
        )
        for members in groups.values()
    ]


def _shown(e: Evidence) -> bool:
    op = _field(e.operator)
    return bool(op and _field(e.origin) and not _is_platform(op))


def _countable(groups: list[Group]) -> list[Group]:
    """Groups that may be counted as independent.

    Groups shown to be independent all count, except that those whose quotes could not
    be confirmed count at most once. Groups whose independence cannot be shown count
    (as one) only when nothing else exists, which lands in 40–59%.
    """
    shown = [g for g in groups if g.shown]
    if not shown:
        return groups[:1]
    confirmed = [g for g in shown if g.confirmed]
    unconfirmed = sorted((g for g in shown if not g.confirmed), key=lambda g: (not g.primary, g.best_grade))
    return confirmed + unconfirmed[:1]


def assess(evidence: list[Evidence], *, counter_searched: bool = True, rebuttal_found: bool = False) -> Verdict:
    """Apply the rubric, then the party's-own-document floor (see :func:`_own_document_floor`)."""
    result = _assess(evidence, counter_searched=counter_searched, rebuttal_found=rebuttal_found)
    return _own_document_floor(result, evidence, rebuttal_found)


def _own_document_floor(result: Verdict, evidence: list[Evidence], rebuttal_found: bool) -> Verdict:
    """A party's own primary document (공시, 보도자료) shows what the party said or filed. The code
    cannot tell an "X announced Y" claim from Y itself, so it never makes a claim pass; it only lifts
    a weaker result to 70 (조건부) when nothing contradicts it."""
    own = [e for e in evidence if e.stance == "support" and e.conflict_of_interest and e.primary
           and e.grade == "A" and not e.checked.startswith("unchecked")]
    if not own or rebuttal_found or result.independent_contra or result.confidence >= 70:
        return result
    return replace(result, confidence=70, band=(60, 79), path="none",
                   label="조건부(당사자 1차 자료: 발표·공시 사실만 확인)",
                   best_grade="A" if result.best_grade != "A" else result.best_grade)


def _assess(evidence: list[Evidence], *, counter_searched: bool = True, rebuttal_found: bool = False) -> Verdict:
    """Apply the rubric.

    Two guards keep a pass honest: path 2 needs a counter-evidence search, and a
    pass needs at least one deciding source whose quote the code confirmed.
    Either failing caps the claim at 70 (조건부) instead of 80+.
    """
    support_all = independent_groups([e for e in evidence if e.stance == "support"])
    support = _countable(support_all)
    contra = _countable(independent_groups([e for e in evidence if e.stance == "contradict"]))
    sup, con = len(support), len(contra)
    a_sup = sum(1 for g in support if g.best_grade == "A")
    primary_groups = [g for g in support if g.primary]

    def verdict(conf: int, band: tuple[int, int], path: Path, label: str) -> Verdict:
        side = contra if label == "반박됨" else support
        best: Grade = "A" if any(g.best_grade == "A" for g in side) else ("B" if side else "C")
        return Verdict(conf, band, path, label, sup, con, best)

    def capped(reason: str) -> Verdict:
        return verdict(70, (60, 79), "none", f"조건부({reason})")

    if con > sup:  # 반대 근거 우세
        return verdict(25, (0, 39), "none", "반박됨")
    if con:  # 지지와 반대가 함께 있음(동수 포함)
        return verdict(50, (40, 59), "none", "미확정(출처 간 결론 혼재)")
    if sup == 0:
        return verdict(20, (0, 39), "none", "근거 부족")
    a_groups = [g for g in support if g.best_grade == "A"]
    if not any(g.shown for g in support):
        return verdict(50, (40, 59), "none", "미확정(독립성 미확인)")
    if a_sup >= 2 or primary_groups:
        deciding = primary_groups if primary_groups and a_sup < 2 else a_groups
        if not any(g.confirmed for g in deciding):
            return capped("인용 미확인")
        return verdict(95, (90, 99), "path1", "사실")
    if sup >= 3:
        if rebuttal_found:  # path 2 needs a clean counter search; a credible rebuttal (A/B, no COI) blocks it
            return verdict(50, (40, 59), "none", "미확정(반대 근거 검색에서 반박 발견)")
        if not counter_searched:
            return capped("반대 근거 검색 없음")
        if not any(g.confirmed for g in support):
            return capped("인용 미확인")
        if a_sup >= 1:
            return verdict(92, (90, 99), "path2", "사실")
        return verdict(85, (80, 89), "path2", "사실(웹 교차 확인)")
    if a_sup == 1 or sup == 2:
        return verdict(70, (60, 79), "none", "조건부")
    return verdict(50, (40, 59), "none", "미확정")
