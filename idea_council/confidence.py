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
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Group:
    best_grade: Grade
    primary: bool
    urls: tuple[str, ...]
    # False only when every member was opened and its quote could not be confirmed
    # (checked starts with "unchecked"). "" means no source check ran, e.g. in tests.
    confirmed: bool = True


@dataclass(frozen=True)
class Verdict:
    confidence: int
    band: tuple[int, int]
    path: Path
    label: str
    independent_support: int
    independent_contra: int

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


_DOI = re.compile(r"(10\.\d{4,9}/[^\s?#]+)", re.IGNORECASE)
_DOI_TAIL = re.compile(
    r"(?:v\d+)?(?:[./](?:full|abstract|pdf|epdf|html|fulltext|full-text|supplemental|references)(?:\.pdf)?)*$",
    re.IGNORECASE,
)
_ARXIV = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", re.IGNORECASE)


def doi_of(url: str) -> str | None:
    """DOI in a URL, without version/view suffixes that publishers append (v3, .full, /abstract, /pdf)."""
    match = _DOI.search(unquote(url))
    if not match:
        return None
    doi = match.group(1).rstrip(".,;)/")
    trimmed = _DOI_TAIL.sub("", doi).rstrip(".,;)/")
    return trimmed if "/" in trimmed and trimmed.split("/", 1)[1] else doi


def work_id(url: str) -> str | None:
    """A key for 'the same paper' across mirrors: DOI (arXiv ids mapped to their DataCite DOI)."""
    arxiv = _ARXIV.search(url)
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


def _is_platform(operator: str) -> bool:
    op = _norm(operator)
    return any(op == p or op.startswith((p + " ", p + "(")) for p in PLATFORM_OPERATORS)


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
        if not (item.operator.strip() and item.origin.strip()):
            # Independence cannot be shown without both. These sources only join one
            # shared bucket, so they never bridge otherwise independent groups.
            keys = ["unknown"]
        else:
            keys = ["origin:" + _norm(item.origin)]
            if not _is_platform(item.operator):
                keys.append("op:" + _norm(item.operator))
            # Two pages of the same site share an operator, except on hosting
            # platforms where unrelated authors publish side by side.
            domain = _domain(item.url) if item.url else ""
            if domain and not any(domain == h or domain.endswith("." + h) for h in HOSTING_DOMAINS):
                keys.append("domain:" + domain)
        # One paper reached through several mirrors (doi.org, publisher, PubMed, ...) is one source.
        wid = work_id(item.url) if item.url else None
        if wid:
            keys.append("work:" + wid)
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
        )
        for members in groups.values()
    ]


def _cap_unconfirmed(groups: list[Group]) -> list[Group]:
    """Groups whose quotes could not be confirmed count at most once (the strongest of them)."""
    confirmed = [g for g in groups if g.confirmed]
    unconfirmed = sorted((g for g in groups if not g.confirmed), key=lambda g: (not g.primary, g.best_grade))
    return confirmed + unconfirmed[:1]


def assess(evidence: list[Evidence], *, counter_searched: bool = True) -> Verdict:
    """Apply the rubric.

    Two guards keep a pass honest: path 2 needs a counter-evidence search, and a
    pass needs at least one deciding source whose quote the code confirmed.
    Either failing caps the claim at 70 (조건부) instead of 80+.
    """
    support = _cap_unconfirmed(independent_groups([e for e in evidence if e.stance == "support"]))
    contra = independent_groups([e for e in evidence if e.stance == "contradict"])
    sup, con = len(support), len(contra)
    a_sup = sum(1 for g in support if g.best_grade == "A")
    primary_groups = [g for g in support if g.primary]

    def verdict(conf: int, band: tuple[int, int], path: Path, label: str) -> Verdict:
        return Verdict(conf, band, path, label, sup, con)

    def capped(reason: str) -> Verdict:
        return verdict(70, (60, 79), "none", f"조건부({reason})")

    if con > sup:  # 반대 근거 우세
        return verdict(25, (0, 39), "none", "반박됨")
    if con:  # 지지와 반대가 함께 있음(동수 포함)
        return verdict(50, (40, 59), "none", "미확정(출처 간 결론 혼재)")
    if sup == 0:
        return verdict(20, (0, 39), "none", "근거 부족")
    a_groups = [g for g in support if g.best_grade == "A"]
    if a_sup >= 2 or primary_groups:
        deciding = primary_groups if primary_groups and a_sup < 2 else a_groups
        if not any(g.confirmed for g in deciding):
            return capped("인용 미확인")
        return verdict(95, (90, 99), "path1", "사실")
    if sup >= 3:
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
