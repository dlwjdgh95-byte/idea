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

from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlparse

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
    host = urlparse(url).netloc.lower()
    return host.removeprefix("www.")


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
        keys = []
        if item.operator.strip():
            keys.append("op:" + _norm(item.operator))
        if item.origin.strip():
            keys.append("origin:" + _norm(item.origin))
        if not (item.operator.strip() and item.origin.strip()):
            keys.append("unknown")  # independence cannot be shown without both
        # Two pages of the same site share an operator, except on hosting
        # platforms where unrelated authors publish side by side.
        domain = _domain(item.url) if item.url else ""
        if domain and not any(domain == h or domain.endswith("." + h) for h in HOSTING_DOMAINS):
            keys.append("domain:" + domain)
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
        )
        for members in groups.values()
    ]


def assess(evidence: list[Evidence]) -> Verdict:
    support = independent_groups([e for e in evidence if e.stance == "support"])
    contra = independent_groups([e for e in evidence if e.stance == "contradict"])
    sup, con = len(support), len(contra)
    a_sup = sum(1 for g in support if g.best_grade == "A")
    primary = any(g.primary for g in support)

    def verdict(conf: int, band: tuple[int, int], path: Path, label: str) -> Verdict:
        return Verdict(conf, band, path, label, sup, con)

    if con > sup:  # 반대 근거 우세
        return verdict(25, (0, 39), "none", "반박됨")
    if con:  # 지지와 반대가 함께 있음(동수 포함)
        return verdict(50, (40, 59), "none", "미확정(출처 간 결론 혼재)")
    if sup == 0:
        return verdict(20, (0, 39), "none", "근거 부족")
    if a_sup >= 2 or primary:
        return verdict(95, (90, 99), "path1", "사실")
    if sup >= 3:
        if a_sup >= 1:
            return verdict(92, (90, 99), "path2", "사실")
        return verdict(85, (80, 89), "path2", "사실(웹 교차 확인)")
    if a_sup == 1 or sup == 2:
        return verdict(70, (60, 79), "none", "조건부")
    return verdict(50, (40, 59), "none", "미확정")
