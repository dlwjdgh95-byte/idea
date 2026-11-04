"""The idea council: independent drafts, anonymous peer review, fact-check gate, chairman synthesis.

Stage structure follows karpathy/llm-council (collect -> rank anonymously ->
chairman). Differences, each aimed at a known LLM-judge weakness:

* every reviewer sees the other answers in its own shuffled order (position bias),
* a reviewer never sees or ranks its own answer (self-preference),
* reviews score separate criteria and list suspect claims instead of only ranking,
* key and suspect claims must pass a fact-check gate before the chairman may state them.
"""

from __future__ import annotations

import asyncio
import random
import re
import string
from dataclasses import dataclass, field
from itertools import zip_longest
from typing import Any

from .confidence import assess
from .config import CouncilConfig
from .llm import LLM
from .parsing import (
    SCORE_KEYS,
    Review,
    as_list,
    last_json_block,
    parse_review,
    strip_json_blocks,
)
from .prompts import ROLES, stage1_messages, stage2_messages, stage3_messages
from .verify import ClaimCheck, Verifier


@dataclass
class MemberAnswer:
    member: str
    role: str
    text: str
    claims: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text.strip())


@dataclass
class ReviewRecord:
    reviewer: str
    label_to_member: dict[str, str]
    review: Review
    raw: str = ""
    error: str | None = None


@dataclass
class Standing:
    member: str
    role: str
    average_rank: float | None  # None when no reviewer ranked this member
    rankings_count: int
    mean_scores: dict[str, float]


@dataclass
class CouncilResult:
    question: str
    answers: list[MemberAnswer]
    reviews: list[ReviewRecord]
    standings: list[Standing]
    checks: list[ClaimCheck]
    synthesis: str
    links: list[dict[str, Any]] = field(default_factory=list)
    key_claims: list[str] = field(default_factory=list)
    next_questions: list[str] = field(default_factory=list)
    chairman: str = ""
    chairman_error: str | None = None


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _is_true(value: Any) -> bool:
    return value is True or str(value).strip().lower() == "true"


_PUNCT = re.compile(r"[\W_]+", re.UNICODE)


_NEGATION = re.compile(r"\b(?:not|no|never|false|untrue|myth|wrong|isn't|doesn't)\b|아니|않|없|거짓|틀리|틀렸|사실이 아", re.IGNORECASE)


def _extra_words(a: str, b: str) -> str:
    """The words one wording adds to the other (in original form, for the negation check)."""
    wa, wb = a.lower().split(), b.lower().split()
    short, long = sorted((wa, wb), key=len)
    return " ".join(w for w in long if w not in short)


def _same_claim(a: str, b: str) -> bool:
    """Only true duplicates: equal apart from case, spacing and punctuation, or one wording contained in
    the other with little added. Fuzzy similarity would join claims that differ in one name or number
    ("Germany" vs "France"), which is exactly the disagreement worth checking."""
    x, y = _PUNCT.sub("", a.lower()), _PUNCT.sub("", b.lower())
    if not x or not y:
        return False
    if x == y:
        return True
    short, long = sorted((x, y), key=len)
    if short not in long or len(short) < 0.85 * len(long):
        return False
    extra = long.replace(short, "", 1)
    if any(ch.isdigit() for ch in extra):  # "50%" vs "50% in 2020" are different claims
        return False
    return not _NEGATION.search(_extra_words(a, b))  # "X" vs "It is not true that X"


def merge_claims(items: list[tuple[str, bool, int]]) -> list[tuple[str, bool, int]]:
    """Merge near-duplicate claims, keeping the first wording, OR-ing the key flag and summing flags."""
    kept: list[list[Any]] = []
    for text, is_key, flags in items:
        if not text.strip():
            continue
        for k in kept:
            if _same_claim(text, k[0]):
                k[1] = k[1] or is_key
                k[2] += flags
                break
        else:
            kept.append([text.strip(), is_key, flags])
    return [(t, bool(k), int(f)) for t, k, f in kept]


def aggregate(reviews: list[ReviewRecord], answers: list[MemberAnswer]) -> list[Standing]:
    """Average rank per member across reviewers (lower is better), plus mean criterion scores."""
    positions: dict[str, list[int]] = {a.member: [] for a in answers}
    scores: dict[str, dict[str, list[float]]] = {a.member: {k: [] for k in SCORE_KEYS} for a in answers}
    for record in reviews:
        for pos, label in enumerate(record.review.ranking, start=1):
            member = record.label_to_member.get(label)
            if member in positions:
                positions[member].append(pos)
        for label, s in record.review.scores.items():
            member = record.label_to_member.get(label)
            if member in scores:
                for key, value in s.items():
                    scores[member][key].append(value)
    roles = {a.member: a.role for a in answers}
    standings = [
        Standing(
            member=member,
            role=roles[member],
            average_rank=round(sum(p) / len(p), 2) if p else None,
            rankings_count=len(p),
            mean_scores={k: round(sum(v) / len(v), 2) for k, v in scores[member].items() if v},
        )
        for member, p in positions.items()
    ]
    return sorted(standings, key=lambda s: (s.average_rank is None, s.average_rank or 0.0))


class Council:
    def __init__(
        self,
        llm: LLM,
        config: CouncilConfig | None = None,
        *,
        verifier: Verifier | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.llm = llm
        self.config = config or CouncilConfig()
        self.rng = rng or random.Random(self.config.seed)
        self.verifier = verifier or Verifier(llm, self.config.verifier, check_urls=self.config.check_source_urls)

    # Stage 1 -----------------------------------------------------------------
    def assign_roles(self) -> list[tuple[str, str]]:
        roles = list(self.config.roles)
        if self.config.rotate_roles:
            self.rng.shuffle(roles)
        return [(m, roles[i % len(roles)]) for i, m in enumerate(self.config.members)]

    async def stage1(self, question: str, context: str, other_ideas: list[str]) -> list[MemberAnswer]:
        assignment = self.assign_roles()

        async def ask(member: str, role: str) -> MemberAnswer:
            completion = await self.llm.complete(member, stage1_messages(question, context, role, other_ideas))
            data = last_json_block(completion.text) or {}
            claims = [c for c in as_list(data.get("claims")) if isinstance(c, dict) and isinstance(c.get("text"), str)]
            return MemberAnswer(member, role, strip_json_blocks(completion.text), claims, completion.error)

        results = await asyncio.gather(*(ask(m, r) for m, r in assignment), return_exceptions=True)
        return [
            r if isinstance(r, MemberAnswer) else MemberAnswer(m, role, "", error=f"{type(r).__name__}: {r}")
            for (m, role), r in zip(assignment, results)
        ]

    # Stage 2 -----------------------------------------------------------------
    async def stage2(self, question: str, answers: list[MemberAnswer]) -> list[ReviewRecord]:
        valid = [a for a in answers if a.ok]

        async def review(reviewer: MemberAnswer) -> ReviewRecord:
            others = [a for a in valid if a.member != reviewer.member]  # never rank yourself
            self.rng.shuffle(others)  # a fresh order per reviewer
            labels = list(string.ascii_uppercase[: len(others)])
            label_to_member = {label: a.member for label, a in zip(labels, others)}
            labeled = [(label, a.text) for label, a in zip(labels, others)]
            completion = await self.llm.complete(reviewer.member, stage2_messages(question, labeled))
            return ReviewRecord(
                reviewer=reviewer.member,
                label_to_member=label_to_member,
                review=parse_review(completion.text, labels),
                raw=completion.text,
                error=completion.error,
            )

        if len(valid) < 2:
            return []
        results = await asyncio.gather(*(review(r) for r in valid), return_exceptions=True)
        return [r for r in results if isinstance(r, ReviewRecord)]  # a crashed reviewer just casts no vote

    # Stage 2.5 ---------------------------------------------------------------
    def claims_to_verify(self, answers: list[MemberAnswer], reviews: list[ReviewRecord]) -> list[str]:
        """Pick claims for the gate: those both key and flagged first, then suspect and key claims
        taken in turn so neither kind crowds out the other. Near-duplicates are merged."""
        return [text for text, _ in self.select_claims(answers, reviews)]

    def select_claims(self, answers: list[MemberAnswer], reviews: list[ReviewRecord]) -> list[tuple[str, bool]]:
        flagged: dict[str, int] = {}
        for record in reviews:
            for claims in record.review.suspect_claims.values():
                for claim in claims:
                    flagged[claim] = flagged.get(claim, 0) + 1
        key_texts = self.key_claims(answers)
        merged = merge_claims([(c, False, n) for c, n in flagged.items()] + [(c, True, 0) for c in key_texts])
        both = sorted((m for m in merged if m[1] and m[2]), key=lambda m: -m[2])
        suspect = sorted((m for m in merged if not m[1]), key=lambda m: -m[2])
        key_only = [m for m in merged if m[1] and not m[2]]
        ordered = list(both)
        for pair in zip_longest(suspect, key_only):
            ordered.extend(m for m in pair if m is not None)
        return [(text, is_key) for text, is_key, _ in ordered[: self.config.max_claims_to_verify]]

    @staticmethod
    def key_claims(answers: list[MemberAnswer]) -> list[str]:
        return [str(c["text"]) for a in answers for c in a.claims if _is_true(c.get("key"))]

    async def gate(self, claims: list[str], context: str) -> list[ClaimCheck]:
        results = await asyncio.gather(*(self.verifier.verify(c, context) for c in claims), return_exceptions=True)
        return [
            r if isinstance(r, ClaimCheck)
            else ClaimCheck(claim=c, verdict=assess([]), error=f"{type(r).__name__}: {r}")
            for c, r in zip(claims, results)
        ]

    # Stage 3 -----------------------------------------------------------------
    async def stage3(
        self,
        question: str,
        context: str,
        answers: list[MemberAnswer],
        standings: list[Standing],
        reviews: list[ReviewRecord],
        checks: list[ClaimCheck],
    ) -> tuple[str, dict[str, Any], str | None]:
        answers_block = "\n\n".join(
            f"### {ROLES.get(a.role, (a.role, ''))[0]} ({a.member})\n{a.text}" for a in answers if a.ok
        )
        ranking_block = "\n".join(
            f"- {s.member} ({ROLES.get(s.role, (s.role, ''))[0]}): 평균 순위 {s.average_rank}, "
            f"검토 {s.rankings_count}회, 점수 {s.mean_scores}"
            for s in standings
        )
        review_lines = []
        for record in reviews:
            for label, critique in record.review.critiques.items():
                target = record.label_to_member.get(label, "?")
                if critique:
                    review_lines.append(f"- {record.reviewer} → {target}: {critique}")
        checks_block = "\n".join(
            f"- {c.claim} | 검증 실패({c.error}) | 사실로 쓰지 말 것"
            if c.error
            else f"- {c.claim} | {c.verdict.label} | {c.verdict.confidence}% | 경로 {c.verdict.path} | "
            f"독립 지지 {c.verdict.independent_support}, 반대 {c.verdict.independent_contra} | "
            + ", ".join(e.url for e in c.evidence[:3])
            for c in checks
        ) or "(검증한 주장 없음)"
        completion = await self.llm.complete(
            self.config.chairman,
            stage3_messages(question, context, answers_block, ranking_block, "\n".join(review_lines), checks_block),
        )
        return strip_json_blocks(completion.text), last_json_block(completion.text) or {}, completion.error

    # Full run ----------------------------------------------------------------
    async def run(
        self,
        question: str,
        context: str = "",
        other_ideas: list[str] | None = None,
        *,
        verify: bool = True,
    ) -> CouncilResult:
        answers = await self.stage1(question, context, list(other_ideas or []))
        reviews = await self.stage2(question, answers)
        standings = aggregate(reviews, [a for a in answers if a.ok])
        selected = self.select_claims(answers, reviews) if verify else []
        checks = await self.gate([t for t, _ in selected], context) if selected else []
        for check, (_, is_key) in zip(checks, selected):
            check.key = is_key
        synthesis, extra, error = await self.stage3(question, context, answers, standings, reviews, checks)
        return CouncilResult(
            question=question,
            answers=answers,
            reviews=reviews,
            standings=standings,
            checks=checks,
            synthesis=synthesis,
            links=[x for x in as_list(extra.get("links")) if isinstance(x, dict)],
            key_claims=self.key_claims(answers),
            next_questions=[q for q in as_list(extra.get("next_questions")) if isinstance(q, str)],
            chairman=self.config.chairman,
            chairman_error=error,
        )
