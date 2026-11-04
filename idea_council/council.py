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
import string
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from .config import CouncilConfig
from .llm import LLM
from .parsing import (
    SCORE_KEYS,
    Review,
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
    average_rank: float
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
    next_questions: list[str] = field(default_factory=list)
    chairman: str = ""
    chairman_error: str | None = None


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def merge_similar(claims: list[str], threshold: float = 0.85) -> list[str]:
    kept: list[str] = []
    for claim in claims:
        if not claim.strip():
            continue
        if any(SequenceMatcher(None, _norm(claim), _norm(k)).ratio() >= threshold for k in kept):
            continue
        kept.append(claim.strip())
    return kept


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
            average_rank=round(sum(p) / len(p), 2) if p else float("inf"),
            rankings_count=len(p),
            mean_scores={k: round(sum(v) / len(v), 2) for k, v in scores[member].items() if v},
        )
        for member, p in positions.items()
    ]
    return sorted(standings, key=lambda s: s.average_rank)


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
            claims = [c for c in data.get("claims") or [] if isinstance(c, dict) and c.get("text")]
            return MemberAnswer(member, role, strip_json_blocks(completion.text), claims, completion.error)

        return list(await asyncio.gather(*(ask(m, r) for m, r in assignment)))

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
        return list(await asyncio.gather(*(review(r) for r in valid)))

    # Stage 2.5 ---------------------------------------------------------------
    def claims_to_verify(self, answers: list[MemberAnswer], reviews: list[ReviewRecord]) -> list[str]:
        """Suspect claims first (most flagged first), then key claims, near-duplicates merged."""
        flagged: dict[str, int] = {}
        for record in reviews:
            for claims in record.review.suspect_claims.values():
                for claim in claims:
                    flagged[claim] = flagged.get(claim, 0) + 1
        suspect = sorted(flagged, key=lambda c: -flagged[c])
        key = [str(c["text"]) for a in answers for c in a.claims if c.get("key")]
        return merge_similar(suspect + key)[: self.config.max_claims_to_verify]

    async def gate(self, claims: list[str], context: str) -> list[ClaimCheck]:
        return list(await asyncio.gather(*(self.verifier.verify(c, context) for c in claims)))

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
            f"- {c.claim} | {c.verdict.label} | {c.verdict.confidence}% | 경로 {c.verdict.path} | "
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
        checks = await self.gate(self.claims_to_verify(answers, reviews), context) if verify else []
        synthesis, extra, error = await self.stage3(question, context, answers, standings, reviews, checks)
        return CouncilResult(
            question=question,
            answers=answers,
            reviews=reviews,
            standings=standings,
            checks=checks,
            synthesis=synthesis,
            links=[x for x in extra.get("links") or [] if isinstance(x, dict)],
            next_questions=[str(q) for q in extra.get("next_questions") or []],
            chairman=self.config.chairman,
            chairman_error=error,
        )
