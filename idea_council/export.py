"""Turn a council result into a markdown report, JSON, and Notion row payloads.

Property names match the 아이디어 허브 databases in Notion
(주장·근거, 연결). Writing the rows is left to the caller (Notion MCP or API).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
from typing import Any

from .council import CouncilResult
from .prompts import ROLES

PATH_LABEL = {"path1": "경로1 권위 출처", "path2": "경로2 독립 교차 확인", "none": "미통과"}
GRADE_LABEL = {"A": "A 1차·동료심사·공식통계", "B": "B 신뢰 웹 자료", "C": "C 참고만"}


def _role(role: str) -> str:
    return ROLES.get(role, (role, ""))[0]


def to_markdown(result: CouncilResult) -> str:
    lines = [f"# 카운슬: {result.question}", "", result.synthesis or "(의장 종합 실패)", ""]
    if result.checks:
        lines += ["## 사실 검증", "", "| 주장 | 판정 | 확신도 | 경로 | 독립 출처(지지/반대) |", "| --- | --- | --- | --- | --- |"]
        for c in result.checks:
            v = c.verdict
            lines.append(
                f"| {c.claim} | {v.label} | {v.confidence}% | {PATH_LABEL[v.path]} | "
                f"{v.independent_support}/{v.independent_contra} |"
            )
        lines.append("")
    lines += ["## 상호 검토 순위", ""]
    for i, s in enumerate(result.standings, start=1):
        lines.append(f"{i}. {_role(s.role)} ({s.member}): 평균 {s.average_rank}위, {s.rankings_count}회 평가")
    lines += ["", "## 위원 답변", ""]
    for a in result.answers:
        lines += [f"### {_role(a.role)} ({a.member})", "", a.text if a.ok else f"(실패: {a.error})", ""]
    return "\n".join(lines)


def to_json(result: CouncilResult) -> str:
    return json.dumps(dataclasses.asdict(result), ensure_ascii=False, indent=2, default=str)


def notion_claim_rows(result: CouncilResult, idea_page_url: str | None = None) -> list[dict[str, Any]]:
    today = dt.date.today().isoformat()
    rows = []
    for c in result.checks:
        v = c.verdict
        support = [e for e in c.evidence if e.stance == "support"]
        contra = [e for e in c.evidence if e.stance == "contradict"]
        best = min((e.grade for e in c.evidence), default="C")
        row: dict[str, Any] = {
            "주장": c.claim,
            "판정": "지지" if v.passed else ("반박" if v.label == "반박됨" else "혼재" if "혼재" in v.label else "확인 불가"),
            "확신도(%)": v.confidence / 100,
            "중요도": "핵심",
            "검증 경로": PATH_LABEL[v.path],
            "출처 등급": GRADE_LABEL[best],
            "독립 출처 수": v.independent_support,
            "근거 요약": "; ".join(e.quote for e in support if e.quote)[:1900],
            "반대 근거": "; ".join(f"{e.url} {e.quote}".strip() for e in contra)[:1900],
            "독립성 메모": "; ".join(f"{e.operator or '?'} ← {e.origin or '?'}" for e in c.evidence)[:1900],
            "date:검증일:start": today,
            "date:검증일:is_datetime": 0,
        }
        for i, e in enumerate(c.evidence[:3], start=1):
            row[f"출처 {i}"] = e.url
        if idea_page_url:
            row["아이디어"] = json.dumps([idea_page_url])
        rows.append(row)
    return rows


def notion_link_rows(result: CouncilResult, idea_page_url: str | None = None) -> list[dict[str, Any]]:
    rows = []
    for link in result.links:
        strength = link.get("strength")
        row: dict[str, Any] = {
            "통찰 한 줄": str(link.get("insight", "")).strip() or str(link.get("idea", "")),
            "연결 유형": json.dumps([link["type"]], ensure_ascii=False) if link.get("type") else "[]",
            "발견 경로": "카운슬",
            "다음 질문": str(link.get("idea", "")),
        }
        if isinstance(strength, (int, float)):
            row["강도(%)"] = max(0.0, min(1.0, float(strength) / 100))
        if idea_page_url:
            row["아이디어"] = json.dumps([idea_page_url])
        rows.append(row)
    return rows
