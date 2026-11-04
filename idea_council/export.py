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
from .prompts import LINK_TYPES, ROLES

PATH_LABEL = {"path1": "경로1 권위 출처", "path2": "경로2 독립 교차 확인", "none": "미통과"}
GRADE_LABEL = {"A": "A 1차·동료심사·공식통계", "B": "B 신뢰 웹 자료", "C": "C 참고만"}


def _role(role: str) -> str:
    return ROLES.get(role, (role, ""))[0]


def _cell(text: object) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def to_markdown(result: CouncilResult) -> str:
    lines = [f"# 카운슬: {result.question}", "", result.synthesis or "(의장 종합 실패)", ""]
    if result.checks:
        lines += ["## 사실 검증", "", "| 주장 | 판정 | 확신도 | 경로 | 독립 출처(지지/반대) |", "| --- | --- | --- | --- | --- |"]
        for c in result.checks:
            v = c.verdict
            lines.append(
                f"| {_cell(c.claim)} | {_cell(v.label)} | {v.confidence}% | {PATH_LABEL[v.path]} | "
                f"{v.independent_support}/{v.independent_contra} |"
            )
        lines.append("")
    lines += ["## 상호 검토 순위", ""]
    for i, s in enumerate(result.standings, start=1):
        rank = f"평균 {s.average_rank}위" if s.average_rank is not None else "순위 없음"
        lines.append(f"{i}. {_role(s.role)} ({s.member}): {rank}, {s.rankings_count}회 평가")
    lines += ["", "## 위원 답변", ""]
    for a in result.answers:
        lines += [f"### {_role(a.role)} ({a.member})", "", a.text if a.ok else f"(실패: {a.error})", ""]
    return "\n".join(lines)


def to_json(result: CouncilResult) -> str:
    return json.dumps(dataclasses.asdict(result), ensure_ascii=False, indent=2, default=str)


def _judgement(check: Any) -> str:
    v = check.verdict
    if check.error:
        return "확인 불가"
    if v.passed:
        return "지지"
    if v.label == "반박됨":
        return "반박"
    if "혼재" in v.label:
        return "혼재"
    if v.label.startswith("조건부"):
        return "검증 중"  # 60–79%: the rule says keep verifying
    return "확인 불가"


def notion_claim_rows(result: CouncilResult, idea_page_url: str | None = None) -> list[dict[str, Any]]:
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    key = {" ".join(k.lower().split()) for k in result.key_claims}
    rows = []
    for c in result.checks:
        v = c.verdict
        support = [e for e in c.evidence if e.stance == "support"]
        contra = [e for e in c.evidence if e.stance == "contradict"]
        counted = [e for e in support if not e.conflict_of_interest]
        best = min((e.grade for e in counted), default="C")  # best grade among sources that count
        row: dict[str, Any] = {
            "주장": c.claim,
            "판정": _judgement(c),
            "확신도(%)": v.confidence / 100,
            "중요도": "핵심" if " ".join(c.claim.lower().split()) in key else "보조",
            "검증 경로": PATH_LABEL[v.path],
            "출처 등급": GRADE_LABEL[best],
            "독립 출처 수": v.independent_support,
            "근거 요약": "; ".join(e.quote for e in support if e.quote)[:1900],
            "반대 근거": "; ".join(f"{e.url} {e.quote}".strip() for e in contra)[:1900],
            "독립성 메모": "; ".join(
                f"{e.operator or '?'} ← {e.origin or '?'} [{e.checked or '미확인'}]" for e in c.evidence
            )[:1900],
            "date:검증일:start": today,
            "date:검증일:is_datetime": 0,
        }
        for i, e in enumerate((support + contra)[:3], start=1):
            row[f"출처 {i}"] = e.url
        if idea_page_url:
            row["아이디어"] = json.dumps([idea_page_url])
        rows.append(row)
    return rows


def _link_types(value: Any) -> list[str]:
    values = value if isinstance(value, list) else [value]
    return [t for t in (str(v).strip() for v in values if v) if t in LINK_TYPES]


def notion_link_rows(result: CouncilResult, idea_page_url: str | None = None) -> list[dict[str, Any]]:
    """Rows for the 연결 DB. The other idea is free text from the chairman, so it goes into
    the title; the relation only holds the idea page given on the command line."""
    rows = []
    for link in result.links:
        strength = link.get("strength")
        other = str(link.get("idea", "")).strip()
        insight = str(link.get("insight", "")).strip()
        row: dict[str, Any] = {
            "통찰 한 줄": (f"[{other}] {insight}" if other and insight else insight or other)[:1900],
            "연결 유형": json.dumps(_link_types(link.get("type")), ensure_ascii=False),
            "발견 경로": "카운슬",
            "상태": "후보",
        }
        question = str(link.get("question", "")).strip()
        if question:
            row["다음 질문"] = question[:1900]
        if isinstance(strength, (int, float)) and not isinstance(strength, bool):
            row["강도(%)"] = max(0.0, min(1.0, float(strength) / 100))
        if idea_page_url:
            row["아이디어"] = json.dumps([idea_page_url])
        rows.append(row)
    return rows
