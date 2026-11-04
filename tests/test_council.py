import asyncio
import json
import random

from idea_council.config import CouncilConfig
from idea_council.council import Council
from idea_council.export import notion_claim_rows, notion_link_rows, to_markdown
from idea_council.llm import Completion
from idea_council.verify import Verifier

MEMBERS = ("m/alpha", "m/beta", "m/gamma", "m/delta")


class FakeLLM:
    """Scripted model: answers by stage, records every call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[dict], bool]] = []

    async def complete(self, model, messages, *, web_search=False):
        self.calls.append((model, messages, web_search))
        system = messages[0]["content"]
        if "위원이며" in system:
            claims = {"claims": [{"text": f"{model}의 핵심 주장", "key": True}]}
            return Completion(model, f"ANSWER-FROM-{model}\n```json\n{json.dumps(claims, ensure_ascii=False)}\n```")
        if "검토자" in system:
            body = messages[1]["content"]
            labels = [line.split()[-1] for line in body.splitlines() if line.startswith("### 응답 ")]
            data = {
                "reviews": [
                    {"label": l, "accuracy": 3, "evidence": 3, "novelty": 3, "feasibility": 3,
                     "critique": "ok", "suspect_claims": ["의심 주장"]}
                    for l in labels
                ],
                "ranking": labels,
            }
            return Completion(model, "```json\n" + json.dumps(data, ensure_ascii=False) + "\n```")
        if "사실 검증자" in system:
            data = {"evidence": [
                {"url": f"https://site{i}.example/a", "grade": "B", "stance": "support",
                 "operator": f"운영자{i}", "origin": f"원출처{i}"} for i in range(3)
            ], "counter_search": "반대 검색어 X: 반박 없음"}
            return Completion(model, "```json\n" + json.dumps(data, ensure_ascii=False) + "\n```")
        if "의장" in system:
            extra = {"links": [{"idea": "다른 아이디어", "insight": "같은 문제", "type": "같은 문제", "strength": 65}],
                     "next_questions": ["다음은?"]}
            return Completion(model, "### 결론\n좋음\n```json\n" + json.dumps(extra, ensure_ascii=False) + "\n```")
        raise AssertionError("unexpected prompt")


def run_council():
    llm = FakeLLM()
    config = CouncilConfig(members=MEMBERS, chairman="m/chair", verifier="m/verifier", check_source_urls=False)
    council = Council(llm, config, verifier=Verifier(llm, "m/verifier", check_urls=False), rng=random.Random(7))
    result = asyncio.run(council.run("이 아이디어를 전개해 줘", "맥락 카드", ["다른 아이디어"]))
    return llm, result


def test_reviewers_never_see_their_own_answer():
    llm, _ = run_council()
    reviews = [(m, msgs) for m, msgs, _ in llm.calls if "검토자" in msgs[0]["content"]]
    assert len(reviews) == len(MEMBERS)
    for model, msgs in reviews:
        body = msgs[1]["content"]
        assert f"ANSWER-FROM-{model}" not in body
        assert sum(f"ANSWER-FROM-{m}" in body for m in MEMBERS) == len(MEMBERS) - 1


def test_roles_are_all_assigned_and_ranks_aggregate():
    _, result = run_council()
    assert sorted(a.role for a in result.answers) == sorted(CouncilConfig().roles)
    assert all(s.rankings_count == len(MEMBERS) - 1 for s in result.standings)
    assert [s.average_rank for s in result.standings] == sorted(s.average_rank for s in result.standings)


def test_gate_verifies_suspect_then_key_claims_with_web_search():
    llm, result = run_council()
    verify_calls = [c for c in llm.calls if "사실 검증자" in c[1][0]["content"]]
    assert verify_calls and all(web for _, _, web in verify_calls)
    assert result.checks[0].claim == "의심 주장"  # flagged by every reviewer, so first
    assert all(c.verdict.path == "path2" and c.verdict.passed for c in result.checks)


def test_synthesis_links_and_exports():
    _, result = run_council()
    assert result.synthesis.startswith("### 결론")
    assert result.links[0]["strength"] == 65 and result.next_questions == ["다음은?"]
    rows = notion_claim_rows(result, "https://notion.so/idea")
    assert rows[0]["검증 경로"] == "경로2 독립 교차 확인" and rows[0]["확신도(%)"] == 0.85
    assert notion_link_rows(result)[0]["강도(%)"] == 0.65
    assert "## 사실 검증" in to_markdown(result)


def test_link_rows_only_use_db_options():
    _, result = run_council()
    result.links = [{"idea": "B", "insight": "같은 병목", "type": ["같은 문제", "아무거나"], "strength": 70, "question": "합치면?"}]
    row = notion_link_rows(result)[0]
    assert json.loads(row["연결 유형"]) == ["같은 문제"]
    assert row["통찰 한 줄"] == "[B] 같은 병목" and row["다음 질문"] == "합치면?"


def test_duplicate_members_are_rejected():
    import pytest
    with pytest.raises(ValueError):
        CouncilConfig(members=("m/a", "m/a"))


def test_gate_survives_a_crashing_verifier_and_exports_judgement():
    class Boom:
        async def verify(self, claim, context=""):
            raise RuntimeError("boom")
    llm = FakeLLM()
    config = CouncilConfig(members=MEMBERS, chairman="m/chair", verifier="m/verifier", check_source_urls=False)
    council = Council(llm, config, verifier=Boom(), rng=random.Random(1))
    result = asyncio.run(council.run("q", "c", []))
    assert result.checks and all(c.error for c in result.checks)
    rows = notion_claim_rows(result)
    assert {r["판정"] for r in rows} == {"확인 불가"}
    assert {r["중요도"] for r in rows} == {"핵심", "보조"}  # suspect claim is 보조, members' key claims 핵심


def test_key_claims_are_not_crowded_out_and_keep_their_flag():
    from idea_council.council import MemberAnswer, ReviewRecord
    from idea_council.parsing import Review
    config = CouncilConfig(members=MEMBERS, chairman="c", verifier="v", max_claims_to_verify=4)
    council = Council(FakeLLM(), config, verifier=Verifier(FakeLLM(), "v", check_urls=False))
    answers = [MemberAnswer("m/a", "critic", "t", [{"text": "합계출산율은 0.72명이다.", "key": True},
                                                   {"text": "핵심 주장 둘", "key": "true"}])]
    suspects = {"A": ["서울 인구는 천만이다", "GPT-5는 2024년 출시", "화성에 물이 있다", "커피는 수명을 늘린다",
                   "비트코인은 2009년 시작", "한국 GDP는 세계 10위", "달은 매년 멀어진다", "꿀은 상하지 않는다"] + ["합계출산율은 0.72명이다"]}
    reviews = [ReviewRecord("m/b", {"A": "m/a"}, Review(ranking=["A"], suspect_claims=suspects))]
    picked = council.select_claims(answers, reviews)
    assert picked[0] == ("합계출산율은 0.72명이다", True)  # flagged and key: first, flagged as key
    assert ("핵심 주장 둘", True) in picked and len(picked) == 4


def test_link_strength_accepts_fraction_and_string():
    _, result = run_council()
    result.links = [{"idea": "B", "insight": "x", "type": "유추", "strength": 0.6},
                    {"idea": "C", "insight": "y", "type": "유추", "strength": "70"}]
    rows = notion_link_rows(result)
    assert [r["강도(%)"] for r in rows] == [0.6, 0.7]


def test_claims_differing_in_number_or_direction_are_not_merged():
    from idea_council.council import merge_claims
    a = "2023년 국내 전기차 판매량은 16만 대로 전년 대비 1% 감소했다"
    b = "2023년 국내 전기차 판매량은 16만 대로 전년 대비 1% 증가했다"
    assert len(merge_claims([(a, False, 1), (b, False, 1)])) == 2
    assert len(merge_claims([("Global EV sales rose 35% in 2023", False, 1),
                             ("Global EV sales rose 15% in 2023", False, 1)])) == 2
    assert len(merge_claims([("합계출산율은 0.72명이다.", True, 0), ("합계출산율은 0.72명이다", False, 1)])) == 1


def test_max_claims_must_be_positive():
    import pytest
    with pytest.raises(ValueError):
        CouncilConfig(members=MEMBERS, max_claims_to_verify=0)


def test_claims_with_different_names_are_not_merged():
    from idea_council.council import merge_claims
    pairs = [("Germany has the largest economy in the EU by GDP", "France has the largest economy in the EU by GDP"),
             ("서울에서 아파트 값은 강남구가 가장 비싸다", "서울에서 아파트 값은 서초구가 가장 비싸다")]
    for a, b in pairs:
        assert len(merge_claims([(a, False, 1), (b, False, 1)])) == 2
