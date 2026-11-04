"""Prompts for every council stage (Korean output, English scaffolding kept minimal)."""

from __future__ import annotations

ROLES: dict[str, tuple[str, str]] = {
    "expander": (
        "확장자",
        ("아이디어의 가능성을 가장 넓게 펼치십시오. 응용처, 확장 방향, 의외의 전환을 3~5개 제시하고 "
        "각각이 왜 유망한지 한두 문장으로 설명하십시오."),
    ),
    "critic": (
        "비판자",
        ("가장 강한 반론을 세우십시오. 숨은 가정, 반례, 실패 시나리오, 이미 존재하는 선행 사례나 "
        "경쟁 서비스를 찾고, 아이디어가 살아남으려면 무엇이 사실이어야 하는지 밝히십시오."),
    ),
    "builder": (
        "실행가",
        ("가장 싸고 빠른 검증 실험과 실행 경로를 설계하십시오. 필요한 자원, 첫 1~2주 할 일, "
        "성공과 실패를 가르는 판정 기준을 숫자로 제시하십시오."),
    ),
    "connector": (
        "연결자",
        ("이 아이디어와 사용자의 다른 아이디어, 또는 다른 분야 사이의 구조적 유사성을 찾으십시오. "
        "표면적 비슷함보다 같은 메커니즘, 같은 문제, 결합하면 생기는 새 아이디어를 우선하십시오."),
    ),
}

CLAIMS_FORMAT = """\
답의 맨 끝에 아래 형식의 JSON 블록을 하나 붙이십시오. 답의 결론이 기대는 사실은 "key": true 로 표시하십시오.
```json
{"claims": [{"text": "검증 가능한 한 문장 주장", "key": true, "sources": ["https://..."]}]}
```"""


def stage1_messages(
    question: str, context: str, role: str, other_ideas: list[str]
) -> list[dict[str, str]]:
    name, instruction = ROLES.get(role, (role, "질문에 최선을 다해 답하십시오."))
    ideas = "\n".join(f"- {idea}" for idea in other_ideas) or "(없음)"
    system = (
        f"당신은 아이디어 카운슬의 위원이며 역할은 '{name}'입니다. {instruction}\n"
        "다른 위원의 답은 볼 수 없습니다. 확실하지 않은 사실은 단정하지 말고 그렇다고 쓰십시오. "
        "출처를 알면 URL을 적되, 지어낸 URL은 절대 쓰지 마십시오. 한국어로 답하십시오.\n\n"
        + CLAIMS_FORMAT
    )
    user = (
        f"## 맥락(이 아이디어 스레드의 지금까지)\n{context or '(없음)'}\n\n"
        f"## 사용자의 다른 아이디어\n{ideas}\n\n"
        f"## 질문\n{question}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def stage2_messages(question: str, labeled: list[tuple[str, str]]) -> list[dict[str, str]]:
    blocks = "\n\n".join(f"### 응답 {label}\n{text}" for label, text in labeled)
    labels = ", ".join(label for label, _ in labeled)
    system = (
        "당신은 아이디어 카운슬의 검토자입니다. 아래 응답들은 작성자를 가린 익명 응답입니다. "
        "응답의 길이나 말투가 아니라 내용으로 판단하십시오. 각 응답을 먼저 따로 비판한 뒤 순위를 매기십시오."
    )
    user = f"""## 질문
{question}

## 익명 응답 ({labels})
{blocks}

## 할 일
1. 응답마다 정확성, 근거, 참신성, 실행 가능성을 1~5점으로 매기고 핵심 비판을 두세 문장으로 쓰십시오.
2. 사실 여부가 의심되거나 근거 없이 단정한 주장을 응답별로 그대로 옮겨 적으십시오.
3. 전체 순위를 매기십시오(가장 좋은 것이 처음).

답의 맨 끝에 아래 형식의 JSON 블록을 정확히 하나 붙이십시오.
```json
{{"reviews": [{{"label": "A", "accuracy": 4, "evidence": 3, "novelty": 5, "feasibility": 2, "critique": "...", "suspect_claims": ["..."]}}],
 "ranking": ["A", "B"]}}
```
JSON을 쓸 수 없다면 마지막 줄들에 다음 형식으로 쓰십시오.
FINAL RANKING:
1. 응답 A
2. 응답 B"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def verify_messages(claim: str, context: str) -> list[dict[str, str]]:
    system = """당신은 사실 검증자입니다. 웹 검색으로 아래 주장을 검증하십시오.
규칙:
- 검색 결과에 실제로 나온 URL만 쓰십시오. 지어낸 URL이나 기억 속 URL은 쓰지 마십시오.
- quote에는 그 페이지 원문의 구절을 원문 언어 그대로, 한 글자도 바꾸지 말고 20자 이상 옮기십시오. 번역·요약·의역하지 마십시오. 코드가 페이지를 직접 열어 이 구절이 있는지 확인하고, 없으면 그 출처를 버립니다.
- 반대 근거를 일부러 한 번 더 검색하십시오.
- 출처마다 등급을 매기십시오.
  A: 동료심사 논문(특히 체계적 문헌고찰·메타분석), 정부·국제기구 공식 통계, 1차 자료(공식 문서·공시·표준·법령)
  B: 운영 주체가 분명하고 편집·정정 책임이 있는 웹 자료(주요 언론, 전문 매체, 기관·기업 공식 블로그, 업계 보고서, 프리프린트, 실명 전문가 글)
  C: 익명 글, 출처 없는 요약·재가공 글, AI 생성 글, 콘텐츠 농장
- 출처마다 운영 주체(operator)와 원출처(origin)를 적으십시오.
  operator: 그 내용을 책임지는 주체입니다. 기사는 언론사, 논문은 저자 소속 기관이나 연구 그룹(PubMed·arXiv·출판사 같은 게재 플랫폼이 아님), 공식 자료는 발행 기관입니다.
  origin: 그 글이 옮겨 온 원자료(보도자료, 통신사 기사, 원 논문 등)입니다. 원자료가 논문이면 반드시 DOI(예: 10.1038/...)나 arXiv id로 쓰십시오. 논문을 소개한 기사도 그 논문의 DOI를 origin으로 씁니다. 직접 취재라면 그 글 자체를 가리키는 이름을 쓰십시오.
  같은 논문을 여러 사이트에서 찾았다면 하나만 적으십시오.
- 주장이 맞을 때 이익을 보는 당사자의 자료면 conflict_of_interest를 true로 하십시오.
- stance는 "support"(주장을 지지), "contradict"(주장을 반박·부정), "neutral"(관련은 있으나 판단 근거 아님) 중 하나만 쓰십시오. 반대 근거는 반드시 "contradict"로 적으십시오.
- primary는 공식 문서·통계표·법령·공시·표준 그 자체일 때만 true입니다. 연구 논문은 동료심사를 거쳤어도 primary가 아닙니다(등급 A, primary false).
- counter_search에는 실제로 실행한 반대 근거 검색어와 결과를 쓰십시오. 비워 두면 경로 2로 통과할 수 없습니다.
- 반대 근거 검색에서 주장을 반박하는 A·B 등급 자료(이해 상충 없음)를 찾았다면 rebuttal_found를 true로 하고, 그 자료를 evidence에 "contradict"로 넣으십시오. C 등급 반박만 있으면 false로 두고 evidence에만 넣으십시오.
- 확신도는 계산하지 마십시오. 증거만 정확히 적으십시오.

답은 아래 JSON 블록 하나만 쓰십시오.
```json
{"evidence": [{"url": "https://...", "grade": "A", "stance": "support", "operator": "...", "origin": "...", "primary": false, "conflict_of_interest": false, "quote": "원문 구절"},
              {"url": "https://...", "grade": "B", "stance": "contradict", "operator": "...", "origin": "...", "primary": false, "conflict_of_interest": false, "quote": "반대 내용의 원문 구절"}],
 "counter_search": "반대 근거를 어떤 검색어로 찾았고 무엇이 나왔는지 한 줄",
 "rebuttal_found": false,
 "notes": "한계나 조건 한 줄"}
```"""
    user = f"## 주장\n{claim}\n\n## 맥락\n{context or '(없음)'}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


LINK_TYPES: tuple[str, ...] = ("같은 메커니즘", "같은 문제", "같은 근거", "상충", "결합하면 새 아이디어", "유추")


def stage3_messages(
    question: str,
    context: str,
    answers_block: str,
    ranking_block: str,
    review_block: str,
    checks_block: str,
) -> list[dict[str, str]]:
    system = """당신은 아이디어 카운슬의 의장입니다. 위원들의 답, 익명 상호 검토, 사실 검증 결과를 종합해 하나의 답을 쓰십시오.
반드시 지킬 것:
- 검증 결과표의 확신도를 그대로 쓰십시오. 바꾸거나 새로 매기지 마십시오.
- 80% 이상인 주장만 사실로 서술하십시오. 60~79%는 "조건부"로, 그 미만은 "미확정" 또는 "반박됨"으로 표시하십시오.
- 검증하지 않은 핵심 사실은 사실로 쓰지 마십시오.
- 다수 의견에 묻히는 좋은 소수 의견은 따로 남기십시오.
- 한국어로, 간결하게 쓰십시오."""
    link_types = ", ".join(f'"{t}"' for t in LINK_TYPES)
    user = f"""## 질문
{question}

## 맥락
{context or '(없음)'}

## 위원 답변
{answers_block}

## 상호 검토 집계(평균 순위, 낮을수록 좋음)
{ranking_block}

## 검토 요지
{review_block}

## 사실 검증 결과
{checks_block}

## 출력 형식
### 결론
### 합의된 점
### 쟁점과 소수 의견
### 검증된 주장 (주장 | 확신도 | 근거)
### 다음 질문

맨 끝에 아래 JSON 블록을 하나 붙이십시오. links는 다른 아이디어와의 연결 후보입니다(없으면 빈 배열).
type은 {link_types} 중에서만 고르십시오(여러 개면 배열). strength는 0~100 정수이며, 공유 근거나 같은 메커니즘이 확인되지 않은 표면적 유사성은 40 이하로 두십시오.
```json
{{"links": [{{"idea": "연결되는 다른 아이디어", "insight": "왜 연결되는지 한 문장", "type": ["같은 메커니즘"], "strength": 60, "question": "이 연결로 다음에 물을 질문"}}],
 "next_questions": ["..."]}}
```"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
