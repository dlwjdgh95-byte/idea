# idea

생각을 탐색·검증하고 맥락을 쌓아 심화하는 차세대 검색엔진 프로젝트입니다.
설계 문서: [맥락형 검색엔진 프로토타입 설계](https://claude.ai/artifact/C5iRM7y16L1PWJR2wxhEe4)

## idea-council

아이디어를 전개하는 LLM 카운슬입니다. 구조는 Andrej Karpathy의
[llm-council](https://github.com/karpathy/llm-council)(독립 답변 → 익명 상호 순위 → 의장 종합)을 따르고,
LLM 판정자의 알려진 약점을 막는 장치와 사실 검증 게이트를 더했습니다.

| 단계 | 하는 일 | 원본과 다른 점 |
| --- | --- | --- |
| 1. 독립 전개 | 위원 4명이 서로의 답을 보지 않고 답함 | 위원마다 역할(확장자·비판자·실행가·연결자)을 주고, 실행마다 역할을 섞음 |
| 2. 익명 상호 검토 | 다른 위원의 답을 A, B, C로 가려 비판하고 순위를 매김 | 검토자마다 제시 순서를 섞고, 자기 답은 아예 보여 주지 않음. 정확성·근거·참신성·실행 가능성 점수와 의심 주장을 함께 받음 |
| 2.5. 검증 게이트 | 의심받은 주장과 핵심 주장을 웹 검색으로 검증 | 원본에 없음. 확신도는 모델이 아니라 규칙(`confidence.py`)이 계산 |
| 3. 의장 종합 | 결론, 합의, 쟁점·소수 의견, 검증된 주장, 다음 질문, 연결 후보 | 검증 결과표의 확신도를 그대로 쓰고, 80% 미만은 사실로 쓰지 않음 |

### 확신도 규칙 (`idea_council/confidence.py`)

검증자 모델은 증거만 모읍니다. 출처마다 등급(A/B/C), 운영 주체, 원출처, 1차 자료 여부, 이해관계를 적게 하고,
운영 주체나 원출처가 같은 출처는 하나로 묶어 독립 출처 수를 셉니다.

| 확신도 | 조건 |
| --- | --- |
| 90–99% | 독립 A등급 2개 이상 일치, 또는 1차 자료로 확인(경로 1), 또는 경로 2에 A등급 1개 추가 |
| 80–89% | 독립 출처 3곳 이상 일치, 반대 근거 없음(경로 2, 웹 교차 확인) |
| 60–79% | A등급 1개뿐이거나 독립 출처 2곳만 일치(조건부) |
| 40–59% | 독립성이 확인되지 않은 출처뿐이거나 결론이 혼재(미확정) |
| 40% 미만 | 반대 근거 우세 또는 근거 부족 |

80% 이상이 "검증 통과"입니다. 응답하지 않는 URL은 증거에서 뺍니다.

### 실행

```bash
uv venv && uv pip install -e '.[dev]'
export OPENROUTER_API_KEY=...
idea-council "이 아이디어의 가장 약한 고리는?" --context-file card.md --ideas-file ideas.txt
```

결과는 `runs/`에 markdown, JSON, Notion 행 payload(`*.notion.json`, 아이디어 허브의 주장·근거 DB와 연결 DB 속성 이름)로 저장됩니다.

환경 변수: `COUNCIL_MODELS`(쉼표 구분), `CHAIRMAN_MODEL`, `VERIFIER_MODEL`, `COUNCIL_ROLES`, `COUNCIL_ROTATE_ROLES=0`, `COUNCIL_MAX_CLAIMS`, `COUNCIL_SEED`.
기본 모델 목록은 llm-council README의 예시 설정과 같습니다. 새 모델이 나오면 환경 변수로 바꾸십시오.
웹 검색은 OpenRouter의 `openrouter:web_search` 서버 도구를 씁니다.

### 설계 근거

- 서로 다른 모델 계열로 된 판정단은 단일 대형 판정자보다 편향이 적습니다. [Verga et al., 2024](https://arxiv.org/abs/2404.18796)
- LLM 판정자에게는 위치·길이·자기 강화 편향이 있습니다. [Zheng et al., 2023](https://arxiv.org/abs/2306.05685), [Wang et al., 2024](https://aclanthology.org/2024.acl-long.511/)
- LLM은 자기 생성물을 알아보고 선호합니다. [Panickssery et al., 2024](https://doi.org/10.52202/079017-2197)
- 여러 에이전트의 상호 비판은 사실성을 높입니다. [Du et al., 2023](https://arxiv.org/abs/2305.14325)

### 테스트

```bash
pytest
```

테스트는 가짜 모델로 전체 흐름을 돌립니다. 실제 API 호출은 하지 않습니다.
