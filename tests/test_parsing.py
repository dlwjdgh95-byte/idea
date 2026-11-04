from idea_council.parsing import last_json_block, parse_review, strip_json_blocks

JSON_REVIEW = """비판...
```json
{"reviews": [{"label": "A", "accuracy": 4, "evidence": 2, "novelty": 5, "feasibility": 3,
  "critique": "근거가 약함", "suspect_claims": ["X는 2020년에 출시됐다"]},
  {"label": "응답 B", "accuracy": 5, "evidence": 5, "novelty": 2, "feasibility": 4, "critique": "탄탄함"}],
 "ranking": ["응답 B", "A", "Z"]}
```"""


def test_parse_json_review():
    review = parse_review(JSON_REVIEW, ["A", "B"])
    assert review.ranking == ["B", "A"]  # unknown label Z dropped, "응답 B" normalised
    assert review.scores["A"]["evidence"] == 2
    assert review.suspect_claims["A"] == ["X는 2020년에 출시됐다"]
    assert review.parsed_from == "json"


def test_parse_final_ranking_fallback():
    text = "설명\nFINAL RANKING:\n1. 응답 C\n2. Response A\n3. 응답 B"
    review = parse_review(text, ["A", "B", "C"])
    assert review.ranking == ["C", "A", "B"]
    assert review.parsed_from == "text"


def test_last_json_block_skips_invalid_and_strip():
    text = 'a\n```json\n{"x": 1}\n```\nb\n```json\n{broken\n```'
    assert last_json_block(text) == {"x": 1}
    assert strip_json_blocks('본문\n```json\n{"x": 1}\n```') == "본문"
