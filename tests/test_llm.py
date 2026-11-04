import httpx

from idea_council.llm import _parse_response

REQ = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")


def test_bad_bodies_become_errors_not_exceptions():
    assert _parse_response("m", httpx.Response(200, text="<html>gateway</html>", request=REQ)).error
    assert _parse_response("m", httpx.Response(200, json={"error": {"message": "upstream"}}, request=REQ)).error
    choice_error = {"choices": [{"error": {"message": "x"}, "message": {"content": ""}}]}
    assert _parse_response("m", httpx.Response(200, json=choice_error, request=REQ)).error


def test_good_body_is_parsed():
    body = {"choices": [{"message": {"content": [{"type": "text", "text": "안녕"}],
            "annotations": [{"type": "url_citation", "url_citation": {"url": "https://a.com", "title": "A"}}]}}]}
    c = _parse_response("m", httpx.Response(200, json=body, request=REQ))
    assert c.ok and c.text == "안녕" and c.citations[0]["url"] == "https://a.com"
