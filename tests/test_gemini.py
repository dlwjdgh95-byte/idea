import asyncio
import json
import sys

import httpx
import pytest

from idea_council import cli
from idea_council.config import CLAUDE_CHAIRMAN, CLAUDE_MEMBERS, CouncilConfig
from idea_council import gemini as gemini_mod
from idea_council.gemini import (
    AgyClient,
    GeminiClient,
    RoutedClient,
    gemini_client,
    missing_gemini_models,
    parse_gemini_response,
    to_gemini_request,
)
from idea_council.llm import Completion


def test_request_moves_system_into_system_instruction_and_maps_roles():
    body = to_gemini_request(
        [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ],
        web_search=True,
    )
    assert body["systemInstruction"] == {"parts": [{"text": "S"}]}
    assert [c["role"] for c in body["contents"]] == ["user", "model"]
    assert body["tools"] == [{"google_search": {}}]
    assert "tools" not in to_gemini_request([{"role": "user", "content": "a"}], web_search=False)


def test_parse_reads_text_and_grounding_sources_and_skips_thoughts():
    data = {
        "candidates": [
            {
                "content": {"parts": [{"text": "생각", "thought": True}, {"text": "안녕"}, {"text": "하세요"}]},
                "groundingMetadata": {"groundingChunks": [{"web": {"uri": "https://a.com", "title": "A"}}, {}]},
            }
        ]
    }
    c = parse_gemini_response("gemini-x", data)
    assert c.ok and c.text == "안녕하세요"
    assert c.citations == [{"url": "https://a.com", "title": "A", "content": ""}]


def test_parse_errors_never_raise():
    assert parse_gemini_response("g", [1]).error
    assert parse_gemini_response("g", {"error": {"message": "bad"}}).error
    assert "blocked" in parse_gemini_response("g", {"promptFeedback": {"blockReason": "SAFETY"}}).error
    assert parse_gemini_response("g", {"candidates": []}).error
    assert "SAFETY" in parse_gemini_response("g", {"candidates": [{"finishReason": "SAFETY"}]}).error


def test_client_needs_a_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        GeminiClient()
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    assert GeminiClient().api_key == "k"


def test_client_does_not_retry_a_bad_request(monkeypatch):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.headers["x-goog-api-key"] == "k"
        assert request.url.path.endswith("/models/gemini-x:generateContent")
        return httpx.Response(400, json={"error": {"message": "bad model"}})

    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    c = asyncio.run(GeminiClient("k").complete("gemini-x", [{"role": "user", "content": "hi"}]))
    assert c.error.startswith("HTTP 400") and len(calls) == 1


def test_client_returns_text(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "ok"}]}}]})

    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    c = asyncio.run(GeminiClient("k").complete("gemini-x", [{"role": "user", "content": "hi"}]))
    assert c.ok and c.text == "ok"


def test_missing_models_follows_pages(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("pageToken") == "p2":
            return httpx.Response(200, json={"models": [{"name": "models/gemini-b"}]})
        return httpx.Response(200, json={"models": [{"name": "models/gemini-a"}], "nextPageToken": "p2"})

    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    missing = asyncio.run(missing_gemini_models(["opus", "gemini-a", "gemini-b", "gemini-c"], api_key="k"))
    assert missing == ["gemini-c"]


class Recorder:
    def __init__(self, name):
        self.name, self.models = name, []

    async def complete(self, model, messages, *, web_search=False):
        self.models.append(model)
        return Completion(model=model, text=self.name)


def test_routed_client_sends_gemini_ids_to_gemini():
    claude, gemini = Recorder("claude"), Recorder("gemini")
    routed = RoutedClient(claude, gemini)
    assert asyncio.run(routed.complete("gemini-x", [])).text == "gemini"
    assert asyncio.run(routed.complete("opus", [])).text == "claude"
    assert (claude.models, gemini.models) == (["opus"], ["gemini-x"])


def test_mixed_backend_seats_gemini_next_to_claude(monkeypatch):
    for var in ("COUNCIL_MODELS", "CHAIRMAN_MODEL", "VERIFIER_MODEL", "GEMINI_MODELS"):
        monkeypatch.delenv(var, raising=False)
    config = CouncilConfig.from_env("mixed")
    assert config.members == (*CLAUDE_MEMBERS, "gemini-4-argon-high", "gemini-3.8-flash-high")
    assert config.chairman == CLAUDE_CHAIRMAN
    monkeypatch.setenv("GEMINI_MODELS", "gemini-other")
    assert CouncilConfig.from_env("mixed").members[4:] == ("gemini-other",)


def test_cli_defaults_to_mixed_only_when_a_key_is_set(monkeypatch):
    seen = {}

    def stop(backend="claude-code"):
        seen["backend"] = backend
        raise ValueError("stop here")

    monkeypatch.setattr(cli.CouncilConfig, "from_env", staticmethod(stop))
    monkeypatch.delenv("COUNCIL_BACKEND", raising=False)
    monkeypatch.delenv("AGY_OAUTH_TOKEN_B64", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert cli.main(["q"]) == 2 and seen["backend"] == "claude-code"
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert cli.main(["q"]) == 2 and seen["backend"] == "mixed"
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.setenv("AGY_OAUTH_TOKEN_B64", "dA==")
    assert cli.main(["q"]) == 2 and seen["backend"] == "mixed"


def fake_agy(tmp_path, body: str) -> str:
    script = tmp_path / "fake_agy.py"
    script.write_text(body, encoding="utf-8")
    wrapper = tmp_path / "agy"
    wrapper.write_text(f"#!/bin/sh\nexec {sys.executable} {script} \"$@\"\n", encoding="utf-8")
    wrapper.chmod(0o755)
    return str(wrapper)


def test_agy_client_passes_one_prompt_and_keeps_secrets_out(tmp_path, monkeypatch):
    monkeypatch.setenv("BLOGGER_REFRESH_TOKEN", "secret")
    binary = fake_agy(tmp_path, (
        "import json, os, sys\n"
        "a = sys.argv[1:]\n"
        "print(json.dumps({'prompt': a[a.index('-p') + 1], 'model': a[a.index('--model') + 1],\n"
        "  'sandbox': '--sandbox' in a, 'leak': 'BLOGGER_REFRESH_TOKEN' in os.environ, 'cwd': os.listdir('.')}))\n"
    ))
    c = asyncio.run(AgyClient(binary).complete(
        "gemini-x", [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]))
    out = json.loads(c.text)
    assert c.ok and out["model"] == "gemini-x" and out["sandbox"] and out["cwd"] == []
    assert out["prompt"].startswith("S") and out["prompt"].endswith("U")
    assert not out["leak"]


def test_agy_client_reports_failures(tmp_path):
    binary = fake_agy(tmp_path, "import sys\nprint('not logged in', file=sys.stderr)\nsys.exit(3)\n")
    c = asyncio.run(AgyClient(binary, max_retries=1).complete("gemini-x", [{"role": "user", "content": "U"}]))
    assert c.error == "agy exit 3: not logged in"
    empty = fake_agy(tmp_path, "pass\n")
    assert asyncio.run(AgyClient(empty, max_retries=1).complete("g", [{"role": "user", "content": "U"}])).error


def test_gemini_client_prefers_agy_and_falls_back_to_the_key(monkeypatch):
    monkeypatch.setattr(gemini_mod, "find_agy", lambda setup=True: "/bin/agy")
    monkeypatch.setenv("AGY_OAUTH_TOKEN_B64", "dA==")
    assert isinstance(gemini_client(), AgyClient)
    monkeypatch.setattr(gemini_mod, "find_agy", lambda setup=True: None)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert isinstance(gemini_client(), GeminiClient)
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        gemini_client()
