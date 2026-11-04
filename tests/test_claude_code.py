import asyncio
import json
import random
import sys
import textwrap

from idea_council.claude_code import ClaudeCodeClient, parse_cli_output, render_messages
from idea_council.cli import previous_roles
from idea_council.config import CLAUDE_MEMBERS, OPENROUTER_MEMBERS, CouncilConfig
from idea_council.council import Council, roles_from_run
from idea_council.llm import Completion
from idea_council.verify import Verifier


def fake_cli(tmp_path, body: str):
    """A stand-in for the claude binary: a python script that prints CLI-style JSON."""
    script = tmp_path / "fake_claude.py"
    script.write_text(textwrap.dedent(body), encoding="utf-8")
    wrapper = tmp_path / "claude"
    wrapper.write_text(f"#!/bin/sh\nexec {sys.executable} {script} \"$@\"\n", encoding="utf-8")
    wrapper.chmod(0o755)
    return str(wrapper)


def test_render_messages_splits_system_and_passes_single_user_turn():
    system, prompt = render_messages([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}])
    assert (system, prompt) == ("S", "U")
    _, transcript = render_messages(
        [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"}]
    )
    assert "### 사용자\na" in transcript and "### 어시스턴트\nb" in transcript


def test_parse_cli_output_errors_never_raise():
    assert parse_cli_output("m", "not json").error
    assert parse_cli_output("m", "[1]").error
    assert parse_cli_output("m", json.dumps({"is_error": True, "result": "boom"})).error
    assert parse_cli_output("m", json.dumps({"subtype": "error_max_turns"})).error
    assert parse_cli_output("m", json.dumps({"subtype": "success", "result": None})).error
    ok = parse_cli_output("m", json.dumps({"subtype": "success", "is_error": False, "result": "안녕"}))
    assert ok.ok and ok.text == "안녕"


def test_command_has_no_tools_unless_web_search(tmp_path):
    client = ClaudeCodeClient(binary="/bin/true", workdir=str(tmp_path))
    plain = client.command("haiku", "S", web_search=False)
    assert plain[plain.index("--tools") + 1] == ""
    assert "--allowedTools" not in plain
    assert plain[plain.index("--system-prompt") + 1] == "S"
    assert {"--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence"} <= set(plain)
    web = client.command("sonnet", "S", web_search=True)
    assert web[web.index("--tools") + 1] == "WebSearch,WebFetch"
    assert web[web.index("--allowedTools") + 1] == "WebSearch,WebFetch"


def test_complete_runs_the_binary_with_prompt_on_stdin(tmp_path):
    binary = fake_cli(tmp_path, """
        import json, sys
        args = sys.argv[1:]
        prompt = sys.stdin.read()
        print(json.dumps({"subtype": "success", "is_error": False,
                          "result": args[args.index("--model") + 1] + "|" + prompt}))
    """)
    client = ClaudeCodeClient(binary=binary, workdir=str(tmp_path))
    c = asyncio.run(client.complete("haiku", [{"role": "system", "content": "S"}, {"role": "user", "content": "질문"}]))
    assert c.ok and c.text == "haiku|질문"


def test_complete_reports_cli_failure_and_timeout(tmp_path):
    failing = fake_cli(tmp_path, """
        import json, sys
        print(json.dumps({"is_error": True, "result": "rate limited"}))
        sys.exit(1)
    """)
    c = asyncio.run(ClaudeCodeClient(binary=failing, workdir=str(tmp_path), max_retries=1).complete("m", []))
    assert not c.ok and "rate limited" in c.error

    slow = fake_cli(tmp_path, """
        import time
        time.sleep(5)
    """)
    c = asyncio.run(ClaudeCodeClient(binary=slow, workdir=str(tmp_path), max_retries=1, timeout_s=0.5).complete("m", []))
    assert not c.ok and "timed out" in c.error


def test_missing_binary_is_an_error_not_a_crash(tmp_path):
    c = asyncio.run(ClaudeCodeClient(binary=str(tmp_path / "nope"), workdir=str(tmp_path)).complete("m", []))
    assert not c.ok


def test_claude_code_defaults_mix_model_sizes(monkeypatch):
    for var in ("COUNCIL_MODELS", "CHAIRMAN_MODEL", "VERIFIER_MODEL"):
        monkeypatch.delenv(var, raising=False)
    config = CouncilConfig.from_env("claude-code")
    assert config.members == CLAUDE_MEMBERS
    assert {"opus", "sonnet", "haiku"} <= set(config.members)


def test_the_council_runs_inside_claude_by_default(monkeypatch):
    for var in ("COUNCIL_MODELS", "CHAIRMAN_MODEL", "VERIFIER_MODEL"):
        monkeypatch.delenv(var, raising=False)
    assert CouncilConfig().members == CLAUDE_MEMBERS
    assert CouncilConfig.from_env().members == CLAUDE_MEMBERS
    assert CouncilConfig.from_env("openrouter").members == OPENROUTER_MEMBERS


def test_roles_rotate_one_step_from_the_previous_run():
    members = ("opus", "sonnet", "haiku", "fable")
    config = CouncilConfig(members=members, chairman="opus", verifier="sonnet")
    seen = {m: [] for m in members}
    previous: dict[str, str] = {}
    for _ in range(len(config.roles)):
        council = Council(object(), config, verifier=object(), rng=random.Random(1), previous_roles=previous)
        previous = dict(council.assign_roles())
        assert sorted(previous.values()) == sorted(config.roles)
        for m, r in previous.items():
            seen[m].append(r)
    assert all(sorted(rs) == sorted(config.roles) for rs in seen.values())  # every model played every role


def test_previous_roles_reads_the_newest_run(tmp_path):
    (tmp_path / "20260101-000000Z.json").write_text(
        json.dumps({"answers": [{"member": "opus", "role": "critic"}]}), encoding="utf-8")
    (tmp_path / "20260102-000000Z.json").write_text(
        json.dumps({"answers": [{"member": "opus", "role": "builder"}]}), encoding="utf-8")
    (tmp_path / "20260103-000000Z.notion.json").write_text("{}", encoding="utf-8")
    assert previous_roles(tmp_path) == {"opus": "builder"}
    assert previous_roles(tmp_path / "missing") == {}
    assert roles_from_run(["bad"]) == {}


class EchoLLM:
    def __init__(self) -> None:
        self.chairman_prompt = ""

    async def complete(self, model, messages, *, web_search=False):
        system = messages[0]["content"]
        if "의장" in system:
            self.chairman_prompt = messages[1]["content"]
            return Completion(model, "### 결론\nok")
        if "검토자" in system:
            body = messages[1]["content"]
            labels = [line.split()[-1] for line in body.splitlines() if line.startswith("### 응답 ")]
            data = {"reviews": [{"label": lab, "critique": "평", "accuracy": 3} for lab in labels], "ranking": labels}
            return Completion(model, "```json\n" + json.dumps(data, ensure_ascii=False) + "\n```")
        return Completion(model, f"답 {model}")


def test_chairman_never_sees_model_names():
    members = ("opus", "sonnet", "haiku", "fable")
    llm = EchoLLM()
    config = CouncilConfig(members=members, chairman="opus", verifier="sonnet", check_source_urls=False)
    council = Council(llm, config, verifier=Verifier(llm, "sonnet", check_urls=False), rng=random.Random(3))
    asyncio.run(council.run("질문", verify=False))
    sections = llm.chairman_prompt.split("## 위원 답변")[1].split("## 사실 검증 결과")[0]
    for m in members:
        assert f"({m})" not in sections and f"- {m}" not in sections and f"{m} →" not in sections
    assert "### 확장자" in sections and "평균 순위" in sections and "→" in sections
