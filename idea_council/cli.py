"""Command line entry point: ``idea-council "질문" --context-file card.md``."""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import pathlib
import sys

import httpx

from .claude_code import ClaudeCodeClient
from .config import BACKENDS, DEFAULT_BACKEND, CouncilConfig
from .council import Council, roles_from_run
from .export import notion_claim_rows, notion_link_rows, to_json, to_markdown
from .gemini import AgyClient, RoutedClient, gemini_available, gemini_client, missing_gemini_models
from .llm import LLM, OpenRouterClient, missing_models


def _read(path: str | None) -> str:
    return pathlib.Path(path).read_text(encoding="utf-8") if path else ""


def previous_roles(out: pathlib.Path) -> dict[str, str]:
    """Roles from the newest saved run in ``out``, so the next run can rotate them one step."""
    runs = sorted(p for p in out.glob("*.json") if not p.name.endswith(".notion.json")) if out.is_dir() else []
    for path in reversed(runs):
        try:
            roles = roles_from_run(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
        if roles:
            return roles
    return {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="idea-council", description="LLM 카운슬로 아이디어를 전개합니다.")
    parser.add_argument("question", help="카운슬에 던질 질문")
    parser.add_argument("--context-file", help="아이디어 스레드의 맥락 카드(markdown)")
    parser.add_argument("--ideas-file", help="다른 아이디어 목록, 한 줄에 하나")
    parser.add_argument("--idea-url", help="Notion 아이디어 페이지 URL(행 payload에 넣음)")
    parser.add_argument("--out", default="runs", help="결과 저장 폴더")
    parser.add_argument("--no-verify", action="store_true", help="사실 검증 게이트를 건너뜀")
    parser.add_argument("--skip-model-check", action="store_true", help="OpenRouter·Gemini 모델 목록 확인을 건너뜀")
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default=os.environ.get("COUNCIL_BACKEND") or ("mixed" if gemini_available() else DEFAULT_BACKEND),
        help=(
            "claude-code(로컬 claude CLI, 키 불필요), mixed(Claude 4명 + Gemini 1명, AGY_OAUTH_TOKEN_B64 또는 "
            "GEMINI_API_KEY가 있으면 기본), openrouter(여러 회사 모델, 키 필요)"
        ),
    )
    args = parser.parse_args(argv)

    try:
        context = _read(args.context_file)
        ideas_text = _read(args.ideas_file)
    except OSError as exc:
        sys.stderr.write(f"파일을 읽지 못했습니다: {exc}\n")
        return 2
    try:
        config = CouncilConfig.from_env(args.backend)
    except ValueError as exc:
        sys.stderr.write(f"설정 오류: {exc}\n")
        return 2
    if args.backend == "openrouter" and not args.skip_model_check:
        try:
            missing = asyncio.run(missing_models(config.all_models()))
        except httpx.HTTPError as exc:
            sys.stderr.write(f"경고: OpenRouter 모델 목록을 확인하지 못했습니다({exc}). 그대로 진행합니다.\n")
        else:
            if missing:
                sys.stderr.write(
                    "OpenRouter에 없는 모델입니다: " + ", ".join(missing)
                    + "\nCOUNCIL_MODELS, CHAIRMAN_MODEL, VERIFIER_MODEL 환경 변수로 바꾸십시오.\n"
                )
                return 2
    gemini = None
    if args.backend == "mixed":
        try:
            gemini = gemini_client(timeout_s=max(config.timeout_s, 300.0))
        except RuntimeError as exc:
            sys.stderr.write(f"{exc}\n")
            return 2
    if gemini is not None and not isinstance(gemini, AgyClient) and not args.skip_model_check:
        try:
            missing = asyncio.run(missing_gemini_models(config.all_models()))
        except httpx.HTTPError as exc:
            sys.stderr.write(f"경고: Gemini 모델 목록을 확인하지 못했습니다({exc}). 그대로 진행합니다.\n")
        else:
            if missing:
                sys.stderr.write(
                    "Gemini API에 없는 모델입니다: " + ", ".join(missing)
                    + "\nGEMINI_MODEL 환경 변수로 바꾸십시오.\n"
                )
                return 2
    llm: LLM
    try:
        if args.backend == "claude-code":
            llm = ClaudeCodeClient(timeout_s=max(config.timeout_s, 300.0))
        elif args.backend == "mixed":
            llm = RoutedClient(ClaudeCodeClient(timeout_s=max(config.timeout_s, 300.0)), gemini)
        else:
            llm = OpenRouterClient(
                timeout_s=config.timeout_s,
                web_search_max_results=config.web_search_max_results,
                extra_headers=dict(config.extra_headers),
            )
    except RuntimeError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    ideas = [line.strip() for line in ideas_text.splitlines() if line.strip()]
    out = pathlib.Path(args.out)
    council = Council(llm, config, previous_roles=previous_roles(out))
    result = asyncio.run(council.run(args.question, context, ideas, verify=not args.no_verify))

    out.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    (out / f"{stamp}.md").write_text(to_markdown(result), encoding="utf-8")
    (out / f"{stamp}.json").write_text(to_json(result), encoding="utf-8")
    (out / f"{stamp}.notion.json").write_text(
        json.dumps(
            {"claims": notion_claim_rows(result, args.idea_url), "links": notion_link_rows(result, args.idea_url)},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    sys.stdout.write(to_markdown(result) + "\n")
    return 1 if result.chairman_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
