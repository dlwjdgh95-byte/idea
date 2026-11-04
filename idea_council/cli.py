"""Command line entry point: ``idea-council "질문" --context-file card.md``."""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import pathlib
import sys

import httpx

from .config import CouncilConfig
from .council import Council
from .export import notion_claim_rows, notion_link_rows, to_json, to_markdown
from .llm import OpenRouterClient, missing_models


def _read(path: str | None) -> str:
    return pathlib.Path(path).read_text(encoding="utf-8") if path else ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="idea-council", description="LLM 카운슬로 아이디어를 전개합니다.")
    parser.add_argument("question", help="카운슬에 던질 질문")
    parser.add_argument("--context-file", help="아이디어 스레드의 맥락 카드(markdown)")
    parser.add_argument("--ideas-file", help="다른 아이디어 목록, 한 줄에 하나")
    parser.add_argument("--idea-url", help="Notion 아이디어 페이지 URL(행 payload에 넣음)")
    parser.add_argument("--out", default="runs", help="결과 저장 폴더")
    parser.add_argument("--no-verify", action="store_true", help="사실 검증 게이트를 건너뜀")
    parser.add_argument("--skip-model-check", action="store_true", help="OpenRouter 모델 목록 확인을 건너뜀")
    args = parser.parse_args(argv)

    try:
        context = _read(args.context_file)
        ideas_text = _read(args.ideas_file)
    except OSError as exc:
        sys.stderr.write(f"파일을 읽지 못했습니다: {exc}\n")
        return 2
    try:
        config = CouncilConfig.from_env()
    except ValueError as exc:
        sys.stderr.write(f"설정 오류: {exc}\n")
        return 2
    if not args.skip_model_check:
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
    try:
        llm = OpenRouterClient(
            timeout_s=config.timeout_s,
            web_search_max_results=config.web_search_max_results,
            extra_headers=dict(config.extra_headers),
        )
    except RuntimeError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    ideas = [line.strip() for line in ideas_text.splitlines() if line.strip()]
    result = asyncio.run(
        Council(llm, config).run(args.question, context, ideas, verify=not args.no_verify)
    )

    out = pathlib.Path(args.out)
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
