"""Parsing helpers for model output: JSON blocks, rankings, reviews."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_RANK_LINE = re.compile(r"^\s*\d+[.)]\s*(?:응답|Response)\s*([A-Z])\b", re.MULTILINE)
_ANY_LABEL = re.compile(r"(?:응답|Response)\s*([A-Z])\b")
SCORE_KEYS = ("accuracy", "evidence", "novelty", "feasibility")


def last_json_block(text: str) -> dict[str, Any] | None:
    """Return the last fenced JSON object in ``text`` that parses."""
    for raw in reversed(_FENCE.findall(text or "")):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def strip_json_blocks(text: str) -> str:
    return _FENCE.sub("", text or "").strip()


@dataclass
class Review:
    ranking: list[str]
    scores: dict[str, dict[str, float]] = field(default_factory=dict)
    critiques: dict[str, str] = field(default_factory=dict)
    suspect_claims: dict[str, list[str]] = field(default_factory=dict)
    parsed_from: str = "json"


def _clean_label(value: Any) -> str:
    match = re.search(r"([A-Z])\s*$", str(value).strip())
    return match.group(1) if match else ""


def parse_review(text: str, valid_labels: list[str]) -> Review:
    valid = set(valid_labels)
    data = last_json_block(text)
    if data is not None:
        ranking = [_clean_label(x) for x in data.get("ranking") or []]
        ranking = list(dict.fromkeys(label for label in ranking if label in valid))
        review = Review(ranking=ranking)
        for item in data.get("reviews") or []:
            if not isinstance(item, dict):
                continue
            label = _clean_label(item.get("label", ""))
            if label not in valid:
                continue
            scores = {}
            for key in SCORE_KEYS:
                try:
                    scores[key] = float(item[key])
                except (KeyError, TypeError, ValueError):
                    continue
            review.scores[label] = scores
            review.critiques[label] = str(item.get("critique", "")).strip()
            review.suspect_claims[label] = [
                str(c).strip() for c in item.get("suspect_claims") or [] if str(c).strip()
            ]
        if review.ranking:
            return review

    # Fallback, same idea as karpathy/llm-council: read "FINAL RANKING:".
    section = text.split("FINAL RANKING:", 1)[1] if "FINAL RANKING:" in (text or "") else (text or "")
    labels = _RANK_LINE.findall(section) or _ANY_LABEL.findall(section)
    ranking = list(dict.fromkeys(label for label in labels if label in valid))
    return Review(ranking=ranking, parsed_from="text")
