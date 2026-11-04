"""Parsing helpers for model output: JSON blocks, rankings, reviews."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

_FENCE = re.compile(r"```[ \t]*[a-z0-9]*[ \t]*\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_RANK_LINE = re.compile(r"^\s*\d+[.)]\s*(?:응답|Response)\s*([A-Z])\b", re.MULTILINE)
_ANY_LABEL = re.compile(r"(?:응답|Response)\s*([A-Z])\b")
SCORE_KEYS = ("accuracy", "evidence", "novelty", "feasibility")


def as_list(value: Any) -> list[Any]:
    """Model JSON is loose: a bare string means a one-item list, anything else non-list means none."""
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        return [value]
    return []


def last_json_block(text: str) -> dict[str, Any] | None:
    """Return the last fenced JSON object in ``text`` that parses."""
    for raw in reversed(_FENCE.findall(text or "")):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    # No fenced block: accept a reply that is (or ends with) bare JSON.
    text = (text or "").strip()
    start = text.find("{")
    while start != -1:
        try:
            value, end = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        if isinstance(value, dict) and not text[start + end:].strip():
            return value
        start = text.find("{", start + 1)
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
        raw_ranking = data.get("ranking")
        if isinstance(raw_ranking, str):  # "A > B > C" or "A, B, C"
            ranking = re.findall(r"(?<![A-Za-z])([A-Z])(?![A-Za-z])", raw_ranking)
        else:
            ranking = [_clean_label(x) for x in as_list(raw_ranking)]
        ranking = list(dict.fromkeys(label for label in ranking if label in valid))
        review = Review(ranking=ranking)
        for item in as_list(data.get("reviews")):
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
                str(c).strip() for c in as_list(item.get("suspect_claims")) if isinstance(c, str) and c.strip()
            ]
        if not review.ranking:
            # Keep the scores and suspect claims; take the ranking from the text if there is one.
            review.ranking = _text_ranking(text, valid)
        return review

    return Review(ranking=_text_ranking(text, valid), parsed_from="text")


def _text_ranking(text: str, valid: set[str]) -> list[str]:
    """Fallback, same idea as karpathy/llm-council: read the list after "FINAL RANKING:".

    Without that marker only numbered lines count; the order labels are merely
    mentioned in is not a ranking.
    """
    text = text or ""
    if "FINAL RANKING:" in text:
        section = text.split("FINAL RANKING:", 1)[1]
        labels = _RANK_LINE.findall(section) or _ANY_LABEL.findall(section)
    else:
        labels = _RANK_LINE.findall(text)
    return list(dict.fromkeys(label for label in labels if label in valid))
