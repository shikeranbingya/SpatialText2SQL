"""Parse LLM responses for question generation."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

from src.synthesis.database.utils import stable_jsonify, to_text

from .models import QuestionGenerationCandidate


def _extract_json_payload(text: str) -> str:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.I | re.S)
    if fenced:
        return fenced.group(1).strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if first_brace >= 0 and last_brace > first_brace:
        return stripped[first_brace : last_brace + 1].strip()
    return stripped


def parse_question_generation_response(
    response_text: str,
    *,
    raw_response: Any = None,
) -> QuestionGenerationCandidate:
    raw_text = to_text(response_text)
    payload_text = _extract_json_payload(raw_text)
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        return QuestionGenerationCandidate(
            question="",
            raw_response_text=raw_text,
            raw_response=stable_jsonify(raw_response),
            parse_error=f"Failed to parse question-generation JSON response: {exc}",
        )
    if not isinstance(payload, Mapping):
        return QuestionGenerationCandidate(
            question="",
            raw_response_text=raw_text,
            raw_response=stable_jsonify(raw_response or payload),
            parse_error="Question-generation response must be a JSON object.",
        )
    question = to_text(payload.get("question"))
    if not question:
        return QuestionGenerationCandidate(
            question="",
            raw_response_text=raw_text,
            raw_response=stable_jsonify(raw_response or payload),
            parse_error="Question-generation response is missing the 'question' field.",
        )
    spatial_phrases = payload.get("spatial_phrases", [])
    if isinstance(spatial_phrases, str):
        spatial_phrase_list = [spatial_phrases] if spatial_phrases else []
    elif isinstance(spatial_phrases, list):
        spatial_phrase_list = [to_text(item) for item in spatial_phrases if to_text(item)]
    else:
        spatial_phrase_list = []
    return QuestionGenerationCandidate(
        question=question,
        style=to_text(payload.get("style")),
        reasoning_summary=to_text(payload.get("reasoning_summary")),
        spatial_phrases=spatial_phrase_list,
        raw_response_text=raw_text,
        raw_response=stable_jsonify(raw_response or payload),
    )
