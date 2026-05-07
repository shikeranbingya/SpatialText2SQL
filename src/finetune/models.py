"""Data models for TRL-based spatial Text-to-SQL fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .utils import stable_jsonify, to_text


def _as_text_list(value: Any) -> list[str]:
    normalized = stable_jsonify(value)
    if normalized in (None, ""):
        return []
    if isinstance(normalized, str):
        return [normalized] if normalized else []
    if isinstance(normalized, list):
        return [to_text(item) for item in normalized if to_text(item)]
    text = to_text(normalized)
    return [text] if text else []


def _as_mapping(value: Any) -> dict[str, Any]:
    normalized = stable_jsonify(value)
    if isinstance(normalized, Mapping):
        return {str(key): stable_jsonify(val) for key, val in normalized.items()}
    return {}


def _as_list_of_mappings(value: Any) -> list[dict[str, Any]]:
    normalized = stable_jsonify(value)
    if normalized in (None, ""):
        return []
    if isinstance(normalized, Mapping):
        normalized = [normalized]
    if not isinstance(normalized, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in normalized:
        if isinstance(item, Mapping):
            rows.append({str(key): stable_jsonify(val) for key, val in item.items()})
    return rows


@dataclass(frozen=True)
class RawFinetuneSample:
    database_id: str
    sql: str
    question: str
    difficulty: str
    question_id: str = ""
    city: str = ""
    instruction: str = ""
    input_text: str = ""
    output_text: str = ""
    sql_reasoning_summary: str = ""
    used_tables: list[str] = field(default_factory=list)
    used_columns: list[str] = field(default_factory=list)
    used_spatial_functions: list[str] = field(default_factory=list)
    sql_features: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RawFinetuneSample":
        database_id = to_text(payload.get("database_id"))
        sql = to_text(payload.get("sql"))
        question = to_text(payload.get("question"))
        difficulty = to_text(
            payload.get("difficulty")
            or payload.get("source_difficulty_level")
            or payload.get("difficulty_level")
        )
        if not database_id:
            raise ValueError("Missing required field: database_id")
        if not sql:
            raise ValueError("Missing required field: sql")
        if not question:
            raise ValueError("Missing required field: question")
        if not difficulty:
            raise ValueError("Missing required field: difficulty/source_difficulty_level")
        return cls(
            question_id=to_text(payload.get("question_id")),
            database_id=database_id,
            city=to_text(payload.get("city")),
            sql=sql,
            question=question,
            difficulty=difficulty,
            instruction=to_text(payload.get("instruction")),
            input_text=to_text(payload.get("input")),
            output_text=to_text(payload.get("output")),
            sql_reasoning_summary=to_text(payload.get("sql_reasoning_summary")),
            used_tables=_as_text_list(payload.get("used_tables")),
            used_columns=_as_text_list(payload.get("used_columns")),
            used_spatial_functions=_as_text_list(payload.get("used_spatial_functions")),
            sql_features=_as_mapping(payload.get("sql_features")),
            metadata=_as_mapping(payload.get("metadata")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "database_id": self.database_id,
            "city": self.city,
            "question": self.question,
            "sql": self.sql,
            "difficulty": self.difficulty,
            "instruction": self.instruction,
            "input": self.input_text,
            "output": self.output_text,
            "sql_reasoning_summary": self.sql_reasoning_summary,
            "used_tables": list(self.used_tables),
            "used_columns": list(self.used_columns),
            "used_spatial_functions": list(self.used_spatial_functions),
            "sql_features": stable_jsonify(self.sql_features),
            "metadata": stable_jsonify(self.metadata),
        }


@dataclass(frozen=True)
class PreparedFinetuneSample:
    question_id: int
    database_id: str
    question: str
    sql: str
    difficulty: str
    prompt: str
    completion: str
    instruction: str = ""
    input_text: str = ""
    output_text: str = ""
    cot: str = ""
    sql_reasoning_summary: str = ""
    schema: list[str] = field(default_factory=list)
    representative_values: dict[str, Any] = field(default_factory=dict)
    used_tables: list[str] = field(default_factory=list)
    used_columns: list[str] = field(default_factory=list)
    used_spatial_functions: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PreparedFinetuneSample":
        question_id = payload.get("question_id")
        if question_id in (None, ""):
            raise ValueError("Missing required field: question_id")
        return cls(
            question_id=int(question_id),
            database_id=to_text(payload.get("database_id")),
            question=to_text(payload.get("question")),
            sql=to_text(payload.get("sql")),
            difficulty=to_text(payload.get("difficulty")),
            prompt=to_text(payload.get("prompt")),
            completion=to_text(payload.get("completion")),
            instruction=to_text(payload.get("instruction")),
            input_text=to_text(payload.get("input")),
            output_text=to_text(payload.get("output")),
            cot=to_text(payload.get("cot")),
            sql_reasoning_summary=to_text(payload.get("sql_reasoning_summary")),
            schema=_as_text_list(payload.get("schema")),
            representative_values=_as_mapping(payload.get("representative_values")),
            used_tables=_as_text_list(payload.get("used_tables")),
            used_columns=_as_text_list(payload.get("used_columns")),
            used_spatial_functions=_as_text_list(payload.get("used_spatial_functions")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "database_id": self.database_id,
            "question": self.question,
            "sql": self.sql,
            "difficulty": self.difficulty,
            "prompt": self.prompt,
            "completion": self.completion,
            "instruction": self.instruction,
            "input": self.input_text,
            "output": self.output_text,
            "cot": self.cot,
            "sql_reasoning_summary": self.sql_reasoning_summary,
            "schema": list(self.schema),
            "representative_values": stable_jsonify(self.representative_values),
            "used_tables": list(self.used_tables),
            "used_columns": list(self.used_columns),
            "used_spatial_functions": list(self.used_spatial_functions),
        }
