"""Structured models for constraint-guided SQL synthesis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from src.synthesis.database.utils import stable_jsonify, to_text


DIFFICULTY_LEVELS = ("easy", "medium", "hard", "extra-hard")


def _as_list_of_text(values: Any) -> list[str]:
    normalized = stable_jsonify(values)
    if normalized in (None, ""):
        return []
    if isinstance(normalized, str):
        return [normalized] if normalized else []
    if isinstance(normalized, list):
        return [to_text(item) for item in normalized if to_text(item)]
    return [to_text(normalized)] if to_text(normalized) else []


def _as_mapping(value: Any) -> dict[str, Any]:
    normalized = stable_jsonify(value)
    if isinstance(normalized, Mapping):
        return {str(key): stable_jsonify(val) for key, val in normalized.items()}
    return {}


@dataclass
class PostGISFunction:
    function_name: str
    signature: str
    input_args: list[str] = field(default_factory=list)
    return_type: str = ""
    description: str = ""
    example_usages: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    compatible_difficulties: list[str] = field(default_factory=list)
    source: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PostGISFunction":
        return cls(
            function_name=to_text(payload.get("function_name")),
            signature=to_text(payload.get("signature")),
            input_args=_as_list_of_text(payload.get("input_args")),
            return_type=to_text(payload.get("return_type")),
            description=to_text(payload.get("description")),
            example_usages=_as_list_of_text(payload.get("example_usages")),
            categories=_as_list_of_text(payload.get("categories")),
            compatible_difficulties=_as_list_of_text(payload.get("compatible_difficulties")),
            source=_as_list_of_text(payload.get("source")),
            metadata=_as_mapping(payload.get("metadata")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "function_name": self.function_name,
            "signature": self.signature,
            "input_args": list(self.input_args),
            "return_type": self.return_type,
            "description": self.description,
            "example_usages": list(self.example_usages),
            "categories": list(self.categories),
            "compatible_difficulties": list(self.compatible_difficulties),
            "source": list(self.source),
            "metadata": stable_jsonify(self.metadata),
        }


@dataclass
class SQLGenerationCandidate:
    sql: str
    used_tables: list[str] = field(default_factory=list)
    used_columns: list[str] = field(default_factory=list)
    used_spatial_functions: list[str] = field(default_factory=list)
    reasoning_summary: str = ""
    raw_response_text: str = ""
    raw_response: Any = None
    parse_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "sql": self.sql,
            "used_tables": list(self.used_tables),
            "used_columns": list(self.used_columns),
            "used_spatial_functions": list(self.used_spatial_functions),
            "reasoning_summary": self.reasoning_summary,
            "raw_response_text": self.raw_response_text,
            "raw_response": stable_jsonify(self.raw_response),
            "parse_error": self.parse_error,
        }


@dataclass
class SQLValidationResult:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    detected_tables: list[str] = field(default_factory=list)
    detected_columns: list[str] = field(default_factory=list)
    detected_spatial_functions: list[str] = field(default_factory=list)
    detected_difficulty_features: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "detected_tables": list(self.detected_tables),
            "detected_columns": list(self.detected_columns),
            "detected_spatial_functions": list(self.detected_spatial_functions),
            "detected_difficulty_features": stable_jsonify(self.detected_difficulty_features),
        }


@dataclass
class SQLExecutionResult:
    executed: bool
    success: bool
    error_message: str = ""
    row_count: int = 0
    empty_result: bool = False
    sample_rows: list[dict[str, Any]] = field(default_factory=list)
    execution_time_ms: float = 0.0
    actual_database: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "executed": self.executed,
            "success": self.success,
            "error_message": self.error_message,
            "row_count": self.row_count,
            "empty_result": self.empty_result,
            "sample_rows": stable_jsonify(self.sample_rows),
            "execution_time_ms": self.execution_time_ms,
            "actual_database": self.actual_database,
        }


@dataclass
class SynthesizedSQLQuery:
    sql_id: str
    database_id: str
    city: str
    difficulty_level: str
    sql: str
    reasoning_summary: str = ""
    used_tables: list[str] = field(default_factory=list)
    used_columns: list[str] = field(default_factory=list)
    used_spatial_functions: list[str] = field(default_factory=list)
    structural_constraints: dict[str, Any] = field(default_factory=dict)
    spatial_function_constraints: list[dict[str, Any]] = field(default_factory=list)
    prompt: str = ""
    feedback_prompts: list[str] = field(default_factory=list)
    minor_revision_prompts: list[str] = field(default_factory=list)
    validation_result: dict[str, Any] = field(default_factory=dict)
    execution_result: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    revision_rounds: int = 0
    generation_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sql_id": self.sql_id,
            "database_id": self.database_id,
            "city": self.city,
            "difficulty_level": self.difficulty_level,
            "sql": self.sql,
            "reasoning_summary": self.reasoning_summary,
            "used_tables": list(self.used_tables),
            "used_columns": list(self.used_columns),
            "used_spatial_functions": list(self.used_spatial_functions),
            "structural_constraints": stable_jsonify(self.structural_constraints),
            "spatial_function_constraints": stable_jsonify(self.spatial_function_constraints),
            "prompt": self.prompt,
            "feedback_prompts": list(self.feedback_prompts),
            "minor_revision_prompts": list(self.minor_revision_prompts),
            "validation_result": stable_jsonify(self.validation_result),
            "execution_result": stable_jsonify(self.execution_result),
            "metadata": stable_jsonify(self.metadata),
            "revision_rounds": self.revision_rounds,
            "generation_metadata": stable_jsonify(self.generation_metadata),
        }
