"""Structured models for diversity-aware question generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.synthesis.database.models import SynthesizedSpatialDatabase
from src.synthesis.database.utils import stable_jsonify, to_text


QUESTION_STYLES = (
    "factual_lookup",
    "comparative_analysis",
    "aggregation_inquiry",
    "ranking_inquiry",
    "exploratory_analysis",
)


def _as_text_list(values: Any) -> list[str]:
    normalized = stable_jsonify(values)
    if normalized in (None, ""):
        return []
    if isinstance(normalized, str):
        return [normalized] if normalized else []
    if isinstance(normalized, Sequence) and not isinstance(normalized, (str, bytes)):
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
    if not isinstance(normalized, Sequence) or isinstance(normalized, (str, bytes)):
        return []
    rows: list[dict[str, Any]] = []
    for item in normalized:
        if isinstance(item, Mapping):
            rows.append({str(key): stable_jsonify(val) for key, val in item.items()})
    return rows


@dataclass(frozen=True)
class SQLQuestionSource:
    sql_id: str
    database_id: str
    city: str
    difficulty_level: str
    sql: str
    used_tables: list[str] = field(default_factory=list)
    used_columns: list[str] = field(default_factory=list)
    used_spatial_functions: list[str] = field(default_factory=list)
    structural_constraints: dict[str, Any] = field(default_factory=dict)
    spatial_function_constraints: list[dict[str, Any]] = field(default_factory=list)
    validation_result: dict[str, Any] = field(default_factory=dict)
    execution_result: dict[str, Any] = field(default_factory=dict)
    generation_metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SQLQuestionSource":
        sql_id = to_text(payload.get("sql_id"))
        database_id = to_text(payload.get("database_id"))
        city = to_text(payload.get("city"))
        difficulty_level = to_text(payload.get("difficulty_level"))
        sql = to_text(payload.get("sql"))
        if not sql_id:
            raise ValueError("Missing required field: sql_id")
        if not database_id:
            raise ValueError("Missing required field: database_id")
        if not city:
            raise ValueError("Missing required field: city")
        if not sql:
            raise ValueError("Missing required field: sql")
        return cls(
            sql_id=sql_id,
            database_id=database_id,
            city=city,
            difficulty_level=difficulty_level,
            sql=sql,
            used_tables=_as_text_list(payload.get("used_tables")),
            used_columns=_as_text_list(payload.get("used_columns")),
            used_spatial_functions=_as_text_list(payload.get("used_spatial_functions")),
            structural_constraints=_as_mapping(payload.get("structural_constraints")),
            spatial_function_constraints=_as_list_of_mappings(payload.get("spatial_function_constraints")),
            validation_result=_as_mapping(payload.get("validation_result")),
            execution_result=_as_mapping(payload.get("execution_result")),
            generation_metadata=_as_mapping(payload.get("generation_metadata")),
        )


@dataclass(frozen=True)
class QuestionGenerationContext:
    database_id: str
    city: str
    selected_table_names: list[str]
    schema: list[dict[str, Any]]
    representative_values: dict[str, Any]
    spatial_fields: list[dict[str, Any]]
    table_contexts: list[dict[str, Any]]

    @classmethod
    def from_database(cls, database: SynthesizedSpatialDatabase) -> "QuestionGenerationContext":
        table_contexts: list[dict[str, Any]] = []
        representative_values: dict[str, Any] = {}
        for table in database.selected_tables:
            table_name = to_text(table.table_name)
            table_contexts.append(
                {
                    "table_id": table.table_id,
                    "table_name": table_name,
                    "semantic_summary": table.semantic_summary,
                    "normalized_schema": stable_jsonify(table.normalized_schema),
                    "representative_values": stable_jsonify(table.representative_values),
                    "spatial_fields": stable_jsonify(table.spatial_fields),
                }
            )
            if table_name:
                representative_values[table_name] = stable_jsonify(table.representative_values)
        return cls(
            database_id=database.database_id,
            city=database.city,
            selected_table_names=list(database.selected_table_names),
            schema=stable_jsonify(database.schema),
            representative_values=stable_jsonify(representative_values or database.representative_values),
            spatial_fields=stable_jsonify(database.spatial_fields),
            table_contexts=table_contexts,
        )

    def to_prompt_payload(self) -> dict[str, Any]:
        return {
            "database_id": self.database_id,
            "city": self.city,
            "selected_table_names": list(self.selected_table_names),
            "schema": stable_jsonify(self.schema),
            "representative_values": stable_jsonify(self.representative_values),
            "spatial_fields": stable_jsonify(self.spatial_fields),
            "table_contexts": stable_jsonify(self.table_contexts),
        }


@dataclass(frozen=True)
class SpatialRelationConstraint:
    function_name: str
    preferred_phrase: str
    alternate_phrases: list[str] = field(default_factory=list)
    semantics_note: str = ""
    threshold: str = ""
    direction_note: str = ""
    required_keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "function_name": self.function_name,
            "preferred_phrase": self.preferred_phrase,
            "alternate_phrases": list(self.alternate_phrases),
            "semantics_note": self.semantics_note,
            "threshold": self.threshold,
            "direction_note": self.direction_note,
            "required_keywords": list(self.required_keywords),
        }


@dataclass(frozen=True)
class SQLFeatureSummary:
    tables: list[str] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    postgis_functions: list[str] = field(default_factory=list)
    aggregates: list[str] = field(default_factory=list)
    group_by_columns: list[str] = field(default_factory=list)
    order_by: list[dict[str, str]] = field(default_factory=list)
    limit: int | None = None
    spatial_predicates: list[str] = field(default_factory=list)
    distance_thresholds: list[str] = field(default_factory=list)
    filters: list[str] = field(default_factory=list)
    has_cte: bool = False
    has_subquery: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "tables": list(self.tables),
            "columns": list(self.columns),
            "postgis_functions": list(self.postgis_functions),
            "aggregates": list(self.aggregates),
            "group_by_columns": list(self.group_by_columns),
            "order_by": stable_jsonify(self.order_by),
            "limit": self.limit,
            "spatial_predicates": list(self.spatial_predicates),
            "distance_thresholds": list(self.distance_thresholds),
            "filters": list(self.filters),
            "has_cte": self.has_cte,
            "has_subquery": self.has_subquery,
        }


@dataclass
class QuestionGenerationCandidate:
    question: str
    style: str = ""
    reasoning_summary: str = ""
    spatial_phrases: list[str] = field(default_factory=list)
    raw_response_text: str = ""
    raw_response: Any = None
    parse_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "style": self.style,
            "reasoning_summary": self.reasoning_summary,
            "spatial_phrases": list(self.spatial_phrases),
            "raw_response_text": self.raw_response_text,
            "raw_response": stable_jsonify(self.raw_response),
            "parse_error": self.parse_error,
        }


@dataclass
class QuestionValidationResult:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    preserved_thresholds: list[str] = field(default_factory=list)
    detected_style_markers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "preserved_thresholds": list(self.preserved_thresholds),
            "detected_style_markers": list(self.detected_style_markers),
        }


@dataclass
class SynthesizedQuestion:
    question_id: str
    sql_id: str
    database_id: str
    city: str
    style: str
    question: str
    sql: str
    reasoning_summary: str = ""
    spatial_phrases: list[str] = field(default_factory=list)
    source_difficulty_level: str = ""
    used_tables: list[str] = field(default_factory=list)
    used_columns: list[str] = field(default_factory=list)
    used_spatial_functions: list[str] = field(default_factory=list)
    spatial_relation_constraints: list[dict[str, Any]] = field(default_factory=list)
    sql_features: dict[str, Any] = field(default_factory=dict)
    prompt: str = ""
    feedback_prompts: list[str] = field(default_factory=list)
    validation_result: dict[str, Any] = field(default_factory=dict)
    generation_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "sql_id": self.sql_id,
            "database_id": self.database_id,
            "city": self.city,
            "style": self.style,
            "question": self.question,
            "sql": self.sql,
            "reasoning_summary": self.reasoning_summary,
            "spatial_phrases": list(self.spatial_phrases),
            "source_difficulty_level": self.source_difficulty_level,
            "used_tables": list(self.used_tables),
            "used_columns": list(self.used_columns),
            "used_spatial_functions": list(self.used_spatial_functions),
            "spatial_relation_constraints": stable_jsonify(self.spatial_relation_constraints),
            "sql_features": stable_jsonify(self.sql_features),
            "prompt": self.prompt,
            "feedback_prompts": list(self.feedback_prompts),
            "validation_result": stable_jsonify(self.validation_result),
            "generation_metadata": stable_jsonify(self.generation_metadata),
        }
