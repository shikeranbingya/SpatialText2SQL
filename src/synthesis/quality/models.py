"""Structured data models for NL-SQL quality control."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.synthesis.database.utils import stable_jsonify, to_text


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


@dataclass(frozen=True)
class ColumnSchema:
    column_name: str
    column_type: str = ""
    data_type: str = ""
    udt_name: str = ""
    spatial_type: str = ""
    geometry_type: str = ""
    srid: int | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ColumnSchema":
        srid = payload.get("srid")
        try:
            srid_value = int(srid) if srid not in (None, "") else None
        except (TypeError, ValueError):
            srid_value = None
        return cls(
            column_name=to_text(payload.get("column_name") or payload.get("canonical_name") or payload.get("name")),
            column_type=to_text(payload.get("column_type") or payload.get("canonical_type") or payload.get("type")),
            data_type=to_text(payload.get("data_type")).lower(),
            udt_name=to_text(payload.get("udt_name")).lower(),
            spatial_type=to_text(payload.get("spatial_type")).lower(),
            geometry_type=to_text(payload.get("geometry_type")).upper(),
            srid=srid_value,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "column_name": self.column_name,
            "column_type": self.column_type,
            "data_type": self.data_type,
            "udt_name": self.udt_name,
            "spatial_type": self.spatial_type,
            "geometry_type": self.geometry_type,
            "srid": self.srid,
        }


@dataclass(frozen=True)
class TableSchema:
    table_name: str
    columns: dict[str, ColumnSchema] = field(default_factory=dict)
    representative_values: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(
        cls,
        table_name: str,
        *,
        columns: Sequence[Mapping[str, Any]],
        representative_values: Mapping[str, Any] | None = None,
    ) -> "TableSchema":
        return cls(
            table_name=table_name,
            columns={
                column.column_name: column
                for column in (ColumnSchema.from_dict(item) for item in columns if isinstance(item, Mapping))
                if column.column_name
            },
            representative_values=_as_mapping(representative_values),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "table_name": self.table_name,
            "columns": {name: column.to_dict() for name, column in self.columns.items()},
            "representative_values": stable_jsonify(self.representative_values),
        }


@dataclass(frozen=True)
class DatabaseSchema:
    database_id: str
    tables: dict[str, TableSchema] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "database_id": self.database_id,
            "tables": {name: table.to_dict() for name, table in self.tables.items()},
        }


@dataclass
class NLSQLSample:
    sample_id: str
    database_id: str
    question: str
    sql: str
    difficulty_level: str = ""
    used_tables: list[str] = field(default_factory=list)
    used_columns: list[str] = field(default_factory=list)
    used_spatial_functions: list[str] = field(default_factory=list)
    linguistic_style: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NLSQLSample":
        sample_id = to_text(payload.get("sample_id") or payload.get("question_id") or payload.get("sql_id"))
        database_id = to_text(payload.get("database_id"))
        question = to_text(payload.get("question"))
        sql = to_text(payload.get("sql"))
        if not sample_id:
            raise ValueError("Missing required field: sample_id")
        if not database_id:
            raise ValueError("Missing required field: database_id")
        if not question:
            raise ValueError("Missing required field: question")
        if not sql:
            raise ValueError("Missing required field: sql")
        consumed = {
            "sample_id",
            "question_id",
            "sql_id",
            "database_id",
            "question",
            "sql",
            "difficulty_level",
            "source_difficulty_level",
            "used_tables",
            "used_columns",
            "used_spatial_functions",
            "linguistic_style",
            "style",
            "metadata",
        }
        metadata = _as_mapping(payload.get("metadata"))
        for key, value in payload.items():
            if key not in consumed and key not in metadata:
                metadata[str(key)] = stable_jsonify(value)
        return cls(
            sample_id=sample_id,
            database_id=database_id,
            question=question,
            sql=sql,
            difficulty_level=to_text(payload.get("difficulty_level") or payload.get("source_difficulty_level")),
            used_tables=_as_text_list(payload.get("used_tables")),
            used_columns=_as_text_list(payload.get("used_columns")),
            used_spatial_functions=_as_text_list(payload.get("used_spatial_functions")),
            linguistic_style=to_text(payload.get("linguistic_style") or payload.get("style")),
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "database_id": self.database_id,
            "question": self.question,
            "sql": self.sql,
            "difficulty_level": self.difficulty_level,
            "used_tables": list(self.used_tables),
            "used_columns": list(self.used_columns),
            "used_spatial_functions": list(self.used_spatial_functions),
            "linguistic_style": self.linguistic_style,
            "metadata": stable_jsonify(self.metadata),
        }


@dataclass
class ValidationResult:
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    execution_status: str = "not_run"
    row_count: int = 0
    result_preview: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "execution_status": self.execution_status,
            "row_count": self.row_count,
            "result_preview": stable_jsonify(self.result_preview),
        }


@dataclass
class QualityControlReport:
    total_samples: int
    passed_samples: int
    failed_samples: int
    failure_reasons: dict[str, int] = field(default_factory=dict)
    duplicate_count: int = 0
    distribution_by_difficulty: dict[str, int] = field(default_factory=dict)
    distribution_by_spatial_function: dict[str, int] = field(default_factory=dict)
    distribution_by_linguistic_style: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "passed_samples": self.passed_samples,
            "failed_samples": self.failed_samples,
            "failure_reasons": dict(self.failure_reasons),
            "duplicate_count": self.duplicate_count,
            "distribution_by_difficulty": dict(self.distribution_by_difficulty),
            "distribution_by_spatial_function": dict(self.distribution_by_spatial_function),
            "distribution_by_linguistic_style": dict(self.distribution_by_linguistic_style),
        }


@dataclass(frozen=True)
class ParsedSQL:
    sql: str
    statement_count: int
    tables: list[str]
    columns: list[str]
    aliases: dict[str, str]
    postgis_functions: list[str]
    aggregates: list[str]
    group_by_columns: list[str]
    order_by: list[dict[str, str]]
    limit: int | None
    spatial_predicates: list[str]
    distance_thresholds: list[str]
    filters: list[str]
    has_cte: bool
    has_subquery: bool
    string_literals: list[str] = field(default_factory=list)
    comparison_operators: list[str] = field(default_factory=list)
    function_calls: dict[str, list[str]] = field(default_factory=dict)

