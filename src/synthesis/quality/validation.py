"""Validation logic for NL-SQL samples."""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Iterable, Mapping, Sequence

from src.synthesis.database.utils import to_text
from src.synthesis.question.validator import (
    AGGREGATE_MARKERS,
    COMPARATIVE_MARKERS,
    EXPLORATORY_MARKERS,
    GROUPING_MARKERS,
    RANKING_MARKERS,
)
from src.synthesis.sql.function_library import PostGISFunctionLibrary

from .analysis import (
    DefaultSQLAnalyzer,
    SQLAnalyzer,
    contains_disallowed_sql,
    split_top_level_commas,
)
from .config import QualityControlConfig
from .models import DatabaseSchema, NLSQLSample, ParsedSQL, ValidationResult
from .registry import DatabaseClient

LOGGER = logging.getLogger(__name__)


SPATIAL_KEYWORDS_BY_FUNCTION: dict[str, list[str]] = {
    "st_contains": ["contain", "inside", "enclose"],
    "st_within": ["within", "inside", "located in"],
    "st_intersects": ["intersect", "overlap", "cross"],
    "st_dwithin": ["within", "near", "at most", "no more than", "distance"],
    "st_distance": ["distance", "far", "how far"],
    "st_buffer": ["buffer", "within", "around"],
    "st_area": ["area", "size", "surface"],
    "st_length": ["length", "long"],
    "st_centroid": ["centroid", "center", "centre"],
    "st_touches": ["touch", "boundary", "edge"],
}
ALL_SPATIAL_KEYWORDS = sorted({item for values in SPATIAL_KEYWORDS_BY_FUNCTION.values() for item in values})
RAW_POSTGIS_PATTERN = re.compile(r"\bST_[A-Za-z0-9_]+\b")


def _normalize_identifier_phrase(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[_\-]+", " ", value.lower())).strip()


def _normalize_question_text(value: str) -> str:
    lowered = value.lower()
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _question_contains_phrase(question: str, phrase: str) -> bool:
    normalized_question = f" {_normalize_question_text(question)} "
    normalized_phrase = _normalize_identifier_phrase(phrase)
    if not normalized_phrase:
        return False
    return f" {normalized_phrase} " in normalized_question


def _extract_argument_column_refs(args_text: str) -> list[str]:
    refs: list[str] = []
    for arg in split_top_level_commas(args_text):
        for match in re.finditer(r"\b(?:[a-zA-Z_][\w]*\.)?([a-zA-Z_][\w]*)\b", arg):
            token = match.group(1)
            if token.lower().startswith("st_"):
                continue
            refs.append(token)
    return refs


@dataclass
class SampleValidationArtifact:
    parsed_sql: ParsedSQL
    validation_result: ValidationResult


@dataclass
class SQLSampleValidator:
    function_library: PostGISFunctionLibrary
    sql_analyzer: SQLAnalyzer = field(default_factory=DefaultSQLAnalyzer)

    def validate(
        self,
        *,
        sample: NLSQLSample,
        schema: DatabaseSchema,
        database_client: DatabaseClient,
        config: QualityControlConfig,
    ) -> SampleValidationArtifact:
        parsed_sql = self.sql_analyzer.analyze(sample.sql)
        errors: list[str] = []
        warnings: list[str] = []

        if parsed_sql.statement_count != 1:
            errors.append("SQL must contain exactly one statement.")
        if contains_disallowed_sql(parsed_sql.sql):
            errors.append("SQL contains disallowed non-read-only or administrative operations.")
        if not re.match(r"^\s*(select|with)\b", parsed_sql.sql, re.I):
            errors.append("SQL must start with SELECT or WITH.")

        unknown_tables = [table for table in parsed_sql.tables if table not in schema.tables]
        if unknown_tables:
            errors.append(f"Unknown tables referenced: {', '.join(sorted(set(unknown_tables)))}")

        allowed_columns = {
            column_name
            for table in schema.tables.values()
            for column_name in table.columns.keys()
        }
        unknown_columns = [column for column in parsed_sql.columns if column not in allowed_columns]
        if unknown_columns:
            errors.append(f"Unknown columns referenced: {', '.join(sorted(set(unknown_columns)))}")

        if not parsed_sql.postgis_functions:
            errors.append("SQL does not use any PostGIS function.")

        for function_name in parsed_sql.postgis_functions:
            signatures = self.function_library.get_function_signatures(function_name)
            if not signatures:
                errors.append(f"Disallowed or unknown PostGIS function: {function_name}")
                continue
            self._validate_spatial_argument_compatibility(
                function_name=function_name,
                parsed_sql=parsed_sql,
                schema=schema,
                signatures=signatures,
                errors=errors,
                warnings=warnings,
            )

        execution_status = "not_run"
        row_count = 0
        preview: list[dict[str, object]] = []
        if not errors:
            try:
                row_count, preview = database_client.execute_read_only(
                    parsed_sql.sql,
                    max_preview_rows=config.run.max_result_rows,
                )
                execution_status = "passed"
                if row_count == 0 and not config.run.allow_empty_result:
                    errors.append("SQL executed successfully but returned no rows.")
                    execution_status = "empty_result"
            except Exception as exc:
                execution_status = "execution_failed"
                errors.append(f"Execution failed: {exc}")

        validation_result = ValidationResult(
            passed=not errors,
            errors=errors,
            warnings=warnings,
            execution_status=execution_status,
            row_count=row_count,
            result_preview=preview,
        )
        return SampleValidationArtifact(parsed_sql=parsed_sql, validation_result=validation_result)

    @staticmethod
    def _validate_spatial_argument_compatibility(
        *,
        function_name: str,
        parsed_sql: ParsedSQL,
        schema: DatabaseSchema,
        signatures,
        errors: list[str],
        warnings: list[str],
    ) -> None:
        geography_supported = any(
            any("geography" in arg.lower() for arg in signature.input_args)
            for signature in signatures
        )
        geometry_supported = any(
            any("geometry" in arg.lower() for arg in signature.input_args)
            for signature in signatures
        )
        for args_text in parsed_sql.function_calls.get(function_name, []):
            refs = _extract_argument_column_refs(args_text)
            for column_name in refs:
                column_schema = None
                for table in schema.tables.values():
                    if column_name in table.columns:
                        column_schema = table.columns[column_name]
                        break
                if column_schema is None:
                    continue
                raw_args = args_text.lower()
                if column_schema.spatial_type == "geography" and not geography_supported and "::geometry" not in raw_args:
                    errors.append(
                        f"{function_name} uses geography column '{column_name}' without a compatible geography signature or explicit cast."
                    )
                elif column_schema.spatial_type == "geometry" and not geometry_supported and geography_supported and "::geography" not in raw_args:
                    errors.append(
                        f"{function_name} uses geometry column '{column_name}' without a compatible geometry signature or explicit cast."
                    )
                elif not column_schema.spatial_type and "spatial" in column_schema.column_type.lower():
                    warnings.append(
                        f"Could not fully verify spatial type compatibility for {function_name} on column '{column_name}'."
                    )


@dataclass
class SemanticConsistencyChecker:
    def check(
        self,
        *,
        sample: NLSQLSample,
        parsed_sql: ParsedSQL,
        schema: DatabaseSchema,
        config: QualityControlConfig,
    ) -> tuple[list[str], list[str]]:
        errors: list[str] = []
        warnings: list[str] = []
        question = sample.question
        lowered = f" {_normalize_question_text(question)} "

        if RAW_POSTGIS_PATTERN.search(question) and not config.semantic.debug_mode:
            self._add_issue(
                "Question exposes raw PostGIS function names.",
                errors,
                warnings,
                config,
            )

        if parsed_sql.distance_thresholds:
            for threshold in parsed_sql.distance_thresholds:
                if threshold not in question:
                    self._add_issue(
                        f"Question does not preserve the distance threshold {threshold}.",
                        errors,
                        warnings,
                        config,
                    )

        if parsed_sql.aggregates:
            required_markers = {
                marker
                for aggregate in parsed_sql.aggregates
                for marker in AGGREGATE_MARKERS.get(aggregate.upper(), [])
            }
            if required_markers and not any(marker in lowered for marker in required_markers):
                self._add_issue(
                    "Question misses aggregation semantics present in SQL.",
                    errors,
                    warnings,
                    config,
                )
        elif any(marker in lowered for markers in AGGREGATE_MARKERS.values() for marker in markers):
            self._add_issue(
                "Question introduces aggregation semantics that are not present in SQL.",
                errors,
                warnings,
                config,
            )

        if parsed_sql.group_by_columns:
            if not any(marker in lowered for marker in GROUPING_MARKERS):
                self._add_issue(
                    "Question misses GROUP BY semantics present in SQL.",
                    errors,
                    warnings,
                    config,
                )
        elif any(marker in lowered for marker in GROUPING_MARKERS):
            self._add_issue(
                "Question introduces grouping semantics that are not present in SQL.",
                errors,
                warnings,
                config,
            )

        if parsed_sql.order_by or parsed_sql.limit is not None:
            if not any(marker in lowered for marker in RANKING_MARKERS):
                self._add_issue(
                    "Question misses ranking or top-k semantics present in SQL.",
                    errors,
                    warnings,
                    config,
                )
            if parsed_sql.limit is not None and str(parsed_sql.limit) not in question:
                self._add_issue(
                    f"Question does not preserve the LIMIT/top-k value {parsed_sql.limit}.",
                    errors,
                    warnings,
                    config,
                )
        elif any(marker in lowered for marker in RANKING_MARKERS):
            self._add_issue(
                "Question introduces ranking or top-k semantics that are not present in SQL.",
                errors,
                warnings,
                config,
            )

        if parsed_sql.comparison_operators and any(marker in lowered for marker in COMPARATIVE_MARKERS):
            pass
        elif parsed_sql.comparison_operators and not any(marker in lowered for marker in COMPARATIVE_MARKERS):
            warnings.append("Question may be missing explicit comparative language.")
        elif any(marker in lowered for marker in COMPARATIVE_MARKERS):
            warnings.append("Question may introduce unsupported comparative wording.")

        if parsed_sql.has_cte or parsed_sql.has_subquery:
            if not any(marker in lowered for marker in EXPLORATORY_MARKERS):
                warnings.append("Question may not reflect the analytical complexity of the SQL.")

        supported_spatial_keywords = {
            keyword
            for function_name in parsed_sql.postgis_functions
            for keyword in SPATIAL_KEYWORDS_BY_FUNCTION.get(function_name.lower(), [])
        }
        if supported_spatial_keywords and not any(keyword in lowered for keyword in supported_spatial_keywords):
            self._add_issue(
                "Question misses explicit spatial relation wording required by the SQL.",
                errors,
                warnings,
                config,
            )
        unsupported_spatial_keywords = [
            keyword for keyword in ALL_SPATIAL_KEYWORDS
            if keyword in lowered and keyword not in supported_spatial_keywords
        ]
        if unsupported_spatial_keywords:
            self._add_issue(
                f"Question introduces unsupported spatial relation wording: {', '.join(sorted(set(unsupported_spatial_keywords)))}",
                errors,
                warnings,
                config,
            )

        used_tables = {table.lower() for table in sample.used_tables or parsed_sql.tables}
        all_tables = {table_name.lower() for table_name in schema.tables.keys()}
        for table_name in all_tables - used_tables:
            if _question_contains_phrase(question, table_name):
                self._add_issue(
                    f"Question references table/entity '{table_name}' that is not used by the SQL.",
                    errors,
                    warnings,
                    config,
                )

        used_columns = {column.lower() for column in sample.used_columns or parsed_sql.columns}
        all_columns = {
            column_name.lower()
            for table in schema.tables.values()
            for column_name in table.columns.keys()
        }
        for column_name in all_columns - used_columns:
            if _question_contains_phrase(question, column_name):
                self._add_issue(
                    f"Question references attribute '{column_name}' that is not used by the SQL.",
                    errors,
                    warnings,
                    config,
                )

        string_literals = [literal.replace("''", "'").strip().lower() for literal in parsed_sql.string_literals if literal.strip()]
        for literal in string_literals:
            if len(literal) >= 3 and literal not in lowered:
                warnings.append(f"Question may be missing literal filter value '{literal}'.")

        return errors, warnings

    @staticmethod
    def _add_issue(
        message: str,
        errors: list[str],
        warnings: list[str],
        config: QualityControlConfig,
    ) -> None:
        if config.semantic.mode == "strict":
            errors.append(message)
        else:
            warnings.append(message)


def build_distribution(samples: Sequence[NLSQLSample], bucket_getter) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for sample in samples:
        for bucket in bucket_getter(sample):
            if bucket:
                counter[bucket] += 1
    return dict(counter)


def question_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, _normalize_question_text(left), _normalize_question_text(right)).ratio()

