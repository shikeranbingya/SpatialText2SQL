"""Constraint-guided SQL synthesis on top of synthesized spatial databases."""

from __future__ import annotations

from collections import Counter
import logging
import re
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.prompting.prompt_builder import PromptBuilder
from src.synthesis.database.models import SynthesizedSpatialDatabase
from src.synthesis.database.utils import stable_jsonify, to_text

from .config import SQLSynthesisConfig
from .execution import SQLExecutionChecker
from .function_library import PostGISFunctionLibrary, build_required_function_constraints
from .generator import SQLGenerator
from .models import (
    DIFFICULTY_LEVELS,
    SQLExecutionResult,
    SQLGenerationCandidate,
    SQLValidationResult,
    SynthesizedSQLQuery,
)
from .parser import parse_sql_generation_response
from .prompt_metadata import PostGISPromptMetadataProvider
from .validator import SQLValidator, contains_dangerous_sql

LOGGER = logging.getLogger(__name__)


class ConstraintGuidedSQLSynthesizer:
    def __init__(
        self,
        *,
        config: SQLSynthesisConfig,
        function_library: PostGISFunctionLibrary,
        sql_generator: SQLGenerator,
        prompt_builder: PromptBuilder,
        validator: SQLValidator | None = None,
        execution_checker: SQLExecutionChecker | None = None,
        prompt_metadata_provider: PostGISPromptMetadataProvider | None = None,
        existing_sql_id_offsets: Mapping[str, int] | None = None,
    ) -> None:
        self.config = config
        self.function_library = function_library
        self.sql_generator = sql_generator
        self.prompt_builder = prompt_builder
        self.validator = validator or SQLValidator(function_library)
        self.execution_checker = execution_checker or SQLExecutionChecker(
            config.database,
            config.execution,
        )
        self.prompt_metadata_provider = prompt_metadata_provider
        self.rng = np.random.default_rng(self.config.synthesis.random_seed)
        self.sql_id_offsets = {
            to_text(database_id): max(int(offset), 0)
            for database_id, offset in (existing_sql_id_offsets or {}).items()
            if to_text(database_id)
        }
        self.last_run_stats = self._empty_run_stats()

    @staticmethod
    def _sample_tag(database: SynthesizedSpatialDatabase, sample_index: int) -> str:
        return f"{database.city}/{database.database_id}/sql_{sample_index + 1:04d}"

    def _next_sql_id(self, database_id: str) -> str:
        next_value = self.sql_id_offsets.get(database_id, 0) + 1
        self.sql_id_offsets[database_id] = next_value
        return f"{database_id}_{next_value:04d}"

    @staticmethod
    def _extract_srid(crs_value: Any) -> int | None:
        match = re.search(r"(\d+)", to_text(crs_value))
        return int(match.group(1)) if match else None

    def _fallback_create_table_ddl(self, table: Any, schema_name: str) -> str:
        spatial_field_by_name = {
            to_text(field.get("canonical_name")).lower(): field
            for field in (getattr(table, "spatial_fields", None) or [])
            if isinstance(field, Mapping) and to_text(field.get("canonical_name"))
        }
        column_lines: list[str] = []
        for column in getattr(table, "normalized_schema", []) or []:
            if not isinstance(column, Mapping):
                continue
            column_name = to_text(column.get("canonical_name") or column.get("name"))
            if not column_name:
                continue
            column_type = to_text(column.get("canonical_type") or column.get("type") or "text")
            if column_type.lower() == "spatial":
                field = spatial_field_by_name.get(column_name.lower(), {})
                srid = self._extract_srid(field.get("crs"))
                geometry_type = "GEOMETRY"
                spatial_type = "geometry"
                if srid is not None:
                    column_type = f"{spatial_type}({geometry_type},{srid})"
                else:
                    column_type = spatial_type
            column_lines.append(f"    {column_name} {column_type}")
        body = ",\n".join(column_lines)
        del schema_name
        return f"CREATE TABLE {table.table_name} (\n{body}\n);"

    def _build_row_metadata(
        self,
        *,
        database: SynthesizedSpatialDatabase,
        database_runtime_metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if isinstance(database_runtime_metadata, Mapping) and database_runtime_metadata.get("tables"):
            raw_tables = [
                stable_jsonify(item)
                for item in database_runtime_metadata.get("tables", [])
                if isinstance(item, Mapping)
            ]
            schema_ddls = [
                to_text(item.get("create_table_ddl"))
                for item in raw_tables
                if to_text(item.get("create_table_ddl"))
            ]
            schema_name = to_text(database_runtime_metadata.get("schema_name"))
            return {
                "database_context": {
                    "database_id": database.database_id,
                    "city": database.city,
                    "schema_name": schema_name,
                    "selected_table_names": list(database.selected_table_names),
                    "schema_ddls": schema_ddls,
                    "tables": raw_tables,
                }
            }

        schema_name = database.database_id
        fallback_tables = []
        schema_ddls: list[str] = []
        for table in database.selected_tables:
            create_table_ddl = self._fallback_create_table_ddl(table, schema_name)
            schema_ddls.append(create_table_ddl)
            fallback_tables.append(
                {
                    "table_name": table.table_name,
                    "create_table_ddl": create_table_ddl,
                    "columns": [
                        {
                            "column_name": to_text(column.get("canonical_name") or column.get("name")),
                            "column_type": to_text(column.get("canonical_type") or column.get("type") or "text"),
                        }
                        for column in (table.normalized_schema or [])
                        if isinstance(column, Mapping) and to_text(column.get("canonical_name") or column.get("name"))
                    ],
                    "spatial_fields": stable_jsonify(table.spatial_fields),
                    "representative_values": stable_jsonify(table.representative_values),
                }
            )
        return {
            "database_context": {
                "database_id": database.database_id,
                "city": database.city,
                "schema_name": schema_name,
                "selected_table_names": list(database.selected_table_names),
                "schema_ddls": schema_ddls,
                "tables": fallback_tables,
            }
        }

    @staticmethod
    def _format_function_names(sampled_functions: Sequence[Any]) -> str:
        names = [to_text(getattr(item, "function_name", "")) for item in sampled_functions]
        names = [name for name in names if name]
        return ", ".join(names) if names else "<none>"

    @staticmethod
    def _format_sql_for_log(sql_text: str) -> str:
        sql_text = (sql_text or "").strip()
        return sql_text or "<empty>"

    def _log_sample_progress(
        self,
        *,
        database: SynthesizedSpatialDatabase,
        sample_index: int,
        target_count: int,
        sampled_functions: Sequence[Any],
        row: SynthesizedSQLQuery | None,
        status: str,
    ) -> None:
        spatial_functions = row.used_spatial_functions if row is not None and row.used_spatial_functions else []
        spatial_function_text = (
            ", ".join(spatial_functions)
            if spatial_functions
            else self._format_function_names(sampled_functions)
        )
        LOGGER.info(
            "SQL synthesis progress %s/%s | city=%s | schema_id=%s | spatial_functions=%s | status=%s",
            sample_index + 1,
            target_count,
            database.city,
            database.database_id,
            spatial_function_text,
            status,
        )

    def synthesize_all(
        self,
        databases: Sequence[SynthesizedSpatialDatabase],
        on_row_generated: Callable[[SynthesizedSQLQuery], None] | None = None,
    ) -> list[SynthesizedSQLQuery]:
        self.last_run_stats = self._empty_run_stats()
        rows: list[SynthesizedSQLQuery] = []
        for database in databases:
            rows.extend(self.synthesize_for_database(database, on_row_generated=on_row_generated))
        return rows

    def synthesize_for_database(
        self,
        database: SynthesizedSpatialDatabase,
        on_row_generated: Callable[[SynthesizedSQLQuery], None] | None = None,
    ) -> list[SynthesizedSQLQuery]:
        if not database.spatial_fields:
            LOGGER.warning("Skipping database %s because it has no spatial fields.", database.database_id)
            return []
        target_count = self._resolve_num_sql_per_database(database.city)
        if target_count <= 0:
            LOGGER.info(
                "Skipping database %s because num_sql_per_database is 0 for city=%s.",
                database.database_id,
                database.city,
            )
            return []
        output_rows: list[SynthesizedSQLQuery] = []
        difficulty_plan = self._build_difficulty_plan(database, target_count)
        difficulty_counts = Counter(difficulty_plan)
        LOGGER.info(
            "Difficulty plan | city=%s | schema_id=%s | target_count=%s | counts=%s",
            database.city,
            database.database_id,
            target_count,
            {level: difficulty_counts.get(level, 0) for level in DIFFICULTY_LEVELS},
        )
        for sample_index, planned_difficulty in enumerate(difficulty_plan):
            difficulty_level, sampled_functions = self._resolve_sampled_functions(
                database=database,
                planned_difficulty=planned_difficulty,
            )
            if not sampled_functions:
                LOGGER.warning(
                    "SQL synthesis progress %s/%s | city=%s | schema_id=%s | planned_difficulty=%s | spatial_functions=<none> | status=no-compatible-functions",
                    sample_index + 1,
                    target_count,
                    database.city,
                    database.database_id,
                    planned_difficulty,
                )
                continue
            prompt_database = self._select_prompt_database(database, difficulty_level)
            database_runtime_metadata = None
            if self.prompt_metadata_provider is not None:
                database_runtime_metadata = self.prompt_metadata_provider.load_database_metadata(prompt_database)
                if database_runtime_metadata:
                    runtime_tables = database_runtime_metadata.get("tables", [])
                    runtime_spatial_fields = sum(
                        len(item.get("spatial_fields", []))
                        for item in runtime_tables
                        if isinstance(item, Mapping)
                    )
                    LOGGER.info(
                        "Loaded live prompt metadata | city=%s | schema_id=%s | prompt_tables=%s | spatial_fields=%s",
                        prompt_database.city,
                        prompt_database.database_id,
                        len(runtime_tables),
                        runtime_spatial_fields,
                    )
                else:
                    LOGGER.warning(
                        "Falling back to file-derived prompt metadata | city=%s | schema_id=%s | prompt_tables=%s",
                        prompt_database.city,
                        prompt_database.database_id,
                        len(prompt_database.selected_tables),
                    )
            if difficulty_level != planned_difficulty:
                LOGGER.info(
                    "Difficulty adjusted for sampling | city=%s | schema_id=%s | planned=%s | actual=%s | prompt_tables=%s",
                    prompt_database.city,
                    prompt_database.database_id,
                    planned_difficulty,
                    difficulty_level,
                    len(prompt_database.selected_tables),
                )
            structural_constraints = self._build_structural_constraints(difficulty_level, prompt_database)
            row = self._synthesize_single_query(
                database=prompt_database,
                sample_index=sample_index,
                difficulty_level=difficulty_level,
                structural_constraints=structural_constraints,
                sampled_functions=sampled_functions,
                database_runtime_metadata=database_runtime_metadata,
                source_database_table_count=len(database.selected_tables),
            )
            self._record_run_outcome(difficulty_level=difficulty_level, kept=row is not None)
            self._log_sample_progress(
                database=prompt_database,
                sample_index=sample_index,
                target_count=target_count,
                sampled_functions=sampled_functions,
                row=row,
                status="kept" if row is not None else "discarded",
            )
            if row is not None:
                output_rows.append(row)
                if on_row_generated is not None:
                    on_row_generated(row)
        return output_rows

    def _synthesize_single_query(
        self,
        *,
        database: SynthesizedSpatialDatabase,
        sample_index: int,
        difficulty_level: str,
        structural_constraints: Mapping[str, Any],
        sampled_functions: Sequence[Any],
        database_runtime_metadata: Mapping[str, Any] | None,
        source_database_table_count: int,
    ) -> SynthesizedSQLQuery | None:
        sample_tag = self._sample_tag(database, sample_index)
        prompt_build_start = time.perf_counter()
        LOGGER.info(
            "SQL synthesis start | sample=%s | difficulty=%s | prompt_tables=%s | candidate_functions=%s",
            sample_tag,
            difficulty_level,
            len(database.selected_tables),
            self._format_function_names(sampled_functions),
        )
        required_functions = build_required_function_constraints(sampled_functions, difficulty_level)
        prompt = self.prompt_builder.build_sql_synthesis_prompt(
            database=database,
            difficulty_level=difficulty_level,
            structural_constraints=dict(structural_constraints),
            sampled_functions=required_functions,
            database_runtime_metadata=dict(database_runtime_metadata) if isinstance(database_runtime_metadata, Mapping) else None,
        )
        prompt_build_ms = (time.perf_counter() - prompt_build_start) * 1000.0
        LOGGER.info(
            "Prompt built | sample=%s | prompt_chars=%s | build_time_ms=%.1f",
            sample_tag,
            len(prompt),
            prompt_build_ms,
        )
        minor_revision_prompts: list[str] = []
        generation_rounds: list[dict[str, Any]] = []
        validation_result = SQLValidationResult(is_valid=False, errors=["SQL generation did not start."])
        execution_result = SQLExecutionResult(executed=False, success=not self.config.execution.enable_execution_check)
        candidate = SQLGenerationCandidate(sql="")
        LOGGER.info(
            "LLM prompt | sample=%s | round=1/1\n%s",
            sample_tag,
            prompt,
        )
        LOGGER.info(
            "LLM request start | sample=%s | round=1/1 | prompt_type=initial | prompt_chars=%s",
            sample_tag,
            len(prompt),
        )
        generation_start = time.perf_counter()
        generation_response = self.sql_generator.generate(prompt)
        generation_ms = (time.perf_counter() - generation_start) * 1000.0
        LOGGER.info(
            "LLM request done | sample=%s | round=1/1 | attempts=%s | response_chars=%s | time_ms=%.1f",
            sample_tag,
            generation_response.attempts,
            len(generation_response.text or ""),
            generation_ms,
        )
        candidate = parse_sql_generation_response(
            generation_response.text,
            raw_response=generation_response.raw_response,
        )
        generation_rounds.append(
            {
                "round": 0,
                "prompt_type": "initial",
                "raw_response_text": candidate.raw_response_text,
                "parse_error": candidate.parse_error,
                "usage": stable_jsonify(generation_response.usage),
                "attempts": generation_response.attempts,
            }
        )
        has_sql = bool(to_text(candidate.sql).strip())

        if candidate.parse_error:
            LOGGER.warning(
                "Candidate parse failed | sample=%s | round=1/1 | error=%s",
                sample_tag,
                candidate.parse_error,
            )
            validation_result = SQLValidationResult(
                is_valid=False,
                errors=[candidate.parse_error],
            )
            execution_result = SQLExecutionResult(executed=False, success=False, error_message="Skipped due to parse failure.")
        else:
            LOGGER.info(
                "Generated SQL | sample=%s | round=1/1\n%s",
                sample_tag,
                self._format_sql_for_log(candidate.sql),
            )
            LOGGER.info(
                "Static validation start | sample=%s | round=1/1 | sql_chars=%s",
                sample_tag,
                len(candidate.sql or ""),
            )
            validation_start = time.perf_counter()
            validation_result = self.validator.validate(
                sql=candidate.sql,
                database=database,
                sampled_functions=[to_text(item.get("function_name")) for item in required_functions],
                difficulty_level=difficulty_level,
                database_runtime_metadata=dict(database_runtime_metadata) if isinstance(database_runtime_metadata, Mapping) else None,
            )
            validation_ms = (time.perf_counter() - validation_start) * 1000.0
            LOGGER.info(
                "Static validation done | sample=%s | round=1/1 | is_valid=%s | errors=%s | detected_tables=%s | detected_spatial_functions=%s | time_ms=%.1f",
                sample_tag,
                validation_result.is_valid,
                len(validation_result.errors),
                len(validation_result.detected_tables),
                ", ".join(validation_result.detected_spatial_functions) or "<none>",
                validation_ms,
            )
            if not validation_result.is_valid:
                LOGGER.warning(
                    "Static validation produced warnings before execution | sample=%s | round=1/1 | errors=%s\n%s",
                    sample_tag,
                    " | ".join(validation_result.errors),
                    self._format_sql_for_log(candidate.sql),
                )
            LOGGER.info(
                "Execution check start | sample=%s | round=1/1 | dry_run=%s | explain_only=%s",
                sample_tag,
                self.config.execution.dry_run,
                self.config.execution.explain_only,
            )
            execution_result = self.execution_checker.check(candidate.sql, database)
            LOGGER.info(
                "Execution check done | sample=%s | round=1/1 | success=%s | executed=%s | empty_result=%s | row_count=%s | time_ms=%s | error=%s",
                sample_tag,
                execution_result.success,
                execution_result.executed,
                execution_result.empty_result,
                execution_result.row_count,
                execution_result.execution_time_ms,
                execution_result.error_message or "",
            )
            if not execution_result.success:
                LOGGER.warning(
                    "Execution failed before minor revision decision | sample=%s | round=1/1 | error=%s\n%s",
                    sample_tag,
                    execution_result.error_message or "",
                    self._format_sql_for_log(candidate.sql),
                )

        sample_success = self._is_sample_success(
            has_sql=has_sql,
            has_dangerous_sql=has_sql and contains_dangerous_sql(candidate.sql),
            execution_result=execution_result,
        )
        minor_revision_applied = False
        if not sample_success and self._should_attempt_minor_revision(execution_result):
            involved_tables = candidate.used_tables or validation_result.detected_tables
            minor_revision_prompt = self.prompt_builder.build_sql_revision_prompt(
                database=database,
                original_sql=candidate.sql,
                execution_error=execution_result.error_message,
                used_tables=involved_tables,
                database_runtime_metadata=dict(database_runtime_metadata) if isinstance(database_runtime_metadata, Mapping) else None,
            )
            minor_revision_prompts.append(minor_revision_prompt)
            minor_revision_applied = True
            LOGGER.info(
                "Minor revision prompt built | sample=%s | round=2/2 | prompt_chars=%s | involved_tables=%s | error=%s",
                sample_tag,
                len(minor_revision_prompt),
                ", ".join(involved_tables) or "<none>",
                execution_result.error_message or "",
            )
            LOGGER.info(
                "LLM prompt | sample=%s | round=2/2\n%s",
                sample_tag,
                minor_revision_prompt,
            )
            LOGGER.info(
                "LLM request start | sample=%s | round=2/2 | prompt_type=minor_revision | prompt_chars=%s",
                sample_tag,
                len(minor_revision_prompt),
            )
            generation_start = time.perf_counter()
            generation_response = self.sql_generator.generate(minor_revision_prompt)
            generation_ms = (time.perf_counter() - generation_start) * 1000.0
            LOGGER.info(
                "LLM request done | sample=%s | round=2/2 | attempts=%s | response_chars=%s | time_ms=%.1f",
                sample_tag,
                generation_response.attempts,
                len(generation_response.text or ""),
                generation_ms,
            )
            revised_candidate = parse_sql_generation_response(
                generation_response.text,
                raw_response=generation_response.raw_response,
            )
            generation_rounds.append(
                {
                    "round": 1,
                    "prompt_type": "minor_revision",
                    "raw_response_text": revised_candidate.raw_response_text,
                    "parse_error": revised_candidate.parse_error,
                    "usage": stable_jsonify(generation_response.usage),
                    "attempts": generation_response.attempts,
                }
            )
            candidate = revised_candidate
            has_sql = bool(to_text(candidate.sql).strip())
            if candidate.parse_error:
                LOGGER.warning(
                    "Minor revision parse failed | sample=%s | round=2/2 | error=%s",
                    sample_tag,
                    candidate.parse_error,
                )
                validation_result = SQLValidationResult(
                    is_valid=False,
                    errors=[candidate.parse_error],
                )
                execution_result = SQLExecutionResult(
                    executed=False,
                    success=False,
                    error_message="Skipped due to parse failure after minor revision.",
                )
            else:
                LOGGER.info(
                    "Minor revision SQL | sample=%s | round=2/2\n%s",
                    sample_tag,
                    self._format_sql_for_log(candidate.sql),
                )
                LOGGER.info(
                    "Static validation start | sample=%s | round=2/2 | sql_chars=%s",
                    sample_tag,
                    len(candidate.sql or ""),
                )
                validation_start = time.perf_counter()
                validation_result = self.validator.validate(
                    sql=candidate.sql,
                    database=database,
                    sampled_functions=[to_text(item.get("function_name")) for item in required_functions],
                    difficulty_level=difficulty_level,
                    database_runtime_metadata=dict(database_runtime_metadata) if isinstance(database_runtime_metadata, Mapping) else None,
                )
                validation_ms = (time.perf_counter() - validation_start) * 1000.0
                LOGGER.info(
                    "Static validation done | sample=%s | round=2/2 | is_valid=%s | errors=%s | detected_tables=%s | detected_spatial_functions=%s | time_ms=%.1f",
                    sample_tag,
                    validation_result.is_valid,
                    len(validation_result.errors),
                    len(validation_result.detected_tables),
                    ", ".join(validation_result.detected_spatial_functions) or "<none>",
                    validation_ms,
                )
                if not validation_result.is_valid:
                    LOGGER.warning(
                        "Minor revision validation produced warnings before execution | sample=%s | round=2/2 | errors=%s\n%s",
                        sample_tag,
                        " | ".join(validation_result.errors),
                        self._format_sql_for_log(candidate.sql),
                    )
                LOGGER.info(
                    "Execution check start | sample=%s | round=2/2 | dry_run=%s | explain_only=%s",
                    sample_tag,
                    self.config.execution.dry_run,
                    self.config.execution.explain_only,
                )
                execution_result = self.execution_checker.check(candidate.sql, database)
                LOGGER.info(
                    "Execution check done | sample=%s | round=2/2 | success=%s | executed=%s | empty_result=%s | row_count=%s | time_ms=%s | error=%s",
                    sample_tag,
                    execution_result.success,
                    execution_result.executed,
                    execution_result.empty_result,
                    execution_result.row_count,
                    execution_result.execution_time_ms,
                    execution_result.error_message or "",
                )
            sample_success = self._is_sample_success(
                has_sql=has_sql,
                has_dangerous_sql=has_sql and contains_dangerous_sql(candidate.sql),
                execution_result=execution_result,
            )
        if sample_success:
            retained_with_warning = (
                not validation_result.is_valid
            )
            if retained_with_warning:
                LOGGER.warning(
                    "Sample retained with warning | sample=%s | validation_ok=%s | execution_ok=%s",
                    sample_tag,
                    validation_result.is_valid,
                    execution_result.success,
                )
            LOGGER.info(
                "Sample succeeded | sample=%s | round=%s\n%s",
                sample_tag,
                "2/2" if minor_revision_applied else "1/1",
                self._format_sql_for_log(candidate.sql),
            )
        else:
            LOGGER.info(
                "Sample discarded | sample=%s | minor_revision_applied=%s | validation_ok=%s | execution_ok=%s",
                sample_tag,
                minor_revision_applied,
                validation_result.is_valid,
                execution_result.success,
            )

        if not sample_success:
            return None

        synthesized = SynthesizedSQLQuery(
            sql_id=self._next_sql_id(database.database_id),
            database_id=database.database_id,
            city=database.city,
            difficulty_level=difficulty_level,
            sql=candidate.sql,
            reasoning_summary=candidate.reasoning_summary,
            used_tables=candidate.used_tables or validation_result.detected_tables,
            used_columns=candidate.used_columns or validation_result.detected_columns,
            used_spatial_functions=candidate.used_spatial_functions or validation_result.detected_spatial_functions,
            structural_constraints=dict(structural_constraints),
            spatial_function_constraints=required_functions,
            prompt=prompt,
            feedback_prompts=[],
            minor_revision_prompts=minor_revision_prompts,
            validation_result=validation_result.to_dict(),
            execution_result=execution_result.to_dict(),
            metadata=self._build_row_metadata(
                database=database,
                database_runtime_metadata=database_runtime_metadata,
            ),
            revision_rounds=len(minor_revision_prompts),
            generation_metadata={
                "provider": self.config.llm.provider,
                "model": self.config.llm.model,
                "database_table_count": source_database_table_count,
                "prompt_table_count": len(database.selected_tables),
                "prompt_table_names": list(database.selected_table_names),
                "sampled_function_names": [item.function_name for item in sampled_functions],
                "sampled_function_signatures": [item.signature for item in sampled_functions],
                "required_function_names": [to_text(item.get("function_name")) for item in required_functions],
                "required_function_signatures": [to_text(item.get("signature")) for item in required_functions],
                "generation_rounds": generation_rounds,
                "minor_revision_applied": minor_revision_applied,
                "retained_with_warning": not validation_result.is_valid,
                "success": sample_success,
            },
        )
        return synthesized

    @staticmethod
    def _empty_run_stats() -> dict[str, Any]:
        return {
            "generated_total": 0,
            "retained_total": 0,
            "generated_by_difficulty": {level: 0 for level in DIFFICULTY_LEVELS},
            "retained_by_difficulty": {level: 0 for level in DIFFICULTY_LEVELS},
        }

    def _record_run_outcome(self, *, difficulty_level: str, kept: bool) -> None:
        self.last_run_stats["generated_total"] += 1
        self.last_run_stats["generated_by_difficulty"][difficulty_level] += 1
        if kept:
            self.last_run_stats["retained_total"] += 1
            self.last_run_stats["retained_by_difficulty"][difficulty_level] += 1

    def get_run_stats(self) -> dict[str, Any]:
        return stable_jsonify(self.last_run_stats)

    def _resolve_num_sql_per_database(self, city: str) -> int:
        config_value = self.config.synthesis.num_sql_per_database
        if isinstance(config_value, Mapping):
            city_key = to_text(city).lower()
            if city_key in config_value:
                return max(int(config_value[city_key]), 0)
            if "default" in config_value:
                return max(int(config_value["default"]), 0)
            return 0
        return max(int(config_value), 0)

    def _build_difficulty_plan(
        self,
        database: SynthesizedSpatialDatabase,
        target_count: int,
    ) -> list[str]:
        if target_count <= 0:
            return []
        fixed = to_text(self.config.synthesis.fixed_difficulty).lower()
        if fixed:
            planned = self._downgrade_difficulty_if_needed(fixed, database)
            return [planned] * target_count

        weights = dict(self.config.synthesis.difficulty_weights)
        for level in DIFFICULTY_LEVELS:
            downgraded = self._downgrade_difficulty_if_needed(level, database)
            if downgraded != level:
                weights[level] = 0.0
        if sum(weights.values()) <= 0:
            fallback = self._downgrade_difficulty_if_needed("easy", database)
            return [fallback] * target_count
        difficulty_counts = self._allocate_difficulty_counts(target_count, weights)
        plan: list[str] = []
        for level in DIFFICULTY_LEVELS:
            plan.extend([level] * difficulty_counts.get(level, 0))
        return plan

    def _resolve_sampled_functions(
        self,
        *,
        database: SynthesizedSpatialDatabase,
        planned_difficulty: str,
    ) -> tuple[str, list[Any]]:
        try:
            base_index = DIFFICULTY_LEVELS.index(planned_difficulty)
        except ValueError:
            base_index = 0
        candidate_levels = list(DIFFICULTY_LEVELS[base_index:]) + list(reversed(DIFFICULTY_LEVELS[:base_index]))
        for difficulty_level in candidate_levels:
            sampled_functions = self.function_library.sample_functions(
                database=database,
                difficulty_level=difficulty_level,
                rng=self.rng,
            )
            if sampled_functions:
                return difficulty_level, sampled_functions
        return planned_difficulty, []

    @staticmethod
    def _allocate_difficulty_counts(
        target_count: int,
        weights: Mapping[str, float],
    ) -> dict[str, int]:
        if target_count <= 0:
            return {level: 0 for level in DIFFICULTY_LEVELS}
        positive_weights = np.array([max(float(weights.get(level, 0.0)), 0.0) for level in DIFFICULTY_LEVELS], dtype=float)
        total_weight = float(positive_weights.sum())
        if total_weight <= 0.0:
            return {level: 0 for level in DIFFICULTY_LEVELS}
        raw_counts = positive_weights / total_weight * float(target_count)
        base_counts = np.floor(raw_counts).astype(int)
        remainder = target_count - int(base_counts.sum())
        if remainder > 0:
            fractional = raw_counts - base_counts
            ranked_indices = sorted(
                range(len(DIFFICULTY_LEVELS)),
                key=lambda idx: (-fractional[idx], idx),
            )
            for idx in ranked_indices[:remainder]:
                base_counts[idx] += 1
        return {
            level: int(base_counts[idx])
            for idx, level in enumerate(DIFFICULTY_LEVELS)
        }

    def _select_prompt_database(
        self,
        database: SynthesizedSpatialDatabase,
        difficulty_level: str,
    ) -> SynthesizedSpatialDatabase:
        selected_tables = list(database.selected_tables)
        if len(selected_tables) <= 1:
            return database
        target_count = self._resolve_prompt_table_count(difficulty_level, len(selected_tables))
        if target_count >= len(selected_tables):
            return database

        ranked_indices = sorted(
            range(len(selected_tables)),
            key=lambda idx: (
                -int(bool(selected_tables[idx].spatial_fields)),
                -len(getattr(selected_tables[idx], "normalized_schema", []) or []),
                idx,
            ),
        )
        chosen_indices = sorted(ranked_indices[:target_count])
        prompt_tables = [selected_tables[idx] for idx in chosen_indices]
        LOGGER.info(
            "Prompt table subset selected | city=%s | schema_id=%s | difficulty=%s | source_tables=%s | prompt_tables=%s | chosen=%s",
            database.city,
            database.database_id,
            difficulty_level,
            len(selected_tables),
            len(prompt_tables),
            ", ".join(table.table_name for table in prompt_tables),
        )
        return SynthesizedSpatialDatabase.from_selected_tables(
            database_id=database.database_id,
            city=database.city,
            selected_tables=prompt_tables,
            sampling_trace=list(database.sampling_trace),
            graph_stats=dict(database.graph_stats),
            synthesize_config=dict(database.synthesize_config),
        )

    @staticmethod
    def _resolve_prompt_table_count(
        difficulty_level: str,
        available_table_count: int,
    ) -> int:
        if difficulty_level == "easy":
            target_count = 1
        elif difficulty_level == "medium":
            target_count = 2
        elif difficulty_level == "hard":
            target_count = 3
        else:
            target_count = 4
        return max(1, min(target_count, available_table_count))

    @staticmethod
    def _downgrade_difficulty_if_needed(difficulty: str, database: SynthesizedSpatialDatabase) -> str:
        table_count = len(database.selected_tables)
        if difficulty == "easy":
            return "easy"
        if difficulty == "medium":
            return "medium" if table_count >= 2 else "easy"
        if difficulty == "hard":
            if table_count >= 3:
                return "hard"
            if table_count >= 2:
                return "medium"
            return "easy"
        if difficulty == "extra-hard":
            if table_count >= 3:
                return "extra-hard"
            if table_count >= 2:
                return "medium"
            return "easy"
        return "easy"

    @staticmethod
    def _build_structural_constraints(
        difficulty_level: str,
        database: SynthesizedSpatialDatabase,
    ) -> dict[str, Any]:
        constraints = {
            "difficulty_level": difficulty_level,
            "database_id": database.database_id,
            "table_count_available": len(database.selected_tables),
            "must_use_spatial_function": True,
            "must_be_read_only": True,
        }
        if difficulty_level == "easy":
            constraints.update(
                {
                    "difficulty_summary": "Single-table spatial filter or lookup.",
                    "min_tables": 1,
                    "max_tables": 1,
                    "min_spatial_joins": 0,
                    "max_spatial_joins": 0,
                    "require_join": False,
                    "allow_group_by": False,
                    "allow_subquery": False,
                    "allow_cte": False,
                }
            )
        elif difficulty_level == "medium":
            constraints.update(
                {
                    "difficulty_summary": "Two-table query with exactly one spatial join.",
                    "min_tables": 2,
                    "max_tables": 2,
                    "min_joins": 1,
                    "max_joins": 1,
                    "min_spatial_joins": 1,
                    "max_spatial_joins": 1,
                    "require_join": True,
                    "allow_group_by": True,
                    "allow_subquery": False,
                    "allow_cte": False,
                }
            )
        elif difficulty_level == "hard":
            constraints.update(
                {
                    "difficulty_summary": "Three-table query with exactly two spatial joins.",
                    "min_tables": 3,
                    "max_tables": 3,
                    "require_join": True,
                    "min_joins": 2,
                    "max_joins": 2,
                    "min_spatial_joins": 2,
                    "max_spatial_joins": 2,
                    "allow_group_by": True,
                    "allow_subquery": False,
                    "allow_cte": False,
                }
            )
        else:
            constraints.update(
                {
                    "min_tables": 3,
                    "max_tables": min(4, len(database.selected_tables)),
                    "difficulty_summary": "Three-to-four-table query with bounded advanced structure.",
                    "require_join": True,
                    "min_spatial_joins": 1,
                    "allow_group_by": True,
                    "allow_subquery": True,
                    "allow_cte": True,
                    "require_complex_structure": True,
                    "min_advanced_ops": 2,
                    "max_advanced_ops": 4,
                    "advanced_op_definition": "Count each spatial join and each nested query (subquery or CTE) as one operation.",
                    "allow_set_operation": False,
                    "prefer_minimal_complexity": True,
                }
            )
        return constraints

    def _is_sample_success(
        self,
        *,
        has_sql: bool,
        has_dangerous_sql: bool,
        execution_result: SQLExecutionResult,
    ) -> bool:
        if not has_sql:
            return False
        if has_dangerous_sql:
            return False
        if not self.config.execution.enable_execution_check or self.config.execution.dry_run:
            return True
        return execution_result.success

    @staticmethod
    def _is_timeout_error(error_message: str) -> bool:
        lowered = to_text(error_message).lower()
        if not lowered:
            return False
        return "timeout" in lowered or "timed out" in lowered

    def _should_attempt_minor_revision(self, execution_result: SQLExecutionResult) -> bool:
        return (
            execution_result.executed
            and not execution_result.success
            and not self._is_timeout_error(execution_result.error_message)
        )
