"""Constraint-guided SQL synthesis on top of synthesized spatial databases."""

from __future__ import annotations

from collections import Counter
import logging
import time
from typing import Any, Mapping, Sequence

import numpy as np

from src.prompting.prompt_builder import PromptBuilder
from src.synthesis.database.models import SynthesizedSpatialDatabase
from src.synthesis.database.utils import stable_jsonify, to_text

from .config import SQLSynthesisConfig
from .execution import SQLExecutionChecker
from .function_library import PostGISFunctionLibrary
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
from .validator import SQLValidator

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

    @staticmethod
    def _sample_tag(database: SynthesizedSpatialDatabase, sample_index: int) -> str:
        return f"{database.city}/{database.database_id}/sql_{sample_index + 1:04d}"

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
    ) -> list[SynthesizedSQLQuery]:
        rows: list[SynthesizedSQLQuery] = []
        for database in databases:
            rows.extend(self.synthesize_for_database(database))
        return rows

    def synthesize_for_database(
        self,
        database: SynthesizedSpatialDatabase,
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
        database_runtime_metadata = None
        if self.prompt_metadata_provider is not None:
            database_runtime_metadata = self.prompt_metadata_provider.load_database_metadata(database)
            if database_runtime_metadata:
                runtime_tables = database_runtime_metadata.get("tables", [])
                runtime_spatial_fields = sum(
                    len(item.get("spatial_fields", []))
                    for item in runtime_tables
                    if isinstance(item, Mapping)
                )
                LOGGER.info(
                    "Loaded live prompt metadata | city=%s | schema_id=%s | tables=%s | spatial_fields=%s",
                    database.city,
                    database.database_id,
                    len(runtime_tables),
                    runtime_spatial_fields,
                )
            else:
                LOGGER.warning(
                    "Falling back to file-derived prompt metadata | city=%s | schema_id=%s",
                    database.city,
                    database.database_id,
                )
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
            if difficulty_level != planned_difficulty:
                LOGGER.info(
                    "Difficulty adjusted for sampling | city=%s | schema_id=%s | planned=%s | actual=%s",
                    database.city,
                    database.database_id,
                    planned_difficulty,
                    difficulty_level,
                )
            structural_constraints = self._build_structural_constraints(difficulty_level, database)
            row = self._synthesize_single_query(
                database=database,
                sample_index=sample_index,
                difficulty_level=difficulty_level,
                structural_constraints=structural_constraints,
                sampled_functions=sampled_functions,
                database_runtime_metadata=database_runtime_metadata,
            )
            self._log_sample_progress(
                database=database,
                sample_index=sample_index,
                target_count=target_count,
                sampled_functions=sampled_functions,
                row=row,
                status="kept" if row is not None else "discarded",
            )
            if row is not None:
                output_rows.append(row)
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
    ) -> SynthesizedSQLQuery | None:
        sample_tag = self._sample_tag(database, sample_index)
        prompt_build_start = time.perf_counter()
        LOGGER.info(
            "SQL synthesis start | sample=%s | difficulty=%s | tables=%s | candidate_functions=%s",
            sample_tag,
            difficulty_level,
            len(database.selected_tables),
            self._format_function_names(sampled_functions),
        )
        prompt = self.prompt_builder.build_sql_synthesis_prompt(
            database=database,
            difficulty_level=difficulty_level,
            structural_constraints=dict(structural_constraints),
            sampled_functions=[item.to_dict() for item in sampled_functions],
            database_runtime_metadata=dict(database_runtime_metadata) if isinstance(database_runtime_metadata, Mapping) else None,
        )
        prompt_build_ms = (time.perf_counter() - prompt_build_start) * 1000.0
        LOGGER.info(
            "Prompt built | sample=%s | prompt_chars=%s | build_time_ms=%.1f",
            sample_tag,
            len(prompt),
            prompt_build_ms,
        )
        feedback_prompts: list[str] = []
        generation_rounds: list[dict[str, Any]] = []
        validation_result = SQLValidationResult(is_valid=False, errors=["SQL generation did not start."])
        execution_result = SQLExecutionResult(executed=False, success=not self.config.execution.enable_execution_check)
        candidate = SQLGenerationCandidate(sql="")
        current_prompt = prompt

        for revision_round in range(self.config.synthesis.max_revision_rounds + 1):
            LOGGER.info(
                "LLM prompt | sample=%s | round=%s/%s\n%s",
                sample_tag,
                revision_round + 1,
                self.config.synthesis.max_revision_rounds + 1,
                current_prompt,
            )
            LOGGER.info(
                "LLM request start | sample=%s | round=%s/%s | prompt_type=%s | prompt_chars=%s",
                sample_tag,
                revision_round + 1,
                self.config.synthesis.max_revision_rounds + 1,
                "initial" if revision_round == 0 else "feedback",
                len(current_prompt),
            )
            generation_start = time.perf_counter()
            generation_response = self.sql_generator.generate(current_prompt)
            generation_ms = (time.perf_counter() - generation_start) * 1000.0
            LOGGER.info(
                "LLM request done | sample=%s | round=%s/%s | attempts=%s | response_chars=%s | time_ms=%.1f",
                sample_tag,
                revision_round + 1,
                self.config.synthesis.max_revision_rounds + 1,
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
                    "round": revision_round,
                    "prompt_type": "initial" if revision_round == 0 else "feedback",
                    "raw_response_text": candidate.raw_response_text,
                    "parse_error": candidate.parse_error,
                    "usage": stable_jsonify(generation_response.usage),
                    "attempts": generation_response.attempts,
                }
            )

            if candidate.parse_error:
                LOGGER.warning(
                    "Candidate parse failed | sample=%s | round=%s/%s | error=%s",
                    sample_tag,
                    revision_round + 1,
                    self.config.synthesis.max_revision_rounds + 1,
                    candidate.parse_error,
                )
                validation_result = SQLValidationResult(
                    is_valid=False,
                    errors=[candidate.parse_error],
                )
                execution_result = SQLExecutionResult(executed=False, success=False, error_message="Skipped due to parse failure.")
            else:
                LOGGER.info(
                    "Generated SQL | sample=%s | round=%s/%s\n%s",
                    sample_tag,
                    revision_round + 1,
                    self.config.synthesis.max_revision_rounds + 1,
                    self._format_sql_for_log(candidate.sql),
                )
                LOGGER.info(
                    "Static validation start | sample=%s | round=%s/%s | sql_chars=%s",
                    sample_tag,
                    revision_round + 1,
                    self.config.synthesis.max_revision_rounds + 1,
                    len(candidate.sql or ""),
                )
                validation_start = time.perf_counter()
                validation_result = self.validator.validate(
                    sql=candidate.sql,
                    database=database,
                    sampled_functions=[item.function_name for item in sampled_functions],
                    difficulty_level=difficulty_level,
                    database_runtime_metadata=dict(database_runtime_metadata) if isinstance(database_runtime_metadata, Mapping) else None,
                )
                validation_ms = (time.perf_counter() - validation_start) * 1000.0
                LOGGER.info(
                    "Static validation done | sample=%s | round=%s/%s | is_valid=%s | errors=%s | detected_tables=%s | detected_spatial_functions=%s | time_ms=%.1f",
                    sample_tag,
                    revision_round + 1,
                    self.config.synthesis.max_revision_rounds + 1,
                    validation_result.is_valid,
                    len(validation_result.errors),
                    len(validation_result.detected_tables),
                    ", ".join(validation_result.detected_spatial_functions) or "<none>",
                    validation_ms,
                )
                if validation_result.is_valid:
                    LOGGER.info(
                        "Execution check start | sample=%s | round=%s/%s | dry_run=%s | explain_only=%s",
                        sample_tag,
                        revision_round + 1,
                        self.config.synthesis.max_revision_rounds + 1,
                        self.config.execution.dry_run,
                        self.config.execution.explain_only,
                    )
                    execution_result = self.execution_checker.check(candidate.sql, database)
                    LOGGER.info(
                        "Execution check done | sample=%s | round=%s/%s | success=%s | executed=%s | empty_result=%s | row_count=%s | time_ms=%s | error=%s",
                        sample_tag,
                        revision_round + 1,
                        self.config.synthesis.max_revision_rounds + 1,
                        execution_result.success,
                        execution_result.executed,
                        execution_result.empty_result,
                        execution_result.row_count,
                        execution_result.execution_time_ms,
                        execution_result.error_message or "",
                    )
                    if not execution_result.success:
                        LOGGER.warning(
                            "Execution failed | sample=%s | round=%s/%s | error=%s\n%s",
                            sample_tag,
                            revision_round + 1,
                            self.config.synthesis.max_revision_rounds + 1,
                            execution_result.error_message or "",
                            self._format_sql_for_log(candidate.sql),
                        )
                else:
                    LOGGER.warning(
                        "Static validation failed | sample=%s | round=%s/%s | errors=%s\n%s",
                        sample_tag,
                        revision_round + 1,
                        self.config.synthesis.max_revision_rounds + 1,
                        " | ".join(validation_result.errors),
                        self._format_sql_for_log(candidate.sql),
                    )
                    execution_result = SQLExecutionResult(
                        executed=False,
                        success=False,
                        error_message="Skipped execution because static validation failed.",
                    )

            if self._is_sample_success(validation_result, execution_result):
                LOGGER.info(
                    "Sample succeeded | sample=%s | round=%s/%s\n%s",
                    sample_tag,
                    revision_round + 1,
                    self.config.synthesis.max_revision_rounds + 1,
                    self._format_sql_for_log(candidate.sql),
                )
                break
            if revision_round >= self.config.synthesis.max_revision_rounds:
                LOGGER.info(
                    "Sample exhausted revisions | sample=%s | final_round=%s/%s",
                    sample_tag,
                    revision_round + 1,
                    self.config.synthesis.max_revision_rounds + 1,
                )
                break

            feedback_prompt = self.prompt_builder.build_sql_feedback_prompt(
                database=database,
                difficulty_level=difficulty_level,
                structural_constraints=dict(structural_constraints),
                sampled_functions=[item.to_dict() for item in sampled_functions],
                original_candidate=candidate.to_dict(),
                validation_errors=list(validation_result.errors),
                execution_error=execution_result.error_message,
                empty_result=execution_result.empty_result,
                database_runtime_metadata=dict(database_runtime_metadata) if isinstance(database_runtime_metadata, Mapping) else None,
            )
            feedback_prompts.append(feedback_prompt)
            LOGGER.info(
                "Feedback prompt built | sample=%s | next_round=%s/%s | prompt_chars=%s | validation_errors=%s | execution_error=%s | empty_result=%s",
                sample_tag,
                revision_round + 2,
                self.config.synthesis.max_revision_rounds + 1,
                len(feedback_prompt),
                len(validation_result.errors),
                execution_result.error_message or "",
                execution_result.empty_result,
            )
            current_prompt = feedback_prompt

        synthesized = SynthesizedSQLQuery(
            sql_id=f"{database.database_id}_{sample_index + 1:04d}",
            database_id=database.database_id,
            city=database.city,
            difficulty_level=difficulty_level,
            sql=candidate.sql,
            used_tables=candidate.used_tables or validation_result.detected_tables,
            used_columns=candidate.used_columns or validation_result.detected_columns,
            used_spatial_functions=candidate.used_spatial_functions or validation_result.detected_spatial_functions,
            structural_constraints=dict(structural_constraints),
            spatial_function_constraints=[item.to_dict() for item in sampled_functions],
            prompt=prompt,
            feedback_prompts=feedback_prompts,
            validation_result=validation_result.to_dict(),
            execution_result=execution_result.to_dict(),
            revision_rounds=len(feedback_prompts),
            generation_metadata={
                "provider": self.config.llm.provider,
                "model": self.config.llm.model,
                "database_table_count": len(database.selected_tables),
                "sampled_function_names": [item.function_name for item in sampled_functions],
                "sampled_function_signatures": [item.signature for item in sampled_functions],
                "generation_rounds": generation_rounds,
                "success": self._is_sample_success(validation_result, execution_result),
            },
        )
        if self._should_keep_sample(validation_result, execution_result):
            return synthesized
        return None

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
                    "min_tables": 1,
                    "max_tables": 1,
                    "require_join": False,
                    "allow_group_by": False,
                    "allow_subquery": False,
                    "allow_cte": False,
                }
            )
        elif difficulty_level == "medium":
            constraints.update(
                {
                    "min_tables": 2,
                    "max_tables": 2,
                    "require_join": True,
                    "allow_group_by": True,
                    "allow_subquery": False,
                    "allow_cte": False,
                }
            )
        elif difficulty_level == "hard":
            constraints.update(
                {
                    "min_tables": 3,
                    "max_tables": len(database.selected_tables),
                    "require_join": True,
                    "min_joins": 2,
                    "allow_group_by": True,
                    "allow_subquery": True,
                    "allow_cte": False,
                }
            )
        else:
            constraints.update(
                {
                    "min_tables": 3,
                    "max_tables": len(database.selected_tables),
                    "require_join": True,
                    "min_joins": 2,
                    "allow_group_by": True,
                    "allow_subquery": True,
                    "allow_cte": True,
                    "require_complex_structure": True,
                }
            )
        return constraints

    def _is_sample_success(
        self,
        validation_result: SQLValidationResult,
        execution_result: SQLExecutionResult,
    ) -> bool:
        if not validation_result.is_valid:
            return False
        if not self.config.execution.enable_execution_check or self.config.execution.dry_run:
            return True
        return execution_result.success

    def _should_keep_sample(
        self,
        validation_result: SQLValidationResult,
        execution_result: SQLExecutionResult,
    ) -> bool:
        if not validation_result.is_valid:
            return self.config.synthesis.keep_invalid
        if self.config.execution.enable_execution_check and not execution_result.success:
            return self.config.synthesis.keep_failed_execution
        return True
