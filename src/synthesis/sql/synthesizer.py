"""Constraint-guided SQL synthesis on top of synthesized spatial databases."""

from __future__ import annotations

import logging
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
        self.rng = np.random.default_rng(self.config.synthesis.random_seed)

    @staticmethod
    def _format_function_names(sampled_functions: Sequence[Any]) -> str:
        names = [to_text(getattr(item, "function_name", "")) for item in sampled_functions]
        names = [name for name in names if name]
        return ", ".join(names) if names else "<none>"

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
        output_rows: list[SynthesizedSQLQuery] = []
        for sample_index in range(target_count):
            difficulty_level = self._sample_difficulty(database)
            structural_constraints = self._build_structural_constraints(difficulty_level, database)
            sampled_functions = self.function_library.sample_functions(
                database=database,
                difficulty_level=difficulty_level,
                rng=self.rng,
            )
            if not sampled_functions:
                LOGGER.warning(
                    "SQL synthesis progress %s/%s | city=%s | schema_id=%s | spatial_functions=<none> | status=no-compatible-functions",
                    sample_index + 1,
                    target_count,
                    database.city,
                    database.database_id,
                )
                continue
            row = self._synthesize_single_query(
                database=database,
                sample_index=sample_index,
                difficulty_level=difficulty_level,
                structural_constraints=structural_constraints,
                sampled_functions=sampled_functions,
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
    ) -> SynthesizedSQLQuery | None:
        prompt = self.prompt_builder.build_sql_synthesis_prompt(
            database=database,
            difficulty_level=difficulty_level,
            structural_constraints=dict(structural_constraints),
            sampled_functions=[item.to_dict() for item in sampled_functions],
        )
        feedback_prompts: list[str] = []
        generation_rounds: list[dict[str, Any]] = []
        validation_result = SQLValidationResult(is_valid=False, errors=["SQL generation did not start."])
        execution_result = SQLExecutionResult(executed=False, success=not self.config.execution.enable_execution_check)
        candidate = SQLGenerationCandidate(sql="")
        current_prompt = prompt

        for revision_round in range(self.config.synthesis.max_revision_rounds + 1):
            generation_response = self.sql_generator.generate(current_prompt)
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
                validation_result = SQLValidationResult(
                    is_valid=False,
                    errors=[candidate.parse_error],
                )
                execution_result = SQLExecutionResult(executed=False, success=False, error_message="Skipped due to parse failure.")
            else:
                validation_result = self.validator.validate(
                    sql=candidate.sql,
                    database=database,
                    sampled_functions=[item.function_name for item in sampled_functions],
                    difficulty_level=difficulty_level,
                )
                if validation_result.is_valid:
                    execution_result = self.execution_checker.check(candidate.sql, database)
                else:
                    execution_result = SQLExecutionResult(
                        executed=False,
                        success=False,
                        error_message="Skipped execution because static validation failed.",
                    )

            if self._is_sample_success(validation_result, execution_result):
                break
            if revision_round >= self.config.synthesis.max_revision_rounds:
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
            )
            feedback_prompts.append(feedback_prompt)
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

    def _sample_difficulty(self, database: SynthesizedSpatialDatabase) -> str:
        fixed = to_text(self.config.synthesis.fixed_difficulty).lower()
        if fixed:
            return self._downgrade_difficulty_if_needed(fixed, database)

        weights = dict(self.config.synthesis.difficulty_weights)
        for level in DIFFICULTY_LEVELS:
            downgraded = self._downgrade_difficulty_if_needed(level, database)
            if downgraded != level:
                weights[level] = 0.0
        if sum(weights.values()) <= 0:
            return self._downgrade_difficulty_if_needed("easy", database)
        levels = list(DIFFICULTY_LEVELS)
        probabilities = np.array([max(weights[level], 0.0) for level in levels], dtype=float)
        probabilities = probabilities / probabilities.sum()
        sampled = levels[int(self.rng.choice(len(levels), p=probabilities))]
        return self._downgrade_difficulty_if_needed(sampled, database)

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
