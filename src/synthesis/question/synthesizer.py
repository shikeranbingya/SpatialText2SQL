"""Question synthesis entrypoint for executable spatial SQL."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.prompting.prompt_builder import PromptBuilder
from src.synthesis.database.utils import stable_jsonify

from .config import QuestionGenerationConfig
from .features import SQLFeatureExtractor
from .generator import QuestionLLMClient
from .models import (
    QuestionGenerationCandidate,
    QuestionGenerationContext,
    SQLQuestionSource,
    SynthesizedQuestion,
)
from .parser import parse_question_generation_response
from .style import STYLE_DESCRIPTIONS, SpatialPhraseSelector, StyleSelector
from .validator import QuestionValidationResult, QuestionValidator

LOGGER = logging.getLogger(__name__)


class DiversityAwareQuestionSynthesizer:
    def __init__(
        self,
        *,
        config: QuestionGenerationConfig,
        llm_client: QuestionLLMClient,
        prompt_builder: PromptBuilder,
        feature_extractor: SQLFeatureExtractor | None = None,
        style_selector: StyleSelector | None = None,
        spatial_phrase_selector: SpatialPhraseSelector | None = None,
        validator: QuestionValidator | None = None,
        existing_question_id_offsets: Mapping[str, int] | None = None,
    ) -> None:
        self.config = config
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder
        self.feature_extractor = feature_extractor or SQLFeatureExtractor()
        self.style_selector = style_selector or StyleSelector()
        self.spatial_phrase_selector = spatial_phrase_selector or SpatialPhraseSelector()
        self.validator = validator or QuestionValidator()
        self.rng = np.random.default_rng(self.config.generation.random_seed)
        self._question_id_offsets: dict[str, int] = dict(existing_question_id_offsets or {})

    def run(
        self,
        sql_queries: Sequence[SQLQuestionSource],
        context_by_database_id: Mapping[str, QuestionGenerationContext],
        on_row_generated: Callable[[SynthesizedQuestion], None] | None = None,
    ) -> list[SynthesizedQuestion]:
        rows: list[SynthesizedQuestion] = []
        for sql_query in sql_queries:
            context = context_by_database_id.get(sql_query.database_id)
            if context is None:
                LOGGER.warning(
                    "Skipping sql_id=%s because database context %s is missing.",
                    sql_query.sql_id,
                    sql_query.database_id,
                )
                continue
            rows.extend(self.run_for_sql(sql_query, context, on_row_generated=on_row_generated))
        return rows

    def run_for_sql(
        self,
        sql_query: SQLQuestionSource,
        context: QuestionGenerationContext,
        on_row_generated: Callable[[SynthesizedQuestion], None] | None = None,
    ) -> list[SynthesizedQuestion]:
        features = self.feature_extractor.extract(sql_query.sql)
        style_plan = self.style_selector.build_style_plan(
            features=features,
            total_questions=self.config.generation.num_questions_per_sql,
            rng=self.rng,
            fixed_style=self.config.generation.fixed_style,
            style_weights=self.config.generation.style_weights,
        )
        LOGGER.info(
            "Question synthesis plan | sql_id=%s | database_id=%s | styles=%s",
            sql_query.sql_id,
            sql_query.database_id,
            style_plan,
        )
        rows: list[SynthesizedQuestion] = []
        for question_index, style in enumerate(style_plan):
            spatial_constraints = self.spatial_phrase_selector.build_constraints(
                features=features,
                rng=self.rng,
            )
            row = self._run_single_question(
                sql_query=sql_query,
                context=context,
                question_index=question_index,
                style=style,
                features=features,
                spatial_constraints=spatial_constraints,
            )
            rows.append(row)
            if on_row_generated is not None:
                on_row_generated(row)
        return rows

    def _next_question_id(self, database_id: str) -> str:
        next_value = self._question_id_offsets.get(database_id, 0) + 1
        self._question_id_offsets[database_id] = next_value
        return f"{database_id}_{next_value:04d}"

    def _run_single_question(
        self,
        *,
        sql_query: SQLQuestionSource,
        context: QuestionGenerationContext,
        question_index: int,
        style: str,
        features,
        spatial_constraints,
    ) -> SynthesizedQuestion:
        sample_tag = f"{sql_query.sql_id}/q_{question_index + 1:03d}"
        prompt_build_start = time.perf_counter()
        prompt = self.prompt_builder.build_question_generation_prompt(
            sql_query=sql_query,
            database_context=context.to_prompt_payload(),
            sql_features=features.to_dict(),
            style_constraint={
                "style": style,
                "description": STYLE_DESCRIPTIONS.get(style, ""),
            },
            spatial_relation_constraints=[item.to_dict() for item in spatial_constraints],
        )
        prompt_build_ms = (time.perf_counter() - prompt_build_start) * 1000.0
        LOGGER.info(
            "Question prompt built | sample=%s | style=%s | prompt_chars=%s | build_time_ms=%.1f",
            sample_tag,
            style,
            len(prompt),
            prompt_build_ms,
        )
        feedback_prompts: list[str] = []
        generation_rounds: list[dict[str, Any]] = []
        candidate = QuestionGenerationCandidate(question="")
        validation_result = QuestionValidationResult(is_valid=False, errors=["Question generation did not start."])
        current_prompt = prompt

        LOGGER.info(
            "Question LLM prompt | sample=%s | round=%s/%s\n%s",
            sample_tag,
            1,
            1,
            current_prompt,
        )
        LOGGER.info(
            "Question LLM request start | sample=%s | round=%s/%s | style=%s | prompt_chars=%s",
            sample_tag,
            1,
            1,
            style,
            len(current_prompt),
        )
        generation_start = time.perf_counter()
        response = self.llm_client.generate(current_prompt)
        generation_ms = (time.perf_counter() - generation_start) * 1000.0
        LOGGER.info(
            "Question LLM request done | sample=%s | round=%s/%s | attempts=%s | response_chars=%s | time_ms=%.1f",
            sample_tag,
            1,
            1,
            response.attempts,
            len(response.text or ""),
            generation_ms,
        )
        candidate = parse_question_generation_response(
            response.text,
            raw_response=response.raw_response,
        )
        generation_rounds.append(
            {
                "round": 0,
                "prompt_type": "initial",
                "raw_response_text": candidate.raw_response_text,
                "parse_error": candidate.parse_error,
                "usage": stable_jsonify(response.usage),
                "attempts": response.attempts,
            }
        )
        if candidate.parse_error:
            LOGGER.warning(
                "Question candidate parse failed | sample=%s | round=%s/%s | error=%s",
                sample_tag,
                1,
                1,
                candidate.parse_error,
            )
            validation_result = QuestionValidationResult(
                is_valid=False,
                errors=[candidate.parse_error],
            )
        else:
            LOGGER.info(
                "Generated question | sample=%s | round=%s/%s\n%s",
                sample_tag,
                1,
                1,
                candidate.question,
            )
            validation_result = self.validator.validate(
                candidate=candidate,
                requested_style=style,
                sql_features=features,
                spatial_constraints=spatial_constraints,
            )
            LOGGER.info(
                "Question validation done | sample=%s | round=%s/%s | is_valid=%s | errors=%s | warnings=%s",
                sample_tag,
                1,
                1,
                validation_result.is_valid,
                len(validation_result.errors),
                len(validation_result.warnings),
            )

        synthesized = SynthesizedQuestion(
            question_id=self._next_question_id(sql_query.database_id),
            sql_id=sql_query.sql_id,
            database_id=sql_query.database_id,
            city=sql_query.city,
            style=style,
            question=candidate.question,
            sql=sql_query.sql,
            reasoning_summary=candidate.reasoning_summary,
            sql_reasoning_summary=sql_query.reasoning_summary,
            spatial_phrases=list(candidate.spatial_phrases),
            source_difficulty_level=sql_query.difficulty_level,
            used_tables=list(sql_query.used_tables or features.tables),
            used_columns=list(sql_query.used_columns or features.columns),
            used_spatial_functions=list(sql_query.used_spatial_functions or features.postgis_functions),
            spatial_relation_constraints=[item.to_dict() for item in spatial_constraints],
            sql_features=features.to_dict(),
            prompt=prompt,
            feedback_prompts=feedback_prompts,
            validation_result=validation_result.to_dict(),
            generation_metadata={
                "style": style,
                "style_description": STYLE_DESCRIPTIONS.get(style, ""),
                "generation_rounds": generation_rounds,
                "sql_difficulty": sql_query.difficulty_level,
                "success": validation_result.is_valid,
            },
        )
        return synthesized

    # Backward-compatible aliases
    generate_all = run
    generate_for_sql = run_for_sql
    _generate_single_question = _run_single_question


class DiversityAwareQuestionGenerator(DiversityAwareQuestionSynthesizer):
    """Backward-compatible alias for the legacy question generator name."""
