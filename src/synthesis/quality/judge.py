"""LLM-based self-consistency judging for NL-SQL quality control."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.prompting.prompt_builder import PromptBuilder
from src.synthesis.database.utils import stable_jsonify, to_text

from .config import QualityControlConfig
from .generator import QualityControlLLMClient
from .models import DatabaseSchema, NLSQLSample, ParsedSQL, ValidationResult

LOGGER = logging.getLogger(__name__)

ALLOWED_REASON_CODES = {
    "ok",
    "wrong_entity",
    "wrong_filter",
    "wrong_aggregation",
    "wrong_grouping",
    "wrong_ranking",
    "wrong_threshold",
    "wrong_spatial_relation",
    "wrong_comparison",
    "execution_contradiction",
    "underspecified_question",
    "extra_requirement",
    "raw_sql_leak",
    "low_confidence_consensus",
    "invalid_judge_output",
    "other",
}

JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S | re.I)


@dataclass
class SelfConsistencyVote:
    decision: str
    confidence: str
    reason_codes: list[str] = field(default_factory=list)
    parse_error: str = ""
    raw_response_text: str = ""
    raw_response: Any = None
    usage: dict[str, Any] | None = None

    @property
    def passed(self) -> bool:
        return self.decision == "pass" and not self.parse_error

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "confidence": self.confidence,
            "reason_codes": list(self.reason_codes),
            "parse_error": self.parse_error,
            "raw_response_text": self.raw_response_text,
            "raw_response": stable_jsonify(self.raw_response),
            "usage": stable_jsonify(self.usage),
        }


@dataclass
class SelfConsistencyJudgment:
    passed: bool
    pass_votes: int
    fail_votes: int
    reason_codes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    rounds: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "pass_votes": self.pass_votes,
            "fail_votes": self.fail_votes,
            "reason_codes": list(self.reason_codes),
            "warnings": list(self.warnings),
            "rounds": stable_jsonify(self.rounds),
        }


class SelfConsistencyQualityJudge:
    def __init__(
        self,
        *,
        llm_client: QualityControlLLMClient,
        prompt_builder: PromptBuilder,
    ) -> None:
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder

    def judge(
        self,
        *,
        sample: NLSQLSample,
        schema: DatabaseSchema,
        parsed_sql: ParsedSQL,
        validation_result: ValidationResult,
        config: QualityControlConfig,
    ) -> SelfConsistencyJudgment:
        prompt = self.prompt_builder.build_quality_control_prompt(
            sample=self._build_sample_payload(sample, parsed_sql),
            schema_lines=self._build_schema_lines(schema),
            sql_feature_summary=self._build_sql_feature_summary(parsed_sql),
            execution_summary=self._build_execution_summary(sample, validation_result),
            representative_values=self._build_representative_values(schema, sample),
            judge_rules=self._build_judge_rules(config),
        )

        rounds: list[SelfConsistencyVote] = []
        for round_index in range(config.judge.self_consistency_rounds):
            LOGGER.info(
                "Quality-control self-consistency request | sample_id=%s | round=%s/%s | prompt_chars=%s",
                sample.sample_id,
                round_index + 1,
                config.judge.self_consistency_rounds,
                len(prompt),
            )
            response = self.llm_client.generate(prompt)
            vote = self._parse_vote(
                response.text,
                raw_response=response.raw_response,
                usage=response.usage,
                max_reason_codes=config.judge.max_reason_codes,
            )
            rounds.append(vote)

        pass_votes = sum(1 for vote in rounds if vote.passed)
        fail_votes = len(rounds) - pass_votes
        fail_reason_counter: Counter[str] = Counter()
        warnings: list[str] = []
        pass_confidences = [vote.confidence for vote in rounds if vote.passed]
        for vote in rounds:
            if vote.parse_error:
                fail_reason_counter["invalid_judge_output"] += 1
            for code in vote.reason_codes:
                if code != "ok":
                    fail_reason_counter[code] += 1

        passed = pass_votes >= config.judge.min_pass_votes
        if config.judge.require_high_confidence and passed:
            passed = any(confidence == "high" for confidence in pass_confidences)
            if not passed:
                fail_reason_counter["low_confidence_consensus"] += 1

        if 0 < fail_votes < len(rounds):
            warnings.append(f"mixed_self_consistency_votes:{pass_votes}/{len(rounds)}")

        if passed:
            reason_codes = ["ok"]
        else:
            ordered = [code for code, _count in fail_reason_counter.most_common(config.judge.max_reason_codes)]
            reason_codes = ordered or ["other"]

        return SelfConsistencyJudgment(
            passed=passed,
            pass_votes=pass_votes,
            fail_votes=fail_votes,
            reason_codes=reason_codes,
            warnings=warnings,
            rounds=[vote.to_dict() for vote in rounds],
        )

    @staticmethod
    def _parse_vote(
        text: str,
        *,
        raw_response: Any,
        usage: dict[str, Any] | None,
        max_reason_codes: int,
    ) -> SelfConsistencyVote:
        candidate = text.strip()
        fenced = JSON_BLOCK_PATTERN.search(candidate)
        if fenced:
            candidate = fenced.group(1).strip()
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            return SelfConsistencyVote(
                decision="fail",
                confidence="low",
                reason_codes=["invalid_judge_output"],
                parse_error=f"Invalid judge JSON: {exc}",
                raw_response_text=text,
                raw_response=raw_response,
                usage=usage,
            )

        decision = to_text(payload.get("decision")).lower()
        if decision not in {"pass", "fail"}:
            decision = "fail"
        confidence = to_text(payload.get("confidence")).lower()
        if confidence not in {"high", "medium", "low"}:
            confidence = "low"
        reason_codes = payload.get("reason_codes")
        if isinstance(reason_codes, str):
            reason_codes = [reason_codes]
        if not isinstance(reason_codes, Sequence):
            reason_codes = ["invalid_judge_output"]
        normalized_codes: list[str] = []
        for code in reason_codes:
            normalized = to_text(code).lower()
            if not normalized:
                continue
            if normalized not in ALLOWED_REASON_CODES:
                normalized = "other"
            if normalized not in normalized_codes:
                normalized_codes.append(normalized)
            if len(normalized_codes) >= max_reason_codes:
                break
        if not normalized_codes:
            normalized_codes = ["ok"] if decision == "pass" else ["other"]
        return SelfConsistencyVote(
            decision=decision,
            confidence=confidence,
            reason_codes=normalized_codes,
            raw_response_text=text,
            raw_response=raw_response,
            usage=usage,
        )

    @staticmethod
    def _build_schema_lines(schema: DatabaseSchema) -> list[str]:
        lines: list[str] = []
        for table_name, table in schema.tables.items():
            columns = []
            for column_name, column in table.columns.items():
                column_type = column.column_type or column.data_type or "text"
                columns.append(f"{column_name} {column_type}")
            lines.append(f"- {table_name}({', '.join(columns)})")
        return lines

    @staticmethod
    def _build_sql_feature_summary(parsed_sql: ParsedSQL) -> dict[str, Any]:
        return {
            "tables": list(parsed_sql.tables),
            "columns": list(parsed_sql.columns),
            "postgis_functions": list(parsed_sql.postgis_functions),
            "aggregates": list(parsed_sql.aggregates),
            "group_by_columns": list(parsed_sql.group_by_columns),
            "order_by": stable_jsonify(parsed_sql.order_by),
            "limit": parsed_sql.limit,
            "spatial_predicates": list(parsed_sql.spatial_predicates),
            "distance_thresholds": list(parsed_sql.distance_thresholds),
            "filters": list(parsed_sql.filters),
            "has_cte": parsed_sql.has_cte,
            "has_subquery": parsed_sql.has_subquery,
        }

    @staticmethod
    def _build_sample_payload(sample: NLSQLSample, parsed_sql: ParsedSQL) -> dict[str, Any]:
        sql_context = sample.metadata.get("sql_context") if isinstance(sample.metadata, Mapping) else None
        return {
            "sample_id": sample.sample_id,
            "sql_id": sample.sql_id,
            "database_id": sample.database_id,
            "difficulty_level": sample.difficulty_level,
            "linguistic_style": sample.linguistic_style,
            "question": sample.question,
            "sql": sample.sql,
            "used_tables": list(sample.used_tables or parsed_sql.tables),
            "used_columns": list(sample.used_columns or parsed_sql.columns),
            "used_spatial_functions": list(sample.used_spatial_functions or parsed_sql.postgis_functions),
            "sql_generation_validation": stable_jsonify((sql_context or {}).get("validation_result", {})),
            "sql_generation_execution": stable_jsonify((sql_context or {}).get("execution_result", {})),
        }

    @staticmethod
    def _build_execution_summary(sample: NLSQLSample, validation_result: ValidationResult) -> dict[str, Any]:
        sql_context = sample.metadata.get("sql_context") if isinstance(sample.metadata, Mapping) else None
        source_execution = (sql_context or {}).get("execution_result", {}) if isinstance(sql_context, Mapping) else {}
        return {
            "quality_control_execution_status": validation_result.execution_status,
            "quality_control_row_count": validation_result.row_count,
            "quality_control_result_preview": stable_jsonify(validation_result.result_preview),
            "synthesized_sql_execution_result": stable_jsonify(source_execution),
        }

    @staticmethod
    def _build_representative_values(schema: DatabaseSchema, sample: NLSQLSample) -> dict[str, Any]:
        table_names = sample.used_tables or []
        representative_values: dict[str, Any] = {}
        for table_name in table_names:
            table = schema.tables.get(table_name)
            if table is None or not table.representative_values:
                continue
            representative_values[table_name] = stable_jsonify(table.representative_values)
        return representative_values

    @staticmethod
    def _build_judge_rules(config: QualityControlConfig) -> dict[str, Any]:
        return {
            "allow_empty_result": config.run.allow_empty_result,
            "must_match_entities": True,
            "must_match_filters": True,
            "must_match_aggregation": True,
            "must_match_grouping": True,
            "must_match_ordering_and_limit": True,
            "must_match_spatial_relation_and_direction": True,
            "must_match_distance_thresholds": True,
            "max_reason_codes": config.judge.max_reason_codes,
        }
