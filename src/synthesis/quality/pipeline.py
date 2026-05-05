"""High-level quality control pipeline for synthetic spatial NL-SQL data."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field

from src.synthesis.sql.function_library import PostGISFunctionLibrary

from .balancing import DiversityBalancer
from .config import QualityControlConfig
from .duplicates import DuplicateDetector
from .judge import SelfConsistencyQualityJudge
from .models import NLSQLSample, QualityControlReport
from .registry import DatabaseRegistry, SchemaRegistry
from .validation import (
    SQLSampleValidator,
    build_distribution,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class QualityControlPipeline:
    function_library: PostGISFunctionLibrary
    self_consistency_judge: SelfConsistencyQualityJudge | None = None
    sql_validator: SQLSampleValidator = field(init=False)

    def __post_init__(self) -> None:
        self.sql_validator = SQLSampleValidator(self.function_library)

    def run(
        self,
        samples: list[NLSQLSample],
        database_registry: DatabaseRegistry,
        schema_registry: SchemaRegistry,
        config: QualityControlConfig,
    ) -> tuple[list[NLSQLSample], QualityControlReport]:
        failure_reasons: Counter[str] = Counter()
        validated_samples: list[NLSQLSample] = []

        for sample in samples:
            LOGGER.info("Quality control validating sample_id=%s database_id=%s", sample.sample_id, sample.database_id)
            try:
                database_client = database_registry.get_client(sample.database_id)
            except Exception as exc:
                failure_reasons[f"database_client:{exc}"] += 1
                continue

            schema = None
            if config.run.prefer_live_schema:
                try:
                    schema = database_client.inspect_schema()
                    schema_registry.set_schema(schema)
                except Exception as exc:
                    LOGGER.warning("Failed to inspect live schema for %s: %s", sample.database_id, exc)
                    schema = schema_registry.get_schema(sample.database_id)
            else:
                schema = schema_registry.get_schema(sample.database_id)
                if schema is None:
                    try:
                        schema = database_client.inspect_schema()
                        schema_registry.set_schema(schema)
                    except Exception as exc:
                        failure_reasons[f"schema_lookup:{exc}"] += 1
                        continue

            if schema is None:
                failure_reasons["missing_schema"] += 1
                continue

            artifact = self.sql_validator.validate(
                sample=sample,
                schema=schema,
                database_client=database_client,
                config=config,
            )
            for error in artifact.validation_result.errors:
                failure_reasons[error] += 1
            if artifact.validation_result.passed and self.self_consistency_judge is not None:
                judgment = self.self_consistency_judge.judge(
                    sample=sample,
                    schema=schema,
                    parsed_sql=artifact.parsed_sql,
                    validation_result=artifact.validation_result,
                    config=config,
                )
                artifact.validation_result.self_consistency = judgment.to_dict()
                artifact.validation_result.warnings.extend(judgment.warnings)
                if not judgment.passed:
                    artifact.validation_result.errors.append(
                        "Self-consistency judge rejected the NL-SQL pair."
                    )
                    artifact.validation_result.errors.extend(
                        [f"judge:{code}" for code in judgment.reason_codes]
                    )
                    artifact.validation_result.passed = False
                    failure_reasons["self_consistency_rejected"] += 1
                    for code in judgment.reason_codes:
                        failure_reasons[f"judge:{code}"] += 1

            if not artifact.validation_result.passed:
                LOGGER.warning(
                    "Quality control rejected sample_id=%s errors=%s warnings=%s",
                    sample.sample_id,
                    artifact.validation_result.errors,
                    artifact.validation_result.warnings,
                )
                continue

            metadata = dict(sample.metadata)
            metadata["quality_control"] = artifact.validation_result.to_dict()
            sample.metadata = metadata
            validated_samples.append(sample)

        duplicate_result = DuplicateDetector(config.duplicates).run(validated_samples)
        for reason in duplicate_result.duplicate_reasons:
            failure_reasons[reason] += 1

        balanced_samples, dropped_by_balance = DiversityBalancer(config.balancing).run(duplicate_result.retained)
        for _sample_id in dropped_by_balance:
            failure_reasons["balancing_drop"] += 1

        report = QualityControlReport(
            total_samples=len(samples),
            passed_samples=len(balanced_samples),
            failed_samples=len(samples) - len(balanced_samples),
            failure_reasons=dict(failure_reasons),
            duplicate_count=duplicate_result.duplicate_count,
            distribution_by_difficulty=build_distribution(balanced_samples, lambda sample: [sample.difficulty_level]),
            distribution_by_spatial_function=build_distribution(balanced_samples, lambda sample: sample.used_spatial_functions),
            distribution_by_linguistic_style=build_distribution(balanced_samples, lambda sample: [sample.linguistic_style]),
        )
        return balanced_samples, report
