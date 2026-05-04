"""Lightweight semantic validation for generated questions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from src.synthesis.database.utils import to_text

from .models import (
    QUESTION_STYLES,
    QuestionGenerationCandidate,
    QuestionValidationResult,
    SQLFeatureSummary,
    SpatialRelationConstraint,
)


AGGREGATE_MARKERS: dict[str, list[str]] = {
    "COUNT": ["how many", "number of", "count", "total number"],
    "SUM": ["total", "sum"],
    "AVG": ["average", "avg", "mean"],
    "MIN": ["minimum", "lowest", "smallest", "least"],
    "MAX": ["maximum", "highest", "largest", "greatest"],
}

RANKING_MARKERS = ["top", "highest", "lowest", "largest", "smallest", "nearest", "farthest", "rank", "first"]
GROUPING_MARKERS = ["for each", "per", "by", "for every"]
COMPARATIVE_MARKERS = ["compare", "difference", "more than", "less than", "higher than", "lower than", "versus"]
EXPLORATORY_MARKERS = ["analyze", "explore", "pattern", "relationship", "distribution"]


def _normalize_numeric_token(value: str) -> set[str]:
    text = value.strip()
    if not text:
        return set()
    variants = {text}
    if re.fullmatch(r"-?\d+(?:\.\d+)?", text):
        number = float(text)
        if number.is_integer():
            variants.add(str(int(number)))
            variants.add(f"{number:.1f}")
    return variants


@dataclass
class QuestionValidator:
    def validate(
        self,
        *,
        candidate: QuestionGenerationCandidate,
        requested_style: str,
        sql_features: SQLFeatureSummary,
        spatial_constraints: Sequence[SpatialRelationConstraint],
    ) -> QuestionValidationResult:
        question = to_text(candidate.question).strip()
        errors: list[str] = []
        warnings: list[str] = []
        preserved_thresholds: list[str] = []
        detected_style_markers: list[str] = []
        lowered = f" {question.lower()} "

        if not question:
            errors.append("Generated question is empty.")
            return QuestionValidationResult(is_valid=False, errors=errors)
        if re.search(r"\bST_[A-Za-z0-9_]+\b", question):
            errors.append("Generated question exposes raw PostGIS function names.")
        if re.search(r"\b(select|from|where|group by|order by|limit)\b", lowered):
            warnings.append("Generated question still contains raw SQL keywords.")

        if requested_style not in QUESTION_STYLES:
            warnings.append(f"Requested style is unknown: {requested_style}")
        elif candidate.style and candidate.style != requested_style:
            errors.append(f"Generated question style mismatch: expected {requested_style}, got {candidate.style}.")

        if sql_features.distance_thresholds:
            for threshold in sql_features.distance_thresholds:
                variants = _normalize_numeric_token(threshold)
                if not any(variant in question for variant in variants):
                    errors.append(f"Question does not preserve the distance/threshold value {threshold}.")
                else:
                    preserved_thresholds.append(threshold)

        for aggregate in sql_features.aggregates:
            markers = AGGREGATE_MARKERS.get(aggregate.upper(), [])
            if markers and not any(marker in lowered for marker in markers):
                errors.append(f"Question does not clearly express aggregate semantics for {aggregate}.")
            else:
                detected_style_markers.extend(marker for marker in markers if marker in lowered)

        if sql_features.group_by_columns and not any(marker in lowered for marker in GROUPING_MARKERS):
            warnings.append("Question may be missing explicit grouping language.")
        else:
            detected_style_markers.extend(marker for marker in GROUPING_MARKERS if marker in lowered)

        if sql_features.order_by or sql_features.limit is not None:
            if not any(marker in lowered for marker in RANKING_MARKERS):
                warnings.append("Question may be missing explicit ranking language.")
            else:
                detected_style_markers.extend(marker for marker in RANKING_MARKERS if marker in lowered)
            if sql_features.limit is not None:
                limit_variants = _normalize_numeric_token(str(sql_features.limit))
                if not any(variant in question for variant in limit_variants):
                    warnings.append(f"Question may be missing the LIMIT/top-k value {sql_features.limit}.")

        if requested_style == "comparative_analysis" and not any(marker in lowered for marker in COMPARATIVE_MARKERS):
            warnings.append("Comparative style question may be missing comparison wording.")
        else:
            detected_style_markers.extend(marker for marker in COMPARATIVE_MARKERS if marker in lowered)

        if requested_style == "exploratory_analysis" and not any(marker in lowered for marker in EXPLORATORY_MARKERS):
            warnings.append("Exploratory style question may be missing analytical wording.")
        else:
            detected_style_markers.extend(marker for marker in EXPLORATORY_MARKERS if marker in lowered)

        if spatial_constraints:
            matched_any = False
            for constraint in spatial_constraints:
                keywords = [word.lower() for word in constraint.required_keywords if word]
                if any(keyword in lowered for keyword in keywords):
                    matched_any = True
            if not matched_any:
                warnings.append("Question may be missing explicit spatial relation wording.")

        return QuestionValidationResult(
            is_valid=not errors,
            errors=errors,
            warnings=warnings,
            preserved_thresholds=preserved_thresholds,
            detected_style_markers=sorted(set(detected_style_markers)),
        )
