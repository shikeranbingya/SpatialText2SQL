"""Duplicate detection for NL-SQL samples."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from .config import DuplicateDetectionConfig
from .models import NLSQLSample
from .validation import question_similarity


def normalize_sql(sql: str) -> str:
    collapsed = re.sub(r"\s+", " ", (sql or "").strip().rstrip(";"))
    return collapsed.lower()


def normalize_question(question: str) -> str:
    lowered = re.sub(r"[^a-z0-9\s]", " ", (question or "").lower())
    return re.sub(r"\s+", " ", lowered).strip()


@dataclass
class DuplicateDetectionResult:
    retained: list[NLSQLSample]
    duplicate_count: int
    duplicate_reasons: list[str]


@dataclass
class DuplicateDetector:
    config: DuplicateDetectionConfig

    def run(self, samples: Sequence[NLSQLSample]) -> DuplicateDetectionResult:
        retained: list[NLSQLSample] = []
        duplicate_count = 0
        duplicate_reasons: list[str] = []
        seen_exact_sql: set[str] = set()
        seen_normalized_sql: set[str] = set()

        for sample in samples:
            exact_sql = sample.sql.strip()
            normalized_sql = normalize_sql(sample.sql)
            normalized_q = normalize_question(sample.question)
            is_duplicate = False

            if self.config.remove_exact_sql_duplicates and exact_sql in seen_exact_sql:
                duplicate_count += 1
                duplicate_reasons.append("exact_sql_duplicate")
                continue
            if self.config.remove_normalized_sql_duplicates and normalized_sql in seen_normalized_sql:
                duplicate_count += 1
                duplicate_reasons.append("normalized_sql_duplicate")
                continue

            if self.config.remove_near_duplicate_questions:
                for kept in retained:
                    question_sim = question_similarity(sample.question, kept.question)
                    if (
                        self.config.treat_same_sql_similar_questions_as_duplicates
                        and normalized_sql == normalize_sql(kept.sql)
                        and question_sim >= self.config.same_sql_similarity_threshold
                    ):
                        is_duplicate = True
                        duplicate_reasons.append("same_sql_similar_question_duplicate")
                        break
                    if question_sim >= self.config.question_similarity_threshold:
                        is_duplicate = True
                        duplicate_reasons.append("near_duplicate_question")
                        break
            if is_duplicate:
                duplicate_count += 1
                continue

            seen_exact_sql.add(exact_sql)
            seen_normalized_sql.add(normalized_sql)
            retained.append(sample)

        return DuplicateDetectionResult(
            retained=retained,
            duplicate_count=duplicate_count,
            duplicate_reasons=duplicate_reasons,
        )

