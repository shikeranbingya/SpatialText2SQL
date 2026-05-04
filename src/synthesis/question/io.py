"""I/O helpers for diversity-aware question generation."""

from __future__ import annotations

import json
from pathlib import Path

from src.synthesis.database.io import load_synthesized_databases
from src.synthesis.database.models import SynthesizedSpatialDatabase

from .models import QuestionGenerationContext, SQLQuestionSource, SynthesizedQuestion


def load_sql_question_sources(input_path: str) -> list[SQLQuestionSource]:
    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"SQL input JSONL file not found: {path}")
    rows: list[SQLQuestionSource] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
            try:
                rows.append(SQLQuestionSource.from_dict(payload))
            except ValueError as exc:
                raise ValueError(f"Invalid SQL query row on line {line_number} of {path}: {exc}") from exc
    return rows


def load_question_generation_contexts(
    database_context_path: str,
) -> dict[str, QuestionGenerationContext]:
    databases: list[SynthesizedSpatialDatabase] = load_synthesized_databases(database_context_path)
    return {
        database.database_id: QuestionGenerationContext.from_database(database)
        for database in databases
    }


def write_synthesized_questions(output_path: str, rows: list[SynthesizedQuestion]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), ensure_ascii=False))
            handle.write("\n")
