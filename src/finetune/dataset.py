"""Dataset preparation for TRL spatial Text-to-SQL fine-tuning."""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

from .config import FinetuneDataConfig
from .models import PreparedFinetuneSample, RawFinetuneSample
from .prompting import FinetunePromptRenderer
from .utils import stable_jsonify, to_text

LOGGER = logging.getLogger(__name__)


class SpatialText2SQLDatasetBuilder:
    def __init__(
        self,
        *,
        data_config: FinetuneDataConfig,
        prompt_renderer: FinetunePromptRenderer | None = None,
    ) -> None:
        self.data_config = data_config
        self.prompt_renderer = prompt_renderer or FinetunePromptRenderer(
            task_description=self.data_config.task_description,
            max_representative_rows=self.data_config.max_representative_rows,
        )

    def prepare_samples(self, rows: Sequence[RawFinetuneSample]) -> list[PreparedFinetuneSample]:
        prepared_rows: list[PreparedFinetuneSample] = []
        next_question_id = self.data_config.question_id_start
        for row in rows:
            metadata = self._load_metadata(row)
            schema_lines, representative_values = FinetunePromptRenderer.build_runtime_prompt_context(
                metadata,
                included_tables=row.used_tables,
                max_representative_rows=self.data_config.max_representative_rows,
            )
            instruction = row.instruction or self.prompt_renderer.render_instruction()
            input_text = row.input_text or self.prompt_renderer.render_input(
                question=row.question,
                schema_lines=schema_lines,
                representative_values=representative_values,
            )
            output_text = row.output_text or self.prompt_renderer.render_output(
                row.sql_reasoning_summary,
                row.sql,
            )
            prompt = self.prompt_renderer.compose_prompt(instruction, input_text)
            completion = output_text
            prepared_rows.append(
                PreparedFinetuneSample(
                    question_id=next_question_id,
                    database_id=row.database_id,
                    question=row.question,
                    sql=row.sql,
                    difficulty=row.difficulty,
                    prompt=prompt,
                    completion=completion,
                    instruction=instruction,
                    input_text=input_text,
                    output_text=output_text,
                    cot=row.sql_reasoning_summary,
                    sql_reasoning_summary=row.sql_reasoning_summary,
                    schema=list(schema_lines),
                    representative_values=stable_jsonify(representative_values),
                    used_tables=list(row.used_tables),
                    used_columns=list(row.used_columns),
                    used_spatial_functions=list(row.used_spatial_functions),
                )
            )
            next_question_id += 1
        return prepared_rows

    def _load_metadata(self, row: RawFinetuneSample) -> dict[str, Any] | None:
        embedded_metadata = self._load_embedded_metadata(row)
        if embedded_metadata is not None:
            return embedded_metadata
        LOGGER.warning(
            "Fine-tune sample %s is missing embedded metadata.database_context; schema prompt context will be empty.",
            row.question_id or row.database_id,
        )
        return None

    @staticmethod
    def _load_embedded_metadata(row: RawFinetuneSample) -> dict[str, Any] | None:
        metadata = row.metadata if isinstance(row.metadata, Mapping) else {}
        database_context = metadata.get("database_context")
        if (
            isinstance(database_context, Mapping)
            and isinstance(database_context.get("tables"), Sequence)
            and not isinstance(database_context.get("tables"), (str, bytes))
        ):
            return {str(key): stable_jsonify(value) for key, value in database_context.items()}
        if isinstance(metadata.get("tables"), Sequence) and not isinstance(metadata.get("tables"), (str, bytes)):
            return {str(key): stable_jsonify(value) for key, value in metadata.items()}
        return None
