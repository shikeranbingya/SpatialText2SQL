"""Prompt rendering for spatial Text-to-SQL fine-tuning."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.synthesis.database.utils import stable_jsonify, to_text


class FinetunePromptRenderer:
    def __init__(
        self,
        *,
        template_path: str | Path,
        task_description: str,
        max_representative_rows: int = 3,
    ) -> None:
        self.template_path = Path(template_path)
        self.task_description = to_text(task_description)
        self.max_representative_rows = max(int(max_representative_rows), 1)
        self._template_text: str | None = None

    def render_prompt(
        self,
        *,
        question: str,
        schema_lines: Sequence[str],
        spatial_lines: Sequence[str],
        representative_values: Mapping[str, Any],
    ) -> str:
        template = self._load_template_text()
        return self._render_template(
            template,
            {
                "task_description": self.task_description,
                "schema_block": "\n".join(schema_lines) if schema_lines else "No schema available.",
                "spatial_field_block": "\n".join(spatial_lines) if spatial_lines else "No spatial fields listed.",
                "representative_values_block": self._stable_json_text(representative_values),
                "question_block": to_text(question),
            },
        )

    def render_completion(self, cot: str, sql: str) -> str:
        _ = cot
        return to_text(sql).strip()

    @staticmethod
    def build_runtime_prompt_context(
        database_runtime_metadata: Mapping[str, Any] | None,
        *,
        included_tables: Sequence[str] | None = None,
        max_representative_rows: int = 3,
    ) -> tuple[list[str], list[str], dict[str, Any]]:
        included = {
            to_text(table_name)
            for table_name in (included_tables or [])
            if to_text(table_name)
        } or None
        schema_lines: list[str] = []
        spatial_lines: list[str] = []
        representative_values: dict[str, Any] = {}
        for table_meta in (database_runtime_metadata or {}).get("tables", []):
            if not isinstance(table_meta, Mapping):
                continue
            table_name = to_text(table_meta.get("table_name"))
            if not table_name:
                continue
            if included is not None and table_name not in included:
                continue
            columns = []
            for column in table_meta.get("columns", []):
                if not isinstance(column, Mapping):
                    continue
                column_name = to_text(column.get("column_name"))
                column_type = to_text(column.get("column_type") or column.get("data_type") or "text")
                if column_name:
                    columns.append(f"{column_name} {column_type}")
            schema_lines.append(f"- {table_name}({', '.join(columns)})")
            geometry_columns = {
                to_text(field.get("column_name") or field.get("canonical_name")).lower()
                for field in table_meta.get("spatial_fields", [])
                if isinstance(field, Mapping) and to_text(field.get("column_name") or field.get("canonical_name"))
            }
            table_rep_values = table_meta.get("representative_values") or {}
            representative_values[table_name] = FinetunePromptRenderer._prepare_representative_rows(
                table_rep_values,
                geometry_columns=geometry_columns,
                limit=max_representative_rows,
            )
            for field in table_meta.get("spatial_fields", []):
                if not isinstance(field, Mapping):
                    continue
                spatial_name = to_text(field.get("column_name") or field.get("canonical_name"))
                column_type = to_text(field.get("column_type"))
                spatial_type = to_text(field.get("spatial_type")) or "spatial"
                geometry_type = to_text(field.get("geometry_type")) or "GEOMETRY"
                srid = field.get("srid")
                if spatial_name:
                    spatial_lines.append(
                        f"- {table_name}.{spatial_name} "
                        f"(type={column_type or spatial_type}, family={spatial_type}, geometry_type={geometry_type}, srid={srid if srid not in (None, '') else 'unknown'})"
                    )
        return schema_lines, spatial_lines, representative_values

    @staticmethod
    def _prepare_representative_rows(
        representative_values: Any,
        *,
        geometry_columns: set[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        normalized = representative_values
        if isinstance(normalized, Mapping) and isinstance(normalized.get("rows"), list):
            normalized = normalized.get("rows")
        rows: list[dict[str, Any]] = []
        if isinstance(normalized, list):
            for item in normalized:
                if isinstance(item, Mapping):
                    rows.append(dict(item))
                else:
                    rows.append({"value": item})
        elif isinstance(normalized, Mapping):
            if FinetunePromptRenderer._looks_column_oriented_samples(normalized):
                rows = FinetunePromptRenderer._transpose_column_oriented_samples(normalized)
            else:
                rows = [dict(normalized)]

        prepared: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in rows:
            normalized_row: dict[str, Any] = {}
            for key, raw_value in row.items():
                column_name = to_text(key)
                if not column_name:
                    continue
                normalized_row[column_name] = FinetunePromptRenderer._normalize_representative_value(
                    column_name,
                    raw_value,
                    geometry_columns=geometry_columns,
                )
            signature = FinetunePromptRenderer._stable_json_text(normalized_row)
            if signature in seen:
                continue
            seen.add(signature)
            prepared.append(normalized_row)
            if len(prepared) >= limit:
                break
        return prepared

    @staticmethod
    def _normalize_representative_value(
        column_name: str,
        value: Any,
        *,
        geometry_columns: set[str],
    ) -> Any:
        if column_name.lower() in geometry_columns:
            if value in (None, ""):
                return None
            return FinetunePromptRenderer._geometry_preview(value)
        return value

    @staticmethod
    def _geometry_preview(value: Any) -> str:
        if isinstance(value, Mapping):
            geometry_type = to_text(value.get("type"))
            return geometry_type.upper() if geometry_type else "GEOMETRY"
        text = to_text(value).strip()
        if not text:
            return "GEOMETRY"
        upper = text.upper()
        if upper.startswith("SRID=") and ";" in text:
            text = text.split(";", 1)[1].strip()
        match = re.match(r"^([A-Za-z]+)", text)
        if match:
            return match.group(1).upper()
        return "GEOMETRY"

    @staticmethod
    def _looks_column_oriented_samples(values: Mapping[str, Any]) -> bool:
        if not values:
            return False
        if any(isinstance(item, Mapping) for item in values.values()):
            return False
        return any(isinstance(item, (list, tuple)) for item in values.values())

    @staticmethod
    def _transpose_column_oriented_samples(values: Mapping[str, Any]) -> list[dict[str, Any]]:
        normalized_columns: dict[str, list[Any]] = {}
        max_len = 0
        for key, raw_value in values.items():
            column_name = to_text(key)
            if not column_name:
                continue
            items = list(raw_value) if isinstance(raw_value, (list, tuple)) else [raw_value]
            normalized_columns[column_name] = items
            max_len = max(max_len, len(items))
        rows: list[dict[str, Any]] = []
        for index in range(max_len):
            row: dict[str, Any] = {}
            for column_name, items in normalized_columns.items():
                row[column_name] = items[index] if index < len(items) else None
            rows.append(row)
        return rows

    def _load_template_text(self) -> str:
        if self._template_text is None:
            self._template_text = self.template_path.read_text(encoding="utf-8")
        return self._template_text

    @staticmethod
    def _render_template(template_text: str, placeholders: Mapping[str, Any]) -> str:
        rendered = template_text
        for key, value in placeholders.items():
            rendered = rendered.replace(f"{{{{{key}}}}}", to_text(value))
        return rendered

    @staticmethod
    def _stable_json_text(value: Any) -> str:
        return json.dumps(stable_jsonify(value), ensure_ascii=False, indent=2, sort_keys=True)
