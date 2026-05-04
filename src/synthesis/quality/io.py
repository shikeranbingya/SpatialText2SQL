"""I/O helpers for quality control."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from src.synthesis.database.io import load_synthesized_databases
from src.synthesis.database.models import SynthesizedSpatialDatabase

from .models import ColumnSchema, DatabaseSchema, NLSQLSample, QualityControlReport, TableSchema
from .registry import InMemorySchemaRegistry


def load_nl_sql_samples(input_path: str) -> list[NLSQLSample]:
    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"NL-SQL sample file not found: {path}")
    samples: list[NLSQLSample] = []
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
                samples.append(NLSQLSample.from_dict(payload))
            except ValueError as exc:
                raise ValueError(f"Invalid NL-SQL sample on line {line_number} of {path}: {exc}") from exc
    return samples


def write_nl_sql_samples(output_path: str, samples: list[NLSQLSample]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_dict(), ensure_ascii=False))
            handle.write("\n")


def write_quality_control_report(output_path: str, report: QualityControlReport) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def load_schema_registry_from_contexts(input_path: str) -> InMemorySchemaRegistry:
    registry = InMemorySchemaRegistry()
    path = Path(input_path)
    if not path.is_file():
        return registry
    databases: list[SynthesizedSpatialDatabase] = load_synthesized_databases(str(path))
    for database in databases:
        tables: dict[str, TableSchema] = {}
        representative_values_by_table: dict[str, Mapping[str, object]] = {}
        for table in database.selected_tables:
            representative_values_by_table[table.table_name] = table.representative_values
        for table_item in database.schema:
            if not isinstance(table_item, Mapping):
                continue
            table_name = str(table_item.get("table_name") or "").strip()
            if not table_name:
                continue
            columns = []
            for column in table_item.get("normalized_schema", []):
                if not isinstance(column, Mapping):
                    continue
                payload = {
                    "column_name": column.get("canonical_name") or column.get("name"),
                    "column_type": column.get("canonical_type") or column.get("type"),
                }
                columns.append(ColumnSchema.from_dict(payload).to_dict())
            for field in database.spatial_fields:
                if not isinstance(field, Mapping) or str(field.get("table_name") or "").strip() != table_name:
                    continue
                column_name = str(field.get("canonical_name") or field.get("column_name") or "").strip()
                for column in columns:
                    if column.get("column_name") == column_name:
                        column["spatial_type"] = str(field.get("spatial_type") or "").lower()
                        column["geometry_type"] = str(field.get("geometry_type") or "").upper()
                        column["srid"] = field.get("srid")
            tables[table_name] = TableSchema.from_payload(
                table_name,
                columns=columns,
                representative_values=representative_values_by_table.get(table_name),
            )
        registry.set_schema(DatabaseSchema(database_id=database.database_id, tables=tables))
    return registry

