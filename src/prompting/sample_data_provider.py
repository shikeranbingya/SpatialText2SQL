"""Build prompt sample data blocks from PostgreSQL tables."""

from __future__ import annotations

import json
import math
import re
import warnings
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml
import psycopg2

from .schema_compactor import DEFAULT_PROJECT_ROOT, SchemaCompactor


@dataclass(frozen=True)
class PromptTableColumn:
    name: str
    column_type: str


@dataclass(frozen=True)
class ResolvedColumn:
    prompt_name: str
    prompt_type: str
    actual_name: str
    data_type: str
    udt_name: str


class PostgresSampleDataProvider:
    """Fetch compact prompt sample rows from PostgreSQL/PostGIS."""

    DATABASE_KEY_BY_DATASET = {
        "spatial_qa": "postgres",
        "spatialsql_pg": "spatial_sql",
        "floodsql_pg": "floodsql",
    }
    SAMPLE_LIMIT = 5
    MAX_TEXT_LENGTH = 96

    def __init__(
        self,
        project_root: str | Path | None = None,
        db_config_path: str | Path | None = None,
        db_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.project_root = Path(project_root).resolve() if project_root else DEFAULT_PROJECT_ROOT
        self.db_config_path = (
            Path(db_config_path).resolve()
            if db_config_path
            else self.project_root / "config" / "db_config.yaml"
        )
        self.db_config = db_config if db_config is not None else self._load_db_config()
        self.schema_compactor = SchemaCompactor(project_root=self.project_root)
        self._connections: Dict[str, Any] = {}
        self._failed_db_keys: set[str] = set()
        self._table_meta_cache: Dict[Tuple[str, str], Optional[Dict[str, Any]]] = {}
        self._sample_cache: Dict[Tuple[str, str, Tuple[Tuple[str, str], ...]], List[str]] = {}

    def build_sample_data(
        self,
        dataset_name: str,
        metadata: Optional[Dict[str, Any]],
        compact_schema: str,
    ) -> str:
        del metadata
        if not dataset_name or not compact_schema.strip():
            return ""

        table_specs = self._parse_prompt_tables(compact_schema)
        if not table_specs:
            return ""

        db_key = self.DATABASE_KEY_BY_DATASET.get(dataset_name)
        if not db_key:
            return ""

        connection = self._get_connection(db_key)
        if connection is None:
            return ""

        lines: List[str] = []
        for table_name, columns in table_specs:
            row_texts = self._get_table_samples(
                connection=connection,
                db_key=db_key,
                table_name=table_name,
                columns=columns,
            )
            if not row_texts:
                continue
            lines.append(f"- {table_name}")
            lines.extend(f"  {row_text}" for row_text in row_texts)
        return "\n".join(lines)

    def _parse_prompt_tables(self, compact_schema: str) -> List[Tuple[str, List[PromptTableColumn]]]:
        tables: List[Tuple[str, List[PromptTableColumn]]] = []
        for raw_line in compact_schema.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            match = re.match(r"^-\s+([A-Za-z0-9_]+)\((.*)\)$", line)
            if match is None:
                match = re.match(r"^(?:table\s+)?([A-Za-z0-9_]+)\((.*)\)$", line, re.I)
            if match is None:
                continue

            table_name, raw_columns = match.groups()
            columns = self._parse_prompt_columns(raw_columns)
            if columns:
                tables.append((table_name, columns))

        if tables:
            return tables

        return [
            (
                table_name,
                [PromptTableColumn(name=column_name, column_type=column_type) for column_name, column_type in columns],
            )
            for table_name, columns in self.schema_compactor._parse_runtime_schema(compact_schema)
        ]

    @staticmethod
    def _parse_prompt_columns(raw_columns: str) -> List[PromptTableColumn]:
        columns: List[PromptTableColumn] = []
        for raw_column in raw_columns.split(","):
            tokens = raw_column.strip().split()
            if len(tokens) < 2:
                continue
            columns.append(
                PromptTableColumn(
                    name=tokens[0].strip('"').lower(),
                    column_type=tokens[1].lower(),
                )
            )
        return columns

    def _load_db_config(self) -> Dict[str, Any]:
        if not self.db_config_path.exists():
            return {}
        with open(self.db_config_path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def _get_connection(self, db_key: str):
        if db_key in self._failed_db_keys:
            return None

        connection = self._connections.get(db_key)
        if connection is not None and not getattr(connection, "closed", False):
            return connection

        db_settings = self._get_db_settings(db_key)
        if not db_settings:
            warnings.warn(
                f"未找到数据库配置 {db_key}，跳过 prompt sample data 注入。",
                RuntimeWarning,
                stacklevel=2,
            )
            self._failed_db_keys.add(db_key)
            return None

        try:
            connection = psycopg2.connect(
                host=db_settings.get("host"),
                port=db_settings.get("port"),
                dbname=db_settings.get("database"),
                user=db_settings.get("user"),
                password=db_settings.get("password"),
                connect_timeout=int(
                    db_settings.get("timeout", {}).get("connection_timeout", 5)
                ),
            )
            connection.autocommit = True
        except Exception as exc:  # pragma: no cover - exercised via mocks/tests
            warnings.warn(
                f"连接数据库 {db_key} 失败，跳过 prompt sample data 注入: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            self._failed_db_keys.add(db_key)
            return None

        self._connections[db_key] = connection
        return connection

    def _get_db_settings(self, db_key: str) -> Dict[str, Any]:
        databases = self.db_config.get("databases", {})
        if db_key in databases:
            return databases[db_key] or {}
        if db_key == "postgres":
            return self.db_config.get("database", {}) or {}
        return {}

    def _get_table_samples(
        self,
        connection,
        db_key: str,
        table_name: str,
        columns: Sequence[PromptTableColumn],
    ) -> List[str]:
        cache_key = (
            db_key,
            table_name,
            tuple((column.name, column.column_type) for column in columns),
        )
        if cache_key in self._sample_cache:
            return self._sample_cache[cache_key]

        try:
            row_texts = self._fetch_table_samples(connection, db_key, table_name, columns)
        except Exception as exc:
            self._rollback_quietly(connection)
            warnings.warn(
                f"查询表 {table_name} 的 sample data 失败，已跳过: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            row_texts = []

        self._sample_cache[cache_key] = row_texts
        return row_texts

    def _fetch_table_samples(
        self,
        connection,
        db_key: str,
        table_name: str,
        columns: Sequence[PromptTableColumn],
    ) -> List[str]:
        table_meta = self._resolve_table_meta(connection, db_key, table_name)
        if not table_meta:
            return []

        resolved_columns: List[ResolvedColumn] = []
        column_lookup = table_meta["columns"]
        for column in columns:
            actual_meta = column_lookup.get(column.name)
            if not actual_meta:
                continue
            resolved_columns.append(
                ResolvedColumn(
                    prompt_name=column.name,
                    prompt_type=column.column_type,
                    actual_name=actual_meta["actual_name"],
                    data_type=actual_meta["data_type"],
                    udt_name=actual_meta["udt_name"],
                )
            )

        if not resolved_columns:
            return []

        fetch_columns = [
            column for column in resolved_columns if not self._uses_placeholder_only(column)
        ]
        rows = self._query_table_rows(
            connection=connection,
            table_name=table_meta["actual_table"],
            fetch_columns=fetch_columns,
        )
        if not rows:
            return []

        rendered_rows: List[str] = []
        for row in rows:
            values = tuple(row) if isinstance(row, (list, tuple)) else (row,)
            value_index = 0
            rendered: Dict[str, Any] = {}
            for column in resolved_columns:
                if self._is_geometry_column(column):
                    rendered[column.prompt_name] = "<geometry>"
                    continue
                if self._is_binary_column(column):
                    rendered[column.prompt_name] = "<binary>"
                    continue
                raw_value = values[value_index] if value_index < len(values) else None
                value_index += 1
                rendered[column.prompt_name] = self._normalize_value(raw_value)
            rendered_rows.append(json.dumps(rendered, ensure_ascii=False))
        return rendered_rows

    def _resolve_table_meta(
        self,
        connection,
        db_key: str,
        requested_table: str,
    ) -> Optional[Dict[str, Any]]:
        cache_key = (db_key, requested_table)
        if cache_key in self._table_meta_cache:
            return self._table_meta_cache[cache_key]

        query = """
            SELECT table_name, column_name, data_type, udt_name, ordinal_position
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND lower(table_name) = lower(%s)
            ORDER BY table_name, ordinal_position
        """
        rows = self._execute_fetchall(connection, query, (requested_table,))
        if not rows:
            warnings.warn(
                f"表 {requested_table} 不存在于目标 PostgreSQL 数据库中，已跳过 sample data 注入。",
                RuntimeWarning,
                stacklevel=2,
            )
            self._table_meta_cache[cache_key] = None
            return None

        tables: Dict[str, List[Tuple[str, str, str]]] = {}
        for table_name, column_name, data_type, udt_name, _ordinal_position in rows:
            tables.setdefault(table_name, []).append(
                (
                    column_name,
                    str(data_type or "").lower(),
                    str(udt_name or "").lower(),
                )
            )

        if requested_table in tables:
            actual_table = requested_table
        else:
            actual_table = sorted(
                tables.keys(),
                key=lambda name: (name.lower() != requested_table.lower(), name),
            )[0]

        column_meta = {
            column_name.lower(): {
                "actual_name": column_name,
                "data_type": data_type,
                "udt_name": udt_name,
            }
            for column_name, data_type, udt_name in tables[actual_table]
        }
        resolved = {
            "actual_table": actual_table,
            "columns": column_meta,
        }
        self._table_meta_cache[cache_key] = resolved
        return resolved

    def _query_table_rows(
        self,
        connection,
        table_name: str,
        fetch_columns: Sequence[ResolvedColumn],
    ) -> List[Tuple[Any, ...]]:
        table_sql = self._quote_identifier(table_name)
        if fetch_columns:
            select_sql = ", ".join(
                self._quote_identifier(column.actual_name) for column in fetch_columns
            )
            query = f"SELECT {select_sql} FROM {table_sql}"
        else:
            query = f"SELECT 1 FROM {table_sql}"

        ordered_query = query
        order_column = next(
            (
                column
                for column in fetch_columns
                if not self._is_binary_column(column)
            ),
            None,
        )
        if order_column is not None:
            ordered_query = (
                f"{query} ORDER BY {self._quote_identifier(order_column.actual_name)} "
                f"NULLS LAST LIMIT %s"
            )
            try:
                return self._execute_fetchall(
                    connection,
                    ordered_query,
                    (self.SAMPLE_LIMIT,),
                )
            except Exception:
                self._rollback_quietly(connection)

        return self._execute_fetchall(
            connection,
            f"{query} LIMIT %s",
            (self.SAMPLE_LIMIT,),
        )

    @staticmethod
    def _execute_fetchall(connection, query: str, params: Sequence[Any]) -> List[Tuple[Any, ...]]:
        cursor = connection.cursor()
        try:
            cursor.execute(query, params)
            return list(cursor.fetchall())
        finally:
            close = getattr(cursor, "close", None)
            if callable(close):
                close()

    @staticmethod
    def _rollback_quietly(connection) -> None:
        rollback = getattr(connection, "rollback", None)
        if callable(rollback):
            try:
                rollback()
            except Exception:
                return

    @staticmethod
    def _quote_identifier(identifier: str) -> str:
        escaped = (identifier or "").replace('"', '""')
        return f'"{escaped}"'

    @staticmethod
    def _normalize_value(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, bytes):
            return "<binary>"
        if isinstance(value, Decimal):
            return int(value) if value == value.to_integral_value() else float(value)
        if isinstance(value, float):
            return value if math.isfinite(value) else str(value)
        if isinstance(value, (bool, int)):
            return value
        text = " ".join(str(value).strip().split())
        if not text:
            return ""
        if len(text) > PostgresSampleDataProvider.MAX_TEXT_LENGTH:
            text = text[: PostgresSampleDataProvider.MAX_TEXT_LENGTH - 3].rstrip() + "..."
        return text

    @staticmethod
    def _uses_placeholder_only(column: ResolvedColumn) -> bool:
        return (
            PostgresSampleDataProvider._is_geometry_column(column)
            or PostgresSampleDataProvider._is_binary_column(column)
        )

    @staticmethod
    def _is_geometry_column(column: ResolvedColumn) -> bool:
        return column.prompt_type == "geometry" or column.udt_name == "geometry"

    @staticmethod
    def _is_binary_column(column: ResolvedColumn) -> bool:
        return column.prompt_type == "bytea" or column.udt_name == "bytea"
