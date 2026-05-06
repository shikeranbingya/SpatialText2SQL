"""Database client implementations for NL-SQL quality control."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import psycopg2
from psycopg2 import sql as pg_sql
from psycopg2.extras import RealDictCursor

from src.synthesis.database.migration import normalize_postgres_identifier
from src.synthesis.database.utils import stable_jsonify, to_text

from .config import QualityControlDatabaseConfig
from .models import ColumnSchema, DatabaseSchema, TableSchema
from .registry import DatabaseClient, DatabaseRegistry

LOGGER = logging.getLogger(__name__)


@dataclass
class PostgreSQLDatabaseClient(DatabaseClient):
    config: QualityControlDatabaseConfig
    database_id: str

    def inspect_schema(self) -> DatabaseSchema:
        schema_name = normalize_postgres_identifier(self.database_id, prefix="schema")
        with self._connect() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                self._apply_session_settings(cur, schema_name)
                columns_by_table = self._fetch_columns(cur, schema_name)
                spatial_fields = self._fetch_spatial_fields(cur, schema_name)
                tables: dict[str, TableSchema] = {}
                for table_name, columns in columns_by_table.items():
                    enriched_columns: list[dict[str, Any]] = []
                    spatial_lookup = {
                        field.column_name: field
                        for field in spatial_fields.get(table_name, [])
                    }
                    for column in columns:
                        field = spatial_lookup.get(column.column_name)
                        payload = column.to_dict()
                        if field is not None:
                            payload["spatial_type"] = field.spatial_type
                            payload["geometry_type"] = field.geometry_type
                            payload["srid"] = field.srid
                            payload["column_type"] = field.column_type or payload["column_type"]
                        enriched_columns.append(payload)
                    tables[table_name] = TableSchema.from_payload(
                        table_name,
                        columns=enriched_columns,
                    )
                return DatabaseSchema(database_id=self.database_id, tables=tables)

    def execute_read_only(self, sql: str, *, max_preview_rows: int) -> tuple[int, list[dict[str, object]]]:
        schema_name = normalize_postgres_identifier(self.database_id, prefix="schema")
        sql_text = to_text(sql).rstrip(";")
        preview_query = f"SELECT * FROM ({sql_text}) AS qc_sub LIMIT %s"
        count_query = f"SELECT COUNT(*) AS qc_count FROM ({sql_text}) AS qc_sub"
        with self._connect() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                self._apply_session_settings(cur, schema_name)
                cur.execute(count_query)
                count_row = cur.fetchone() or {}
                row_count = int(count_row.get("qc_count") or 0)
                cur.execute(preview_query, (int(max_preview_rows),))
                preview_rows = [dict(row) for row in (cur.fetchall() or [])]
        return row_count, stable_jsonify(preview_rows)

    def _connect(self):
        return psycopg2.connect(
            host=self.config.host,
            port=self.config.port,
            dbname=self.config.database,
            user=self.config.user,
            password=self.config.password,
            connect_timeout=self.config.connect_timeout,
        )

    def _apply_session_settings(self, cursor, schema_name: str) -> None:
        cursor.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
        cursor.execute("SET statement_timeout = %s", (int(self.config.statement_timeout),))
        configured_search_path = to_text(self.config.search_path)
        if configured_search_path:
            resolved_search_path = configured_search_path.replace("{schema}", schema_name)
            if "{schema}" not in configured_search_path:
                resolved_search_path = f"{schema_name}, {configured_search_path}"
        else:
            resolved_search_path = schema_name
        cursor.execute(pg_sql.SQL("SET search_path TO {}").format(pg_sql.SQL(resolved_search_path)))

    @staticmethod
    def _fetch_columns(cursor, schema_name: str) -> dict[str, list[ColumnSchema]]:
        query = """
            SELECT
                c.table_name,
                c.column_name,
                c.data_type,
                c.udt_name,
                pg_catalog.format_type(a.atttypid, a.atttypmod) AS formatted_type
            FROM information_schema.columns AS c
            JOIN pg_catalog.pg_namespace AS ns
              ON ns.nspname = c.table_schema
            JOIN pg_catalog.pg_class AS cls
              ON cls.relname = c.table_name
             AND cls.relnamespace = ns.oid
            JOIN pg_catalog.pg_attribute AS a
              ON a.attrelid = cls.oid
             AND a.attname = c.column_name
             AND a.attnum > 0
             AND NOT a.attisdropped
            WHERE c.table_schema = %s
            ORDER BY c.table_name, c.ordinal_position
        """
        cursor.execute(query, (schema_name,))
        tables: dict[str, list[ColumnSchema]] = {}
        for row in cursor.fetchall() or []:
            row_dict = dict(row)
            table_name = to_text(row_dict.get("table_name"))
            if not table_name:
                continue
            tables.setdefault(table_name, []).append(
                ColumnSchema.from_dict(
                    {
                        "column_name": row_dict.get("column_name"),
                        "column_type": PostgreSQLDatabaseClient._normalize_column_type(
                            row_dict.get("formatted_type"),
                            row_dict.get("data_type"),
                            row_dict.get("udt_name"),
                        ),
                        "data_type": row_dict.get("data_type"),
                        "udt_name": row_dict.get("udt_name"),
                    }
                )
            )
        return tables

    @staticmethod
    def _fetch_spatial_fields(cursor, schema_name: str) -> dict[str, list[ColumnSchema]]:
        tables: dict[str, list[ColumnSchema]] = {}
        for query, spatial_type in (
            (
                """
                SELECT f_table_name AS table_name, f_geometry_column AS column_name, type, srid
                FROM public.geometry_columns
                WHERE f_table_schema = %s
                ORDER BY f_table_name, f_geometry_column
                """,
                "geometry",
            ),
            (
                """
                SELECT f_table_name AS table_name, f_geography_column AS column_name, type, srid
                FROM public.geography_columns
                WHERE f_table_schema = %s
                ORDER BY f_table_name, f_geography_column
                """,
                "geography",
            ),
        ):
            try:
                cursor.execute(query, (schema_name,))
            except Exception:
                cursor.connection.rollback()
                continue
            for row in cursor.fetchall() or []:
                row_dict = dict(row)
                table_name = to_text(row_dict.get("table_name"))
                column_name = to_text(row_dict.get("column_name"))
                geometry_type = to_text(row_dict.get("type")).upper() or "GEOMETRY"
                srid = row_dict.get("srid")
                type_display = f"{spatial_type}({geometry_type},{srid})" if srid not in (None, "") else f"{spatial_type}({geometry_type})"
                tables.setdefault(table_name, []).append(
                    ColumnSchema.from_dict(
                        {
                            "column_name": column_name,
                            "column_type": type_display,
                            "spatial_type": spatial_type,
                            "geometry_type": geometry_type,
                            "srid": srid,
                        }
                    )
                )
        return tables

    @staticmethod
    def _normalize_column_type(formatted_type: Any, data_type: Any, udt_name: Any) -> str:
        formatted = to_text(formatted_type)
        if formatted:
            return formatted
        data = to_text(data_type).lower()
        udt = to_text(udt_name).lower()
        return formatted or data or udt or "text"


@dataclass
class PostgreSQLDatabaseRegistry(DatabaseRegistry):
    config: QualityControlDatabaseConfig
    _clients: dict[str, PostgreSQLDatabaseClient] = None

    def __post_init__(self) -> None:
        self._clients = {}

    def get_client(self, database_id: str) -> PostgreSQLDatabaseClient:
        if database_id not in self._clients:
            self._clients[database_id] = PostgreSQLDatabaseClient(self.config, database_id)
        return self._clients[database_id]
