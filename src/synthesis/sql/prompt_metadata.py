"""Load prompt-facing schema metadata directly from synthesized PostGIS databases."""

from __future__ import annotations

import json
import logging
import math
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any, Mapping, Sequence

import psycopg2
from psycopg2 import sql as pg_sql
from psycopg2.extras import RealDictCursor

from src.synthesis.database.migration import normalize_postgres_identifier
from src.synthesis.database.models import SynthesizedSpatialDatabase
from src.synthesis.database.utils import stable_jsonify, to_text

from .config import SQLSynthesisDBConfig
from .schema_utils import build_create_table_ddl_query

LOGGER = logging.getLogger(__name__)


class PostGISPromptMetadataProvider:
    """Fetch schema and representative values from the live synthesized PostGIS catalog."""

    MAX_REPRESENTATIVE_ROWS = 3
    MAX_SAMPLE_ROWS = 12
    MAX_TEXT_LENGTH = 120

    def __init__(self, db_config: SQLSynthesisDBConfig) -> None:
        self.db_config = db_config
        self._cache: dict[str, dict[str, Any] | None] = {}

    def load_database_metadata(
        self,
        database: SynthesizedSpatialDatabase,
    ) -> dict[str, Any] | None:
        requested_tables = [
            to_text(name)
            for name in (getattr(database, "selected_table_names", None) or [])
            if to_text(name)
        ]
        if not requested_tables:
            requested_tables = [
                to_text(getattr(table, "table_name", ""))
                for table in getattr(database, "selected_tables", [])
                if to_text(getattr(table, "table_name", ""))
            ]
        cache_key = f"{database.database_id}|{'|'.join(requested_tables)}"
        if cache_key in self._cache:
            return stable_jsonify(self._cache[cache_key])

        schema_name = normalize_postgres_identifier(database.database_id, prefix="schema")
        catalog_name = normalize_postgres_identifier(self.db_config.database, prefix="catalog") or self.db_config.database
        if not requested_tables:
            LOGGER.warning("Prompt metadata provider skipped %s because no table names were supplied.", database.database_id)
            self._cache[cache_key] = None
            return None

        try:
            with self._connect(catalog_name) as conn:
                metadata = self._load_metadata(conn, schema_name, requested_tables, database)
        except Exception as exc:
            LOGGER.warning(
                "Failed to load live PostGIS prompt metadata | schema_id=%s | error=%s",
                database.database_id,
                exc,
            )
            self._cache[cache_key] = None
            return None

        self._cache[cache_key] = metadata
        return stable_jsonify(metadata)

    def _connect(self, catalog_name: str):
        return psycopg2.connect(
            host=self.db_config.host,
            port=self.db_config.port,
            dbname=catalog_name,
            user=self.db_config.user,
            password=self.db_config.password,
            connect_timeout=self.db_config.connect_timeout,
        )

    def _load_metadata(
        self,
        connection,
        schema_name: str,
        requested_tables: Sequence[str],
        database: SynthesizedSpatialDatabase,
    ) -> dict[str, Any]:
        with connection.cursor(cursor_factory=RealDictCursor) as cur:
            self._apply_session_settings(cur, schema_name)
            columns_by_table = self._fetch_columns(cur, schema_name, requested_tables)
            spatial_by_table = self._fetch_spatial_fields(cur, schema_name, requested_tables)
            tables: list[dict[str, Any]] = []
            representative_values: dict[str, Any] = {}
            schema_ddls: list[str] = []
            for table_name in requested_tables:
                columns = columns_by_table.get(table_name, [])
                spatial_fields = spatial_by_table.get(table_name, [])
                if not columns:
                    LOGGER.warning(
                        "Live prompt metadata missing table | schema_id=%s | table=%s",
                        database.database_id,
                        table_name,
                    )
                    continue
                table_representative_values = self._fetch_representative_values(
                    cur,
                    schema_name=schema_name,
                    table_name=table_name,
                    columns=columns,
                )
                create_table_ddl = self._fetch_create_table_ddl(
                    cur,
                    schema_name=schema_name,
                    table_name=table_name,
                )
                representative_values[table_name] = stable_jsonify(table_representative_values)
                if create_table_ddl:
                    schema_ddls.append(create_table_ddl)
                tables.append(
                    {
                        "table_name": table_name,
                        "create_table_ddl": create_table_ddl,
                        "columns": stable_jsonify(columns),
                        "spatial_fields": stable_jsonify(spatial_fields),
                        "representative_values": stable_jsonify(table_representative_values),
                    }
                )
        return {
            "database_id": database.database_id,
            "city": database.city,
            "schema_name": schema_name,
            "schema_ddls": schema_ddls,
            "tables": tables,
            "representative_values": stable_jsonify(representative_values),
        }

    @staticmethod
    def _fetch_create_table_ddl(cursor, schema_name: str, table_name: str) -> str:
        query, params = build_create_table_ddl_query(schema_name, table_name)
        cursor.execute(query, params)
        row = cursor.fetchone() or {}
        if isinstance(row, Mapping):
            return to_text(row.get("create_table_ddl"))
        return ""

    def _apply_session_settings(self, cursor, schema_name: str) -> None:
        cursor.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
        cursor.execute("SET statement_timeout = %s", (int(self.db_config.statement_timeout),))
        configured_search_path = to_text(self.db_config.search_path)
        if configured_search_path:
            resolved_search_path = configured_search_path.replace("{schema}", schema_name)
            if "{schema}" not in configured_search_path:
                resolved_search_path = f"{schema_name}, {configured_search_path}"
        else:
            resolved_search_path = schema_name
        cursor.execute(
            pg_sql.SQL("SET search_path TO {}").format(pg_sql.SQL(resolved_search_path))
        )

    @staticmethod
    def _fetch_columns(cursor, schema_name: str, requested_tables: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
        query = """
            SELECT
                c.table_name,
                c.column_name,
                c.data_type,
                c.udt_name,
                c.ordinal_position,
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
              AND c.table_name = ANY(%s)
            ORDER BY c.table_name, c.ordinal_position
        """
        cursor.execute(query, (schema_name, list(requested_tables)))
        tables: dict[str, list[dict[str, Any]]] = {}
        for row in cursor.fetchall() or []:
            row_dict = dict(row)
            table_name = to_text(row_dict.get("table_name"))
            if not table_name:
                continue
            tables.setdefault(table_name, []).append(
                {
                    "column_name": to_text(row_dict.get("column_name")),
                    "column_type": PostGISPromptMetadataProvider._normalize_column_type(
                        row_dict.get("formatted_type"),
                        row_dict.get("data_type"),
                        row_dict.get("udt_name"),
                    ),
                    "data_type": to_text(row_dict.get("data_type")).lower(),
                    "udt_name": to_text(row_dict.get("udt_name")).lower(),
                }
            )
        return tables

    @staticmethod
    def _fetch_spatial_fields(cursor, schema_name: str, requested_tables: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
        tables: dict[str, list[dict[str, Any]]] = {}
        for query, spatial_type in (
            (
                """
                SELECT
                    f_table_name AS table_name,
                    f_geometry_column AS column_name,
                    type,
                    srid
                FROM public.geometry_columns
                WHERE f_table_schema = %s
                  AND f_table_name = ANY(%s)
                ORDER BY f_table_name, f_geometry_column
                """,
                "geometry",
            ),
            (
                """
                SELECT
                    f_table_name AS table_name,
                    f_geography_column AS column_name,
                    type,
                    srid
                FROM public.geography_columns
                WHERE f_table_schema = %s
                  AND f_table_name = ANY(%s)
                ORDER BY f_table_name, f_geography_column
                """,
                "geography",
            ),
        ):
            try:
                cursor.execute(query, (schema_name, list(requested_tables)))
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
                    {
                        "column_name": column_name,
                        "column_type": type_display,
                        "spatial_type": spatial_type,
                        "geometry_type": geometry_type,
                        "srid": srid,
                    }
                )
        return tables

    def _fetch_representative_values(
        self,
        cursor,
        *,
        schema_name: str,
        table_name: str,
        columns: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        if not columns:
            return []
        query = self._build_sample_query(schema_name=schema_name, table_name=table_name, columns=columns)
        cursor.execute(query, (self.MAX_SAMPLE_ROWS,))
        rows = cursor.fetchall() or []
        representative_rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            normalized_row: dict[str, Any] = {}
            for column in columns:
                column_name = to_text(column.get("column_name"))
                if not column_name:
                    continue
                normalized_row[column_name] = self._normalize_sample_value(row.get(column_name))
            signature = json.dumps(normalized_row, ensure_ascii=False, sort_keys=True)
            if signature in seen:
                continue
            seen.add(signature)
            representative_rows.append(normalized_row)
            if len(representative_rows) >= self.MAX_REPRESENTATIVE_ROWS:
                break
        return representative_rows

    def _build_sample_query(
        self,
        *,
        schema_name: str,
        table_name: str,
        columns: Sequence[Mapping[str, Any]],
    ):
        select_items = []
        for column in columns:
            column_name = to_text(column.get("column_name"))
            if not column_name:
                continue
            identifier = pg_sql.Identifier(column_name)
            alias = pg_sql.Identifier(column_name)
            udt_name = to_text(column.get("udt_name")).lower()
            if udt_name in {"geometry", "geography"}:
                geometry_expr = identifier if udt_name == "geometry" else pg_sql.SQL("{}::geometry").format(identifier)
                select_items.append(
                    pg_sql.SQL(
                        "CASE WHEN {column} IS NULL THEN NULL ELSE "
                        "CONCAT(COALESCE(REPLACE(ST_GeometryType({geometry_expr}), 'ST_', ''), 'GEOMETRY'), "
                        "' (SRID=', COALESCE(ST_SRID({geometry_expr})::text, 'unknown'), ')') END AS {alias}"
                    ).format(
                        column=identifier,
                        geometry_expr=geometry_expr,
                        alias=alias,
                    )
                )
            elif udt_name == "bytea":
                continue
            else:
                select_items.append(
                    pg_sql.SQL("{column} AS {alias}").format(
                        column=identifier,
                        alias=alias,
                    )
                )
        if not select_items:
            select_items = [pg_sql.SQL("1 AS sample_placeholder")]
        return pg_sql.SQL("SELECT {fields} FROM {schema}.{table} LIMIT %s").format(
            fields=pg_sql.SQL(", ").join(select_items),
            schema=pg_sql.Identifier(schema_name),
            table=pg_sql.Identifier(table_name),
        )

    @staticmethod
    def _normalize_column_type(formatted_type: Any, data_type: Any, udt_name: Any) -> str:
        formatted = " ".join(to_text(formatted_type).split())
        if formatted:
            return formatted
        data_type_text = to_text(data_type).lower()
        udt_text = to_text(udt_name).lower()
        if data_type_text == "USER-DEFINED".lower() and udt_text:
            return udt_text
        return data_type_text or udt_text or "text"

    @classmethod
    def _normalize_sample_value(cls, value: Any) -> Any:
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
        if isinstance(value, (datetime, date, time)):
            return value.isoformat()
        if isinstance(value, Mapping):
            return stable_jsonify(value)
        if isinstance(value, (list, tuple)):
            return stable_jsonify(value)
        text = " ".join(str(value).strip().split())
        if not text:
            return ""
        if len(text) > cls.MAX_TEXT_LENGTH:
            text = text[: cls.MAX_TEXT_LENGTH - 3].rstrip() + "..."
        return text
