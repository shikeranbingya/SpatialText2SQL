"""Execution checking for synthesized SQL queries."""

from __future__ import annotations

import logging
import time
from typing import Any, Sequence

import psycopg2
from psycopg2 import sql as pg_sql
from psycopg2.extras import RealDictCursor

from src.synthesis.database.migration import normalize_postgres_identifier
from src.synthesis.database.models import SynthesizedSpatialDatabase
from src.synthesis.database.utils import stable_jsonify, to_text

from .config import SQLExecutionCheckConfig, SQLSynthesisDBConfig
from .models import SQLExecutionResult
from .validator import contains_dangerous_sql

LOGGER = logging.getLogger(__name__)


class SQLExecutionChecker:
    def __init__(
        self,
        db_config: SQLSynthesisDBConfig,
        execution_config: SQLExecutionCheckConfig,
    ) -> None:
        self.db_config = db_config
        self.execution_config = execution_config

    def check(self, sql: str, database: SynthesizedSpatialDatabase) -> SQLExecutionResult:
        sql_text = to_text(sql).rstrip(";")
        if not self.execution_config.enable_execution_check or self.execution_config.dry_run:
            LOGGER.info(
                "Execution skipped | schema_id=%s | enable_execution_check=%s | dry_run=%s",
                database.database_id,
                self.execution_config.enable_execution_check,
                self.execution_config.dry_run,
            )
            return SQLExecutionResult(executed=False, success=True)
        if contains_dangerous_sql(sql_text):
            return SQLExecutionResult(
                executed=False,
                success=False,
                error_message="Refused to execute non-read-only SQL.",
            )
        catalog_name = (
            normalize_postgres_identifier(self.db_config.database, prefix="catalog")
            or self.db_config.database
        )
        schema_name = normalize_postgres_identifier(database.database_id, prefix="schema")
        actual_database = f"{catalog_name}.{schema_name}"
        start = time.perf_counter()
        LOGGER.info(
            "Execution connect start | schema_id=%s | target=%s | sql_chars=%s",
            database.database_id,
            actual_database,
            len(sql_text),
        )
        try:
            conn = self._connect(catalog_name)
        except Exception as exc:
            return SQLExecutionResult(
                executed=False,
                success=False,
                error_message=f"Database connection failed: {exc}",
                actual_database=actual_database,
            )

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                self._apply_session_settings(cur, schema_name)
                execution_sql = f"EXPLAIN {sql_text}" if self.execution_config.explain_only else sql_text
                LOGGER.info(
                    "Execution query start | schema_id=%s | target=%s | explain_only=%s",
                    database.database_id,
                    actual_database,
                    self.execution_config.explain_only,
                )
                cur.execute(execution_sql)
                if self.execution_config.explain_only:
                    rows = cur.fetchmany(self.execution_config.max_result_rows_for_check)
                else:
                    rows = cur.fetchmany(self.execution_config.max_result_rows_for_check)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            sample_rows = [dict(row) for row in rows] if rows else []
            empty_result = len(sample_rows) == 0
            success = True
            LOGGER.info(
                "Execution query done | schema_id=%s | target=%s | success=%s | empty_result=%s | row_count=%s | time_ms=%.1f",
                database.database_id,
                actual_database,
                success,
                empty_result,
                len(sample_rows),
                elapsed_ms,
            )
            return SQLExecutionResult(
                executed=True,
                success=success,
                error_message="",
                row_count=len(sample_rows),
                empty_result=empty_result,
                sample_rows=stable_jsonify(sample_rows),
                execution_time_ms=elapsed_ms,
                actual_database=actual_database,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            LOGGER.warning(
                "Execution query failed | schema_id=%s | target=%s | time_ms=%.1f | error=%s",
                database.database_id,
                actual_database,
                elapsed_ms,
                exc,
            )
            return SQLExecutionResult(
                executed=True,
                success=False,
                error_message=str(exc),
                row_count=0,
                empty_result=False,
                sample_rows=[],
                execution_time_ms=elapsed_ms,
                actual_database=actual_database,
            )
        finally:
            conn.close()

    def _connect(self, actual_database: str):
        return psycopg2.connect(
            host=self.db_config.host,
            port=self.db_config.port,
            dbname=actual_database,
            user=self.db_config.user,
            password=self.db_config.password,
            connect_timeout=self.db_config.connect_timeout,
        )

    def _apply_session_settings(self, cursor, schema_name: str) -> None:
        cursor.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
        cursor.execute("SET statement_timeout = %s", (int(self.execution_config.execution_timeout * 1000),))
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
