"""CLI for NL-SQL quality control."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.synthesis.sql.function_library import PostGISFunctionLibrary

from .config import (
    DEFAULT_QUALITY_CONTROL_CONFIG_PATH,
    load_quality_control_config,
    override_quality_control_config,
)
from .database import PostgreSQLDatabaseRegistry
from .io import (
    load_nl_sql_samples,
    load_schema_registry_from_contexts,
    write_nl_sql_samples,
    write_quality_control_report,
)
from .pipeline import QualityControlPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run quality control over synthetic spatial NL-SQL samples.")
    parser.add_argument("--config", default=str(DEFAULT_QUALITY_CONTROL_CONFIG_PATH))
    parser.add_argument("--input")
    parser.add_argument("--schema-context-path")
    parser.add_argument("--output")
    parser.add_argument("--report-path")
    parser.add_argument("--allow-empty-result", action="store_true")
    parser.add_argument("--semantic-mode")
    parser.add_argument("--debug-mode", action="store_true")
    parser.add_argument("--max-result-rows", type=int)
    parser.add_argument("--question-similarity-threshold", type=float)
    parser.add_argument("--same-sql-similarity-threshold", type=float)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--database")
    parser.add_argument("--user")
    parser.add_argument("--password")
    parser.add_argument("--search-path")
    parser.add_argument("--connect-timeout", type=int)
    parser.add_argument("--statement-timeout", type=int)
    parser.add_argument("--log-level")
    parser.add_argument("--log-path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    config = load_quality_control_config(args.config)
    config = override_quality_control_config(
        config,
        database={key: value for key, value in {
            "host": args.host,
            "port": args.port,
            "database": args.database,
            "user": args.user,
            "password": args.password,
            "search_path": args.search_path,
            "connect_timeout": args.connect_timeout,
            "statement_timeout": args.statement_timeout,
        }.items() if value is not None},
        run={key: value for key, value in {
            "input_path": args.input,
            "schema_context_path": args.schema_context_path,
            "output_path": args.output,
            "report_path": args.report_path,
            "allow_empty_result": args.allow_empty_result if args.allow_empty_result else None,
            "max_result_rows": args.max_result_rows,
        }.items() if value is not None},
        semantic={key: value for key, value in {
            "mode": args.semantic_mode,
            "debug_mode": args.debug_mode if args.debug_mode else None,
        }.items() if value is not None},
        duplicates={key: value for key, value in {
            "question_similarity_threshold": args.question_similarity_threshold,
            "same_sql_similarity_threshold": args.same_sql_similarity_threshold,
        }.items() if value is not None},
        logging={key: value for key, value in {
            "log_level": args.log_level,
            "log_path": args.log_path,
        }.items() if value is not None},
    )

    log_handlers = None
    if config.logging.log_path:
        Path(config.logging.log_path).parent.mkdir(parents=True, exist_ok=True)
        log_handlers = [logging.FileHandler(config.logging.log_path, encoding="utf-8"), logging.StreamHandler()]
    logging.basicConfig(
        level=getattr(logging, config.logging.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=log_handlers,
    )
    logging.info(
        "Quality control config loaded | input=%s | schema_context=%s | output=%s | report=%s",
        config.run.input_path,
        config.run.schema_context_path,
        config.run.output_path,
        config.run.report_path,
    )

    samples = load_nl_sql_samples(config.run.input_path)
    schema_registry = load_schema_registry_from_contexts(config.run.schema_context_path)
    database_registry = PostgreSQLDatabaseRegistry(config.database)
    function_library = PostGISFunctionLibrary.load(
        Path(config.functions.postgis_function_json_path),
        Path(config.functions.st_function_markdown_path),
        list(config.functions.exclude_categories),
    )
    pipeline = QualityControlPipeline(function_library=function_library)
    retained, report = pipeline.run(samples, database_registry, schema_registry, config)
    write_nl_sql_samples(config.run.output_path, retained)
    write_quality_control_report(config.run.report_path, report)
    logging.info(
        "Quality control finished | total=%s | retained=%s | failed=%s | duplicates=%s",
        report.total_samples,
        report.passed_samples,
        report.failed_samples,
        report.duplicate_count,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

