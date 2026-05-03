"""CLI for constraint-guided SQL synthesis."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.prompting.prompt_builder import PromptBuilder

from .config import DEFAULT_SQL_SYNTHESIS_CONFIG_PATH, load_sql_synthesis_config, override_sql_synthesis_config
from .execution import SQLExecutionChecker
from .function_library import PostGISFunctionLibrary
from .generator import build_sql_generator
from .io import load_input_databases, write_sql_queries
from .synthesizer import ConstraintGuidedSQLSynthesizer
from .validator import SQLValidator


def _csv_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate constrained PostGIS SQL samples.")
    parser.add_argument("--config", default=str(DEFAULT_SQL_SYNTHESIS_CONFIG_PATH))
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--num-sql-per-database")
    parser.add_argument("--difficulty")
    parser.add_argument("--difficulty-weights")
    parser.add_argument("--postgis-function-json-path")
    parser.add_argument("--st-function-markdown-path")
    parser.add_argument("--exclude-categories")
    parser.add_argument("--provider")
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--max-retries", type=int)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--database")
    parser.add_argument("--user")
    parser.add_argument("--password")
    parser.add_argument("--search-path")
    parser.add_argument("--connect-timeout", type=int)
    parser.add_argument("--statement-timeout", type=int)
    parser.add_argument("--random-seed", type=int)
    parser.add_argument("--max-revision-rounds", type=int)
    parser.add_argument("--enable-execution-check", dest="enable_execution_check", action="store_true")
    parser.add_argument("--disable-execution-check", dest="enable_execution_check", action="store_false")
    parser.set_defaults(enable_execution_check=None)
    parser.add_argument("--require-non-empty-result", dest="require_non_empty_result", action="store_true")
    parser.add_argument("--allow-empty-result", dest="require_non_empty_result", action="store_false")
    parser.set_defaults(require_non_empty_result=None)
    parser.add_argument("--keep-invalid", action="store_true")
    parser.add_argument("--keep-failed-execution", action="store_true")
    parser.add_argument("--execution-timeout", type=int)
    parser.add_argument("--max-result-rows-for-check", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--explain-only", action="store_true")
    parser.add_argument("--log-level")
    parser.add_argument("--log-path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    config = load_sql_synthesis_config(args.config)
    config = override_sql_synthesis_config(
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
        llm={key: value for key, value in {
            "provider": args.provider,
            "model": args.model,
            "base_url": args.base_url,
            "api_key_env": args.api_key_env,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "timeout": args.timeout,
            "max_retries": args.max_retries,
        }.items() if value is not None},
        synthesis={key: value for key, value in {
            "input_path": args.input,
            "output_path": args.output,
            "num_sql_per_database": args.num_sql_per_database,
            "difficulty": args.difficulty,
            "difficulty_weights": args.difficulty_weights,
            "random_seed": args.random_seed,
            "keep_invalid": args.keep_invalid if args.keep_invalid else None,
            "keep_failed_execution": args.keep_failed_execution if args.keep_failed_execution else None,
            "max_revision_rounds": args.max_revision_rounds,
        }.items() if value is not None},
        functions={key: value for key, value in {
            "postgis_function_json_path": args.postgis_function_json_path,
            "st_function_markdown_path": args.st_function_markdown_path,
            "exclude_categories": _csv_list(args.exclude_categories) if args.exclude_categories else None,
        }.items() if value is not None},
        execution={key: value for key, value in {
            "enable_execution_check": args.enable_execution_check,
            "require_non_empty_result": args.require_non_empty_result,
            "max_result_rows_for_check": args.max_result_rows_for_check,
            "execution_timeout": args.execution_timeout,
            "dry_run": args.dry_run if args.dry_run else None,
            "explain_only": args.explain_only if args.explain_only else None,
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

    databases = load_input_databases(config.synthesis.input_path)
    if not databases:
        raise ValueError("Input synthesized database file is empty.")

    function_library = PostGISFunctionLibrary.load(
        config.functions.postgis_function_json_path,
        config.functions.st_function_markdown_path,
        config.functions.exclude_categories,
    )
    generator = build_sql_generator(
        provider=config.llm.provider,
        model=config.llm.model,
        base_url=config.llm.base_url,
        api_key_env=config.llm.api_key_env,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
        timeout=config.llm.timeout,
        max_retries=config.llm.max_retries,
    )
    prompt_builder = PromptBuilder({"project_root": Path(__file__).resolve().parents[3]})
    validator = SQLValidator(function_library)
    execution_checker = SQLExecutionChecker(config.database, config.execution)
    synthesizer = ConstraintGuidedSQLSynthesizer(
        config=config,
        function_library=function_library,
        sql_generator=generator,
        prompt_builder=prompt_builder,
        validator=validator,
        execution_checker=execution_checker,
    )
    rows = synthesizer.synthesize_all(databases)
    write_sql_queries(config.synthesis.output_path, rows)
    logging.info("Wrote %s SQL samples to %s", len(rows), config.synthesis.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
