"""Configuration handling for SQL synthesis."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.synthesis.database.utils import stable_jsonify, to_text

from .models import DIFFICULTY_LEVELS


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


DEFAULT_SQL_SYNTHESIS_CONFIG_PATH = _project_root() / "config" / "sql_synthesis.yaml"


@dataclass(frozen=True)
class SQLSynthesisDBConfig:
    host: str = "localhost"
    port: int = 5432
    database: str = "postgres"
    user: str = "postgres"
    password: str = "123456"
    search_path: str = "public"
    connect_timeout: int = 10
    statement_timeout: int = 60000


@dataclass(frozen=True)
class SQLSynthesisLLMConfig:
    provider: str = "openai_compatible"
    model: str = "gpt-4o-mini"
    base_url: str = "http://localhost:8000/v1"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.2
    max_tokens: int = 1200
    timeout: int = 120
    max_retries: int = 2


@dataclass(frozen=True)
class SQLSynthesisFunctionConfig:
    postgis_function_json_path: str = str(_project_root() / "data" / "postgis_extracted.json")
    st_function_markdown_path: str = str(_project_root() / "ST_Function.md")
    exclude_categories: list[str] = field(default_factory=lambda: ["raster", "topology"])


@dataclass(frozen=True)
class SQLExecutionCheckConfig:
    enable_execution_check: bool = True
    require_non_empty_result: bool = True
    max_result_rows_for_check: int = 20
    execution_timeout: int = 60
    dry_run: bool = False
    explain_only: bool = False


@dataclass(frozen=True)
class SQLSynthesisRunConfig:
    input_path: str = str(_project_root() / "data" / "processed" / "synthesized_spatial_databases.jsonl")
    output_path: str = str(_project_root() / "data" / "processed" / "synthesized_sql_queries.jsonl")
    num_sql_per_database: dict[str, int] = field(default_factory=lambda: {"default": 5})
    fixed_difficulty: str = ""
    difficulty_weights: dict[str, float] = field(
        default_factory=lambda: {level: 1.0 for level in DIFFICULTY_LEVELS}
    )
    random_seed: int = 42
    keep_invalid: bool = False
    keep_failed_execution: bool = False
    max_revision_rounds: int = 2


@dataclass(frozen=True)
class SQLSynthesisLoggingConfig:
    log_level: str = "INFO"
    log_path: str = ""


@dataclass(frozen=True)
class SQLSynthesisConfig:
    database: SQLSynthesisDBConfig = field(default_factory=SQLSynthesisDBConfig)
    llm: SQLSynthesisLLMConfig = field(default_factory=SQLSynthesisLLMConfig)
    synthesis: SQLSynthesisRunConfig = field(default_factory=SQLSynthesisRunConfig)
    functions: SQLSynthesisFunctionConfig = field(default_factory=SQLSynthesisFunctionConfig)
    execution: SQLExecutionCheckConfig = field(default_factory=SQLExecutionCheckConfig)
    logging: SQLSynthesisLoggingConfig = field(default_factory=SQLSynthesisLoggingConfig)


def _as_positive_int(value: Any, default: int) -> int:
    if value in (None, ""):
        return default
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"Expected a positive integer, got {value!r}")
    return parsed


def _as_non_negative_int(value: Any, default: int) -> int:
    if value in (None, ""):
        return default
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"Expected a non-negative integer, got {value!r}")
    return parsed


def _as_bool(value: Any, default: bool) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected a boolean-like value, got {value!r}")


def _as_float(value: Any, default: float) -> float:
    if value in (None, ""):
        return default
    return float(value)


def _as_text(value: Any, default: str = "") -> str:
    text = to_text(value)
    return text if text else default


def _resolve_path(value: Any, config_path: Path, default: str) -> str:
    text = _as_text(value)
    if not text:
        return default
    path = Path(text)
    if path.is_absolute():
        return str(path)
    return str((config_path.parent.parent / path).resolve())


def _normalize_weights(value: Any) -> dict[str, float]:
    if value in (None, ""):
        return {level: 1.0 for level in DIFFICULTY_LEVELS}
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        parsed: dict[str, float] = {}
        for part in parts:
            if "=" not in part:
                raise ValueError(f"Invalid difficulty weight item: {part!r}")
            key, raw_weight = part.split("=", 1)
            parsed[key.strip().lower()] = float(raw_weight)
        value = parsed
    if not isinstance(value, Mapping):
        raise ValueError("difficulty_weights must be a mapping or comma-separated string.")
    weights = {level: float(value.get(level, 0.0)) for level in DIFFICULTY_LEVELS}
    if sum(max(weight, 0.0) for weight in weights.values()) <= 0:
        raise ValueError("difficulty_weights must contain at least one positive weight.")
    return weights


def _normalize_num_sql_per_database(value: Any, default: Mapping[str, int]) -> dict[str, int]:
    if value in (None, ""):
        return {str(key): int(val) for key, val in default.items()}
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {str(key): int(val) for key, val in default.items()}
        if "=" not in text:
            parsed = int(text)
            if parsed <= 0:
                raise ValueError("num_sql_per_database must be positive.")
            return {"default": parsed}
        parsed_map: dict[str, int] = {}
        for item in text.split(","):
            entry = item.strip()
            if not entry:
                continue
            if "=" not in entry:
                raise ValueError(
                    "num_sql_per_database string must be a positive integer or comma-separated city=count pairs."
                )
            key, raw_value = entry.split("=", 1)
            city = key.strip().lower()
            count = int(raw_value.strip())
            if count <= 0:
                raise ValueError("num_sql_per_database values must be positive.")
            parsed_map[city] = count
        if not parsed_map:
            raise ValueError("num_sql_per_database mapping cannot be empty.")
        return parsed_map
    if isinstance(value, Mapping):
        parsed_map = {}
        for key, raw_value in value.items():
            city = to_text(key).lower()
            if not city:
                continue
            count = int(raw_value)
            if count <= 0:
                raise ValueError("num_sql_per_database values must be positive.")
            parsed_map[city] = count
        if not parsed_map:
            raise ValueError("num_sql_per_database mapping cannot be empty.")
        return parsed_map
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("num_sql_per_database must be positive.")
    return {"default": parsed}


def _normalize_text_list(value: Any, default: list[str]) -> list[str]:
    if value in (None, ""):
        return list(default)
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    normalized = stable_jsonify(value)
    if isinstance(normalized, list):
        return [to_text(item) for item in normalized if to_text(item)]
    return list(default)


def _normalize_difficulty(value: Any) -> str:
    text = _as_text(value).lower()
    if not text:
        return ""
    if text not in DIFFICULTY_LEVELS:
        raise ValueError(f"Unsupported difficulty: {value!r}")
    return text


def load_sql_synthesis_config(config_path: str | Path | None = None) -> SQLSynthesisConfig:
    path = Path(config_path or DEFAULT_SQL_SYNTHESIS_CONFIG_PATH)
    if not path.is_file():
        raise FileNotFoundError(f"SQL synthesis config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return _build_sql_synthesis_config_from_payload(payload, path)


def _build_sql_synthesis_config_from_payload(
    payload: Mapping[str, Any],
    path: Path,
) -> SQLSynthesisConfig:
    if not isinstance(payload, Mapping):
        raise ValueError(f"Invalid SQL synthesis config in {path}: top level must be a mapping.")

    db_section = payload.get("database") or {}
    llm_section = payload.get("llm") or {}
    synthesis_section = payload.get("synthesis") or {}
    function_section = payload.get("functions") or {}
    execution_section = payload.get("execution") or {}
    logging_section = payload.get("logging") or {}
    for section_name, section in (
        ("database", db_section),
        ("llm", llm_section),
        ("synthesis", synthesis_section),
        ("functions", function_section),
        ("execution", execution_section),
        ("logging", logging_section),
    ):
        if section and not isinstance(section, Mapping):
            raise ValueError(f"Invalid SQL synthesis config: '{section_name}' must be a mapping.")

    default_db = SQLSynthesisDBConfig()
    default_llm = SQLSynthesisLLMConfig()
    default_syn = SQLSynthesisRunConfig()
    default_func = SQLSynthesisFunctionConfig()
    default_exec = SQLExecutionCheckConfig()
    default_log = SQLSynthesisLoggingConfig()

    return SQLSynthesisConfig(
        database=SQLSynthesisDBConfig(
            host=_as_text(db_section.get("host"), default_db.host),
            port=_as_positive_int(db_section.get("port"), default_db.port),
            database=_as_text(db_section.get("database"), default_db.database),
            user=_as_text(db_section.get("user"), default_db.user),
            password=_as_text(db_section.get("password"), default_db.password),
            search_path=_as_text(db_section.get("search_path") or db_section.get("schema"), default_db.search_path),
            connect_timeout=_as_positive_int(db_section.get("connect_timeout"), default_db.connect_timeout),
            statement_timeout=_as_positive_int(db_section.get("statement_timeout"), default_db.statement_timeout),
        ),
        llm=SQLSynthesisLLMConfig(
            provider=_as_text(llm_section.get("provider"), default_llm.provider),
            model=_as_text(llm_section.get("model"), default_llm.model),
            base_url=_as_text(llm_section.get("base_url"), default_llm.base_url),
            api_key_env=_as_text(llm_section.get("api_key_env"), default_llm.api_key_env),
            temperature=_as_float(llm_section.get("temperature"), default_llm.temperature),
            max_tokens=_as_positive_int(llm_section.get("max_tokens"), default_llm.max_tokens),
            timeout=_as_positive_int(llm_section.get("timeout"), default_llm.timeout),
            max_retries=_as_non_negative_int(llm_section.get("max_retries"), default_llm.max_retries),
        ),
        synthesis=SQLSynthesisRunConfig(
            input_path=_resolve_path(synthesis_section.get("input_path"), path, default_syn.input_path),
            output_path=_resolve_path(synthesis_section.get("output_path"), path, default_syn.output_path),
            num_sql_per_database=_normalize_num_sql_per_database(
                synthesis_section.get("num_sql_per_database"),
                default_syn.num_sql_per_database,
            ),
            fixed_difficulty=_normalize_difficulty(synthesis_section.get("difficulty") or synthesis_section.get("fixed_difficulty")),
            difficulty_weights=_normalize_weights(synthesis_section.get("difficulty_weights", default_syn.difficulty_weights)),
            random_seed=int(synthesis_section.get("random_seed", default_syn.random_seed)),
            keep_invalid=_as_bool(synthesis_section.get("keep_invalid"), default_syn.keep_invalid),
            keep_failed_execution=_as_bool(
                synthesis_section.get("keep_failed_execution"),
                default_syn.keep_failed_execution,
            ),
            max_revision_rounds=_as_non_negative_int(
                synthesis_section.get("max_revision_rounds"),
                default_syn.max_revision_rounds,
            ),
        ),
        functions=SQLSynthesisFunctionConfig(
            postgis_function_json_path=_resolve_path(
                function_section.get("postgis_function_json_path"),
                path,
                default_func.postgis_function_json_path,
            ),
            st_function_markdown_path=_resolve_path(
                function_section.get("st_function_markdown_path"),
                path,
                default_func.st_function_markdown_path,
            ),
            exclude_categories=_normalize_text_list(
                function_section.get("exclude_categories"),
                default_func.exclude_categories,
            ),
        ),
        execution=SQLExecutionCheckConfig(
            enable_execution_check=_as_bool(
                execution_section.get("enable_execution_check"),
                default_exec.enable_execution_check,
            ),
            require_non_empty_result=_as_bool(
                execution_section.get("require_non_empty_result"),
                default_exec.require_non_empty_result,
            ),
            max_result_rows_for_check=_as_positive_int(
                execution_section.get("max_result_rows_for_check"),
                default_exec.max_result_rows_for_check,
            ),
            execution_timeout=_as_positive_int(
                execution_section.get("execution_timeout"),
                default_exec.execution_timeout,
            ),
            dry_run=_as_bool(execution_section.get("dry_run"), default_exec.dry_run),
            explain_only=_as_bool(
                execution_section.get("explain_only"),
                default_exec.explain_only,
            ),
        ),
        logging=SQLSynthesisLoggingConfig(
            log_level=_as_text(logging_section.get("log_level"), default_log.log_level),
            log_path=_resolve_path(logging_section.get("log_path"), path, default_log.log_path)
            if to_text(logging_section.get("log_path"))
            else default_log.log_path,
        ),
    )


def override_sql_synthesis_config(
    base: SQLSynthesisConfig,
    *,
    database: Mapping[str, Any] | None = None,
    llm: Mapping[str, Any] | None = None,
    synthesis: Mapping[str, Any] | None = None,
    functions: Mapping[str, Any] | None = None,
    execution: Mapping[str, Any] | None = None,
    logging: Mapping[str, Any] | None = None,
) -> SQLSynthesisConfig:
    merged = {
        "database": {**base.database.__dict__, **dict(database or {})},
        "llm": {**base.llm.__dict__, **dict(llm or {})},
        "synthesis": {**base.synthesis.__dict__, **dict(synthesis or {})},
        "functions": {**base.functions.__dict__, **dict(functions or {})},
        "execution": {**base.execution.__dict__, **dict(execution or {})},
        "logging": {**base.logging.__dict__, **dict(logging or {})},
    }
    return _build_sql_synthesis_config_from_payload(
        merged,
        DEFAULT_SQL_SYNTHESIS_CONFIG_PATH,
    )
