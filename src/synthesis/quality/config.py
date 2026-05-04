"""Configuration handling for NL-SQL quality control."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.synthesis.database.utils import stable_jsonify, to_text


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


DEFAULT_QUALITY_CONTROL_CONFIG_PATH = _project_root() / "config" / "quality_control.yaml"


@dataclass(frozen=True)
class QualityControlDatabaseConfig:
    host: str = "localhost"
    port: int = 5432
    database: str = "syntheized"
    user: str = "postgres"
    password: str = "123456"
    search_path: str = "{schema}, public"
    connect_timeout: int = 10
    statement_timeout: int = 60000


@dataclass(frozen=True)
class QualityControlFunctionConfig:
    postgis_function_json_path: str = str(_project_root() / "data" / "postgis_extracted.json")
    st_function_markdown_path: str = str(_project_root() / "ST_Function.md")
    exclude_categories: list[str] = field(default_factory=lambda: ["raster", "topology"])


@dataclass(frozen=True)
class SemanticCheckConfig:
    mode: str = "strict"
    debug_mode: bool = False


@dataclass(frozen=True)
class DuplicateDetectionConfig:
    remove_exact_sql_duplicates: bool = True
    remove_normalized_sql_duplicates: bool = True
    remove_near_duplicate_questions: bool = True
    question_similarity_threshold: float = 0.92
    same_sql_similarity_threshold: float = 0.85
    treat_same_sql_similar_questions_as_duplicates: bool = True


@dataclass(frozen=True)
class BalanceDimensionConfig:
    max_per_bucket: int = 0
    target_distribution: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class DiversityBalancingConfig:
    enabled: bool = False
    difficulty: BalanceDimensionConfig = field(default_factory=BalanceDimensionConfig)
    spatial_function: BalanceDimensionConfig = field(default_factory=BalanceDimensionConfig)
    linguistic_style: BalanceDimensionConfig = field(default_factory=BalanceDimensionConfig)


@dataclass(frozen=True)
class QualityControlRunConfig:
    input_path: str = str(_project_root() / "data" / "processed" / "diversity_aware_questions.jsonl")
    schema_context_path: str = str(_project_root() / "data" / "processed" / "synthesized_spatial_databases.jsonl")
    output_path: str = str(_project_root() / "data" / "processed" / "quality_controlled_nl_sql.jsonl")
    report_path: str = str(_project_root() / "data" / "processed" / "quality_control_report.json")
    allow_empty_result: bool = False
    max_result_rows: int = 5
    prefer_live_schema: bool = True


@dataclass(frozen=True)
class QualityControlLoggingConfig:
    log_level: str = "INFO"
    log_path: str = ""


@dataclass(frozen=True)
class QualityControlConfig:
    database: QualityControlDatabaseConfig = field(default_factory=QualityControlDatabaseConfig)
    functions: QualityControlFunctionConfig = field(default_factory=QualityControlFunctionConfig)
    run: QualityControlRunConfig = field(default_factory=QualityControlRunConfig)
    semantic: SemanticCheckConfig = field(default_factory=SemanticCheckConfig)
    duplicates: DuplicateDetectionConfig = field(default_factory=DuplicateDetectionConfig)
    balancing: DiversityBalancingConfig = field(default_factory=DiversityBalancingConfig)
    logging: QualityControlLoggingConfig = field(default_factory=QualityControlLoggingConfig)


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


def _as_int(value: Any, default: int, *, minimum: int | None = None) -> int:
    if value in (None, ""):
        return default
    parsed = int(value)
    if minimum is not None and parsed < minimum:
        raise ValueError(f"Expected integer >= {minimum}, got {value!r}")
    return parsed


def _as_float(value: Any, default: float, *, minimum: float | None = None, maximum: float | None = None) -> float:
    if value in (None, ""):
        return default
    parsed = float(value)
    if minimum is not None and parsed < minimum:
        raise ValueError(f"Expected float >= {minimum}, got {value!r}")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"Expected float <= {maximum}, got {value!r}")
    return parsed


def _normalize_mode(value: Any) -> str:
    mode = _as_text(value, "strict").lower()
    if mode not in {"strict", "warning_only"}:
        raise ValueError(f"Unsupported semantic mode: {value!r}")
    return mode


def _normalize_distribution(value: Any) -> dict[str, float]:
    if value in (None, ""):
        return {}
    if isinstance(value, str):
        parsed: dict[str, float] = {}
        for part in [item.strip() for item in value.split(",") if item.strip()]:
            if "=" not in part:
                raise ValueError(f"Invalid distribution item: {part!r}")
            key, raw_weight = part.split("=", 1)
            parsed[key.strip()] = float(raw_weight)
        value = parsed
    if not isinstance(value, Mapping):
        raise ValueError("target_distribution must be a mapping or a comma-separated string.")
    distribution = {str(key): float(raw) for key, raw in value.items()}
    positive_total = sum(weight for weight in distribution.values() if weight > 0)
    if distribution and positive_total <= 0:
        raise ValueError("target_distribution must contain at least one positive weight.")
    return distribution


def _build_balance_dimension(payload: Mapping[str, Any] | None, default: BalanceDimensionConfig) -> BalanceDimensionConfig:
    payload = payload or {}
    return BalanceDimensionConfig(
        max_per_bucket=_as_int(payload.get("max_per_bucket"), default.max_per_bucket, minimum=0),
        target_distribution=_normalize_distribution(payload.get("target_distribution", default.target_distribution)),
    )


def load_quality_control_config(config_path: str | Path | None = None) -> QualityControlConfig:
    path = Path(config_path or DEFAULT_QUALITY_CONTROL_CONFIG_PATH)
    if not path.is_file():
        raise FileNotFoundError(f"Quality control config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return _build_quality_control_config_from_payload(payload, path)


def _build_quality_control_config_from_payload(
    payload: Mapping[str, Any],
    path: Path,
) -> QualityControlConfig:
    if not isinstance(payload, Mapping):
        raise ValueError(f"Invalid quality control config in {path}: top level must be a mapping.")

    db_section = payload.get("database") or {}
    function_section = payload.get("functions") or {}
    run_section = payload.get("run") or {}
    semantic_section = payload.get("semantic") or {}
    duplicate_section = payload.get("duplicates") or {}
    balancing_section = payload.get("balancing") or {}
    logging_section = payload.get("logging") or {}
    default_db = QualityControlDatabaseConfig()
    default_function = QualityControlFunctionConfig()
    default_run = QualityControlRunConfig()
    default_semantic = SemanticCheckConfig()
    default_duplicate = DuplicateDetectionConfig()
    default_balancing = DiversityBalancingConfig()
    default_logging = QualityControlLoggingConfig()

    return QualityControlConfig(
        database=QualityControlDatabaseConfig(
            host=_as_text(db_section.get("host"), default_db.host),
            port=_as_int(db_section.get("port"), default_db.port, minimum=1),
            database=_as_text(db_section.get("database"), default_db.database),
            user=_as_text(db_section.get("user"), default_db.user),
            password=_as_text(db_section.get("password"), default_db.password),
            search_path=_as_text(db_section.get("search_path"), default_db.search_path),
            connect_timeout=_as_int(db_section.get("connect_timeout"), default_db.connect_timeout, minimum=1),
            statement_timeout=_as_int(db_section.get("statement_timeout"), default_db.statement_timeout, minimum=1),
        ),
        functions=QualityControlFunctionConfig(
            postgis_function_json_path=_resolve_path(
                function_section.get("postgis_function_json_path"),
                path,
                default_function.postgis_function_json_path,
            ),
            st_function_markdown_path=_resolve_path(
                function_section.get("st_function_markdown_path"),
                path,
                default_function.st_function_markdown_path,
            ),
            exclude_categories=[to_text(item).lower() for item in function_section.get("exclude_categories", default_function.exclude_categories)],
        ),
        run=QualityControlRunConfig(
            input_path=_resolve_path(run_section.get("input_path"), path, default_run.input_path),
            schema_context_path=_resolve_path(
                run_section.get("schema_context_path"),
                path,
                default_run.schema_context_path,
            ),
            output_path=_resolve_path(run_section.get("output_path"), path, default_run.output_path),
            report_path=_resolve_path(run_section.get("report_path"), path, default_run.report_path),
            allow_empty_result=_as_bool(run_section.get("allow_empty_result"), default_run.allow_empty_result),
            max_result_rows=_as_int(run_section.get("max_result_rows"), default_run.max_result_rows, minimum=1),
            prefer_live_schema=_as_bool(run_section.get("prefer_live_schema"), default_run.prefer_live_schema),
        ),
        semantic=SemanticCheckConfig(
            mode=_normalize_mode(semantic_section.get("mode") or default_semantic.mode),
            debug_mode=_as_bool(semantic_section.get("debug_mode"), default_semantic.debug_mode),
        ),
        duplicates=DuplicateDetectionConfig(
            remove_exact_sql_duplicates=_as_bool(
                duplicate_section.get("remove_exact_sql_duplicates"),
                default_duplicate.remove_exact_sql_duplicates,
            ),
            remove_normalized_sql_duplicates=_as_bool(
                duplicate_section.get("remove_normalized_sql_duplicates"),
                default_duplicate.remove_normalized_sql_duplicates,
            ),
            remove_near_duplicate_questions=_as_bool(
                duplicate_section.get("remove_near_duplicate_questions"),
                default_duplicate.remove_near_duplicate_questions,
            ),
            question_similarity_threshold=_as_float(
                duplicate_section.get("question_similarity_threshold"),
                default_duplicate.question_similarity_threshold,
                minimum=0.0,
                maximum=1.0,
            ),
            same_sql_similarity_threshold=_as_float(
                duplicate_section.get("same_sql_similarity_threshold"),
                default_duplicate.same_sql_similarity_threshold,
                minimum=0.0,
                maximum=1.0,
            ),
            treat_same_sql_similar_questions_as_duplicates=_as_bool(
                duplicate_section.get("treat_same_sql_similar_questions_as_duplicates"),
                default_duplicate.treat_same_sql_similar_questions_as_duplicates,
            ),
        ),
        balancing=DiversityBalancingConfig(
            enabled=_as_bool(balancing_section.get("enabled"), default_balancing.enabled),
            difficulty=_build_balance_dimension(
                balancing_section.get("difficulty") if isinstance(balancing_section, Mapping) else None,
                default_balancing.difficulty,
            ),
            spatial_function=_build_balance_dimension(
                balancing_section.get("spatial_function") if isinstance(balancing_section, Mapping) else None,
                default_balancing.spatial_function,
            ),
            linguistic_style=_build_balance_dimension(
                balancing_section.get("linguistic_style") if isinstance(balancing_section, Mapping) else None,
                default_balancing.linguistic_style,
            ),
        ),
        logging=QualityControlLoggingConfig(
            log_level=_as_text(logging_section.get("log_level"), default_logging.log_level),
            log_path=_resolve_path(logging_section.get("log_path"), path, default_logging.log_path)
            if to_text(logging_section.get("log_path"))
            else default_logging.log_path,
        ),
    )


def override_quality_control_config(
    base: QualityControlConfig,
    *,
    database: Mapping[str, Any] | None = None,
    functions: Mapping[str, Any] | None = None,
    run: Mapping[str, Any] | None = None,
    semantic: Mapping[str, Any] | None = None,
    duplicates: Mapping[str, Any] | None = None,
    balancing: Mapping[str, Any] | None = None,
    logging: Mapping[str, Any] | None = None,
) -> QualityControlConfig:
    merged = {
        "database": {**base.database.__dict__, **dict(database or {})},
        "functions": {**base.functions.__dict__, **dict(functions or {})},
        "run": {**base.run.__dict__, **dict(run or {})},
        "semantic": {**base.semantic.__dict__, **dict(semantic or {})},
        "duplicates": {**base.duplicates.__dict__, **dict(duplicates or {})},
        "balancing": {
            "enabled": base.balancing.enabled,
            "difficulty": base.balancing.difficulty.__dict__,
            "spatial_function": base.balancing.spatial_function.__dict__,
            "linguistic_style": base.balancing.linguistic_style.__dict__,
            **dict(balancing or {}),
        },
        "logging": {**base.logging.__dict__, **dict(logging or {})},
    }
    return _build_quality_control_config_from_payload(
        stable_jsonify(merged),
        DEFAULT_QUALITY_CONTROL_CONFIG_PATH,
    )

