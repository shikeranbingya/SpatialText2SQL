"""Configuration handling for diversity-aware question generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.synthesis.database.utils import stable_jsonify, to_text

from .models import QUESTION_STYLES


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


DEFAULT_QUESTION_GENERATION_CONFIG_PATH = _project_root() / "config" / "question_generation.yaml"


@dataclass(frozen=True)
class QuestionGenerationLLMConfig:
    provider: str = "openai_compatible"
    model: str = "gpt-4o-mini"
    base_url: str = "http://localhost:8000/v1"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.2
    max_tokens: int = 800
    timeout: int = 120
    max_retries: int = 2


@dataclass(frozen=True)
class QuestionGenerationRunConfig:
    sql_input_path: str = str(_project_root() / "data" / "processed" / "synthesized_sql_queries.jsonl")
    database_context_path: str = str(_project_root() / "data" / "processed" / "synthesized_spatial_databases.jsonl")
    output_path: str = str(_project_root() / "data" / "processed" / "diversity_aware_questions.jsonl")
    num_questions_per_sql: int = 1
    fixed_style: str = ""
    style_weights: dict[str, float] = field(default_factory=lambda: {style: 1.0 for style in QUESTION_STYLES})
    random_seed: int = 42
    keep_invalid: bool = False
    max_revision_rounds: int = 1


@dataclass(frozen=True)
class QuestionGenerationLoggingConfig:
    log_level: str = "INFO"
    log_path: str = ""


@dataclass(frozen=True)
class QuestionGenerationConfig:
    llm: QuestionGenerationLLMConfig = field(default_factory=QuestionGenerationLLMConfig)
    generation: QuestionGenerationRunConfig = field(default_factory=QuestionGenerationRunConfig)
    logging: QuestionGenerationLoggingConfig = field(default_factory=QuestionGenerationLoggingConfig)


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


def _normalize_style(value: Any) -> str:
    text = _as_text(value).lower()
    if not text:
        return ""
    if text not in QUESTION_STYLES:
        raise ValueError(f"Unsupported question style: {value!r}")
    return text


def _normalize_style_weights(value: Any) -> dict[str, float]:
    if value in (None, ""):
        return {style: 1.0 for style in QUESTION_STYLES}
    if isinstance(value, str):
        parsed: dict[str, float] = {}
        for part in [item.strip() for item in value.split(",") if item.strip()]:
            if "=" not in part:
                raise ValueError(f"Invalid style weight item: {part!r}")
            key, raw_weight = part.split("=", 1)
            parsed[key.strip().lower()] = float(raw_weight)
        value = parsed
    if not isinstance(value, Mapping):
        raise ValueError("style_weights must be a mapping or comma-separated string.")
    weights = {style: float(value.get(style, 0.0)) for style in QUESTION_STYLES}
    if sum(max(weight, 0.0) for weight in weights.values()) <= 0:
        raise ValueError("style_weights must contain at least one positive weight.")
    return weights


def load_question_generation_config(config_path: str | Path | None = None) -> QuestionGenerationConfig:
    path = Path(config_path or DEFAULT_QUESTION_GENERATION_CONFIG_PATH)
    if not path.is_file():
        raise FileNotFoundError(f"Question generation config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return _build_question_generation_config_from_payload(payload, path)


def _build_question_generation_config_from_payload(
    payload: Mapping[str, Any],
    path: Path,
) -> QuestionGenerationConfig:
    if not isinstance(payload, Mapping):
        raise ValueError(f"Invalid question generation config in {path}: top level must be a mapping.")

    llm_section = payload.get("llm") or {}
    generation_section = payload.get("generation") or {}
    logging_section = payload.get("logging") or {}
    for section_name, section in (
        ("llm", llm_section),
        ("generation", generation_section),
        ("logging", logging_section),
    ):
        if section and not isinstance(section, Mapping):
            raise ValueError(f"Invalid question generation config: '{section_name}' must be a mapping.")

    default_llm = QuestionGenerationLLMConfig()
    default_generation = QuestionGenerationRunConfig()
    default_logging = QuestionGenerationLoggingConfig()

    return QuestionGenerationConfig(
        llm=QuestionGenerationLLMConfig(
            provider=_as_text(llm_section.get("provider"), default_llm.provider),
            model=_as_text(llm_section.get("model"), default_llm.model),
            base_url=_as_text(llm_section.get("base_url"), default_llm.base_url),
            api_key_env=_as_text(llm_section.get("api_key_env"), default_llm.api_key_env),
            temperature=_as_float(llm_section.get("temperature"), default_llm.temperature),
            max_tokens=_as_positive_int(llm_section.get("max_tokens"), default_llm.max_tokens),
            timeout=_as_positive_int(llm_section.get("timeout"), default_llm.timeout),
            max_retries=_as_non_negative_int(llm_section.get("max_retries"), default_llm.max_retries),
        ),
        generation=QuestionGenerationRunConfig(
            sql_input_path=_resolve_path(generation_section.get("sql_input_path"), path, default_generation.sql_input_path),
            database_context_path=_resolve_path(
                generation_section.get("database_context_path"),
                path,
                default_generation.database_context_path,
            ),
            output_path=_resolve_path(generation_section.get("output_path"), path, default_generation.output_path),
            num_questions_per_sql=_as_positive_int(
                generation_section.get("num_questions_per_sql"),
                default_generation.num_questions_per_sql,
            ),
            fixed_style=_normalize_style(generation_section.get("style") or generation_section.get("fixed_style")),
            style_weights=_normalize_style_weights(
                generation_section.get("style_weights", default_generation.style_weights)
            ),
            random_seed=int(generation_section.get("random_seed", default_generation.random_seed)),
            keep_invalid=_as_bool(generation_section.get("keep_invalid"), default_generation.keep_invalid),
            max_revision_rounds=_as_non_negative_int(
                generation_section.get("max_revision_rounds"),
                default_generation.max_revision_rounds,
            ),
        ),
        logging=QuestionGenerationLoggingConfig(
            log_level=_as_text(logging_section.get("log_level"), default_logging.log_level),
            log_path=_resolve_path(logging_section.get("log_path"), path, default_logging.log_path)
            if to_text(logging_section.get("log_path"))
            else default_logging.log_path,
        ),
    )


def override_question_generation_config(
    base: QuestionGenerationConfig,
    *,
    llm: Mapping[str, Any] | None = None,
    generation: Mapping[str, Any] | None = None,
    logging: Mapping[str, Any] | None = None,
) -> QuestionGenerationConfig:
    merged = {
        "llm": {**base.llm.__dict__, **dict(llm or {})},
        "generation": {**base.generation.__dict__, **dict(generation or {})},
        "logging": {**base.logging.__dict__, **dict(logging or {})},
    }
    return _build_question_generation_config_from_payload(
        stable_jsonify(merged),
        DEFAULT_QUESTION_GENERATION_CONFIG_PATH,
    )
