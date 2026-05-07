"""Configuration handling for TRL full fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .utils import stable_jsonify, to_text


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


DEFAULT_TRL_FINETUNE_CONFIG_PATH = _project_root() / "config" / "finetune.yaml"


@dataclass(frozen=True)
class FinetuneDataConfig:
    input_path: str = str(_project_root() / "data" / "processed" / "nl2sql.jsonl")
    alpaca_output_path: str = str(_project_root() / "data" / "processed" / "finetune" / "nl2sql_alpaca.jsonl")
    task_description: str = (
        "Translate the spatial question into one executable PostgreSQL/PostGIS SQL query "
        "using the provided schema and representative values."
    )
    eval_ratio: float = 0.02
    question_id_start: int = 0
    max_representative_rows: int = 3
    shuffle_seed: int = 42


@dataclass(frozen=True)
class FinetuneModelConfig:
    model_name_or_path: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    tokenizer_name_or_path: str = ""
    trust_remote_code: bool = False
    torch_dtype: str = "bfloat16"
    attn_implementation: str = ""


@dataclass(frozen=True)
class FinetuneTrainingConfig:
    output_dir: str = str(_project_root() / "outputs" / "finetune" / "trl_spatial_text2sql_full")
    overwrite_output_dir: bool = False
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-5
    num_train_epochs: float = 3.0
    max_steps: int = -1
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200
    save_total_limit: int = 2
    max_length: int = 4096
    packing: bool = False
    completion_only_loss: bool = True
    gradient_checkpointing: bool = True
    bf16: bool = True
    fp16: bool = False
    dataloader_num_workers: int = 0
    report_to: str = "none"
    seed: int = 42
    resume_from_checkpoint: str = ""
    deepspeed_config_path: str = ""


@dataclass(frozen=True)
class FinetuneLoggingConfig:
    log_level: str = "INFO"
    log_path: str = ""


@dataclass(frozen=True)
class FinetuneRuntimeConfig:
    nvidia_gpu_indices: list[int] = field(default_factory=lambda: list(range(8)))
    distributed_backend: str = "accelerate"
    num_processes: int = 0
    num_machines: int = 1
    machine_rank: int = 0
    main_process_port: int = 29500


@dataclass(frozen=True)
class SpatialText2SQLFinetuneConfig:
    data: FinetuneDataConfig = field(default_factory=FinetuneDataConfig)
    model: FinetuneModelConfig = field(default_factory=FinetuneModelConfig)
    training: FinetuneTrainingConfig = field(default_factory=FinetuneTrainingConfig)
    logging: FinetuneLoggingConfig = field(default_factory=FinetuneLoggingConfig)
    runtime: FinetuneRuntimeConfig = field(default_factory=FinetuneRuntimeConfig)


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


def _as_float(value: Any, default: float) -> float:
    if value in (None, ""):
        return default
    return float(value)


def _as_non_negative_int_list(value: Any, default: list[int]) -> list[int]:
    if value in (None, ""):
        return list(default)
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        parsed = [int(part) for part in parts]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        parsed = [int(item) for item in value]
    else:
        parsed = [int(value)]
    for item in parsed:
        if item < 0:
            raise ValueError(f"Expected non-negative GPU indices, got {value!r}")
    return parsed


def _as_runtime_backend(value: Any, default: str) -> str:
    backend = _as_text(value, default).strip().lower()
    if backend not in {"none", "accelerate"}:
        raise ValueError(f"Unsupported distributed backend: {value!r}")
    return backend


def load_trl_finetune_config(config_path: str | Path | None = None) -> SpatialText2SQLFinetuneConfig:
    path = Path(config_path or DEFAULT_TRL_FINETUNE_CONFIG_PATH)
    if not path.is_file():
        raise FileNotFoundError(f"TRL fine-tune config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return _build_trl_finetune_config_from_payload(payload, path)


def _build_trl_finetune_config_from_payload(
    payload: Mapping[str, Any],
    path: Path,
) -> SpatialText2SQLFinetuneConfig:
    if not isinstance(payload, Mapping):
        raise ValueError(f"Invalid TRL fine-tune config in {path}: top level must be a mapping.")

    data_section = payload.get("data") or {}
    model_section = payload.get("model") or {}
    training_section = payload.get("training") or {}
    logging_section = payload.get("logging") or {}
    runtime_section = payload.get("runtime") or {}
    for section_name, section in (
        ("data", data_section),
        ("model", model_section),
        ("training", training_section),
        ("logging", logging_section),
        ("runtime", runtime_section),
    ):
        if section and not isinstance(section, Mapping):
            raise ValueError(f"Invalid TRL fine-tune config: '{section_name}' must be a mapping.")

    default_data = FinetuneDataConfig()
    default_model = FinetuneModelConfig()
    default_training = FinetuneTrainingConfig()
    default_logging = FinetuneLoggingConfig()
    default_runtime = FinetuneRuntimeConfig()

    return SpatialText2SQLFinetuneConfig(
        data=FinetuneDataConfig(
            input_path=_resolve_path(data_section.get("input_path"), path, default_data.input_path),
            alpaca_output_path=_resolve_path(
                data_section.get("alpaca_output_path"),
                path,
                default_data.alpaca_output_path,
            ),
            task_description=_as_text(data_section.get("task_description"), default_data.task_description),
            eval_ratio=_as_float(data_section.get("eval_ratio"), default_data.eval_ratio),
            question_id_start=_as_non_negative_int(data_section.get("question_id_start"), default_data.question_id_start),
            max_representative_rows=_as_positive_int(
                data_section.get("max_representative_rows"),
                default_data.max_representative_rows,
            ),
            shuffle_seed=int(data_section.get("shuffle_seed", default_data.shuffle_seed)),
        ),
        model=FinetuneModelConfig(
            model_name_or_path=_as_text(model_section.get("model_name_or_path"), default_model.model_name_or_path),
            tokenizer_name_or_path=_as_text(model_section.get("tokenizer_name_or_path"), default_model.tokenizer_name_or_path),
            trust_remote_code=_as_bool(model_section.get("trust_remote_code"), default_model.trust_remote_code),
            torch_dtype=_as_text(model_section.get("torch_dtype"), default_model.torch_dtype),
            attn_implementation=_as_text(model_section.get("attn_implementation"), default_model.attn_implementation),
        ),
        training=FinetuneTrainingConfig(
            output_dir=_resolve_path(training_section.get("output_dir"), path, default_training.output_dir),
            overwrite_output_dir=_as_bool(
                training_section.get("overwrite_output_dir"),
                default_training.overwrite_output_dir,
            ),
            per_device_train_batch_size=_as_positive_int(
                training_section.get("per_device_train_batch_size"),
                default_training.per_device_train_batch_size,
            ),
            per_device_eval_batch_size=_as_positive_int(
                training_section.get("per_device_eval_batch_size"),
                default_training.per_device_eval_batch_size,
            ),
            gradient_accumulation_steps=_as_positive_int(
                training_section.get("gradient_accumulation_steps"),
                default_training.gradient_accumulation_steps,
            ),
            learning_rate=_as_float(training_section.get("learning_rate"), default_training.learning_rate),
            num_train_epochs=_as_float(training_section.get("num_train_epochs"), default_training.num_train_epochs),
            max_steps=int(training_section.get("max_steps", default_training.max_steps)),
            weight_decay=_as_float(training_section.get("weight_decay"), default_training.weight_decay),
            warmup_ratio=_as_float(training_section.get("warmup_ratio"), default_training.warmup_ratio),
            lr_scheduler_type=_as_text(training_section.get("lr_scheduler_type"), default_training.lr_scheduler_type),
            logging_steps=_as_positive_int(training_section.get("logging_steps"), default_training.logging_steps),
            save_steps=_as_positive_int(training_section.get("save_steps"), default_training.save_steps),
            eval_steps=_as_positive_int(training_section.get("eval_steps"), default_training.eval_steps),
            save_total_limit=_as_positive_int(training_section.get("save_total_limit"), default_training.save_total_limit),
            max_length=_as_positive_int(training_section.get("max_length"), default_training.max_length),
            packing=_as_bool(training_section.get("packing"), default_training.packing),
            completion_only_loss=_as_bool(
                training_section.get("completion_only_loss"),
                default_training.completion_only_loss,
            ),
            gradient_checkpointing=_as_bool(
                training_section.get("gradient_checkpointing"),
                default_training.gradient_checkpointing,
            ),
            bf16=_as_bool(training_section.get("bf16"), default_training.bf16),
            fp16=_as_bool(training_section.get("fp16"), default_training.fp16),
            dataloader_num_workers=_as_non_negative_int(
                training_section.get("dataloader_num_workers"),
                default_training.dataloader_num_workers,
            ),
            report_to=_as_text(training_section.get("report_to"), default_training.report_to),
            seed=int(training_section.get("seed", default_training.seed)),
            resume_from_checkpoint=_as_text(
                training_section.get("resume_from_checkpoint"),
                default_training.resume_from_checkpoint,
            ),
            deepspeed_config_path=_resolve_path(
                training_section.get("deepspeed_config_path"),
                path,
                default_training.deepspeed_config_path,
            )
            if to_text(training_section.get("deepspeed_config_path"))
            else default_training.deepspeed_config_path,
        ),
        logging=FinetuneLoggingConfig(
            log_level=_as_text(logging_section.get("log_level"), default_logging.log_level),
            log_path=_resolve_path(logging_section.get("log_path"), path, default_logging.log_path)
            if to_text(logging_section.get("log_path"))
            else default_logging.log_path,
        ),
        runtime=FinetuneRuntimeConfig(
            nvidia_gpu_indices=_as_non_negative_int_list(
                runtime_section.get("nvidia_gpu_indices"),
                default_runtime.nvidia_gpu_indices,
            ),
            distributed_backend=_as_runtime_backend(
                runtime_section.get("distributed_backend"),
                default_runtime.distributed_backend,
            ),
            num_processes=_as_non_negative_int(
                runtime_section.get("num_processes"),
                default_runtime.num_processes,
            ),
            num_machines=_as_positive_int(
                runtime_section.get("num_machines"),
                default_runtime.num_machines,
            ),
            machine_rank=_as_non_negative_int(
                runtime_section.get("machine_rank"),
                default_runtime.machine_rank,
            ),
            main_process_port=_as_positive_int(
                runtime_section.get("main_process_port"),
                default_runtime.main_process_port,
            ),
        ),
    )


def override_trl_finetune_config(
    base: SpatialText2SQLFinetuneConfig,
    *,
    data: Mapping[str, Any] | None = None,
    model: Mapping[str, Any] | None = None,
    training: Mapping[str, Any] | None = None,
    logging: Mapping[str, Any] | None = None,
    runtime: Mapping[str, Any] | None = None,
) -> SpatialText2SQLFinetuneConfig:
    merged = {
        "data": {**base.data.__dict__, **dict(data or {})},
        "model": {**base.model.__dict__, **dict(model or {})},
        "training": {**base.training.__dict__, **dict(training or {})},
        "logging": {**base.logging.__dict__, **dict(logging or {})},
        "runtime": {**base.runtime.__dict__, **dict(runtime or {})},
    }
    return _build_trl_finetune_config_from_payload(
        stable_jsonify(merged),
        DEFAULT_TRL_FINETUNE_CONFIG_PATH,
    )
