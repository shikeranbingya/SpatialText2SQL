"""TRL full-parameter fine-tuning runner."""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any, Sequence

import datasets
import numpy as np
import torch
import transformers
import trl

from .config import SpatialText2SQLFinetuneConfig
from .models import PreparedFinetuneSample

LOGGER = logging.getLogger(__name__)


class TRLFullFinetuner:
    def __init__(self, config: SpatialText2SQLFinetuneConfig) -> None:
        self.config = config

    def train(self, rows: Sequence[PreparedFinetuneSample]) -> dict[str, Any]:
        if not rows:
            raise ValueError("No prepared fine-tune rows were provided.")

        train_rows, eval_rows = self._split_rows(rows)
        train_dataset = datasets.Dataset.from_list(
            [{"prompt": row.prompt, "completion": row.completion} for row in train_rows]
        )
        eval_dataset = (
            datasets.Dataset.from_list(
                [{"prompt": row.prompt, "completion": row.completion} for row in eval_rows]
            )
            if eval_rows
            else None
        )

        tokenizer_name = self.config.model.tokenizer_name_or_path or self.config.model.model_name_or_path
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=self.config.model.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model_kwargs = {
            "trust_remote_code": self.config.model.trust_remote_code,
        }
        torch_dtype = self._resolve_torch_dtype()
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        if self.config.model.attn_implementation:
            model_kwargs["attn_implementation"] = self.config.model.attn_implementation
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.config.model.model_name_or_path,
            **model_kwargs,
        )
        if self.config.training.gradient_checkpointing:
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = False

        trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
        total_params = sum(param.numel() for param in model.parameters())
        LOGGER.info(
            "Loaded fine-tune model | name=%s | trainable_params=%s | total_params=%s",
            self.config.model.model_name_or_path,
            trainable_params,
            total_params,
        )

        sft_config = self._build_sft_config(trl.SFTConfig, has_eval=bool(eval_rows))
        output_dir = Path(self.config.training.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        trainer = self._build_trainer(
            trl.SFTTrainer,
            model=model,
            sft_config=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
        )
        train_result = trainer.train(
            resume_from_checkpoint=self.config.training.resume_from_checkpoint or None
        )
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        trainer.save_state()

        metrics = dict(train_result.metrics or {})
        metrics["train_rows"] = len(train_rows)
        metrics["eval_rows"] = len(eval_rows)
        return metrics

    @staticmethod
    def _build_trainer(
        trainer_cls,
        *,
        model,
        sft_config,
        train_dataset,
        eval_dataset,
        tokenizer,
    ):
        signature = inspect.signature(trainer_cls.__init__)
        kwargs: dict[str, Any] = {
            "model": model,
            "args": sft_config,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
        }
        if "processing_class" in signature.parameters:
            kwargs["processing_class"] = tokenizer
        elif "tokenizer" in signature.parameters:
            kwargs["tokenizer"] = tokenizer
        return trainer_cls(**kwargs)

    def _split_rows(
        self,
        rows: Sequence[PreparedFinetuneSample],
    ) -> tuple[list[PreparedFinetuneSample], list[PreparedFinetuneSample]]:
        if len(rows) < 2 or self.config.data.eval_ratio <= 0:
            return list(rows), []
        rng = np.random.default_rng(self.config.data.shuffle_seed)
        indices = list(rng.permutation(len(rows)))
        eval_count = max(1, int(round(len(rows) * self.config.data.eval_ratio)))
        eval_count = min(eval_count, len(rows) - 1)
        eval_indices = set(indices[:eval_count])
        train_rows: list[PreparedFinetuneSample] = []
        eval_rows: list[PreparedFinetuneSample] = []
        for index, row in enumerate(rows):
            if index in eval_indices:
                eval_rows.append(row)
            else:
                train_rows.append(row)
        return train_rows, eval_rows

    def _build_sft_config(self, sft_config_cls, *, has_eval: bool):
        signature = inspect.signature(sft_config_cls.__init__)
        kwargs: dict[str, Any] = {}

        def maybe_set(name: str, value: Any) -> None:
            if name in signature.parameters:
                kwargs[name] = value

        report_to = self.config.training.report_to
        if report_to.strip().lower() == "none":
            report_value: Any = []
        else:
            report_value = [item.strip() for item in report_to.split(",") if item.strip()]

        maybe_set("output_dir", self.config.training.output_dir)
        maybe_set("overwrite_output_dir", self.config.training.overwrite_output_dir)
        maybe_set("per_device_train_batch_size", self.config.training.per_device_train_batch_size)
        maybe_set("per_device_eval_batch_size", self.config.training.per_device_eval_batch_size)
        maybe_set("gradient_accumulation_steps", self.config.training.gradient_accumulation_steps)
        maybe_set("learning_rate", self.config.training.learning_rate)
        maybe_set("num_train_epochs", self.config.training.num_train_epochs)
        maybe_set("max_steps", self.config.training.max_steps)
        maybe_set("weight_decay", self.config.training.weight_decay)
        maybe_set("warmup_ratio", self.config.training.warmup_ratio)
        maybe_set("lr_scheduler_type", self.config.training.lr_scheduler_type)
        maybe_set("logging_steps", self.config.training.logging_steps)
        maybe_set("save_steps", self.config.training.save_steps)
        maybe_set("save_total_limit", self.config.training.save_total_limit)
        maybe_set("max_length", self.config.training.max_length)
        maybe_set("max_seq_length", self.config.training.max_length)
        maybe_set("packing", self.config.training.packing)
        maybe_set("completion_only_loss", self.config.training.completion_only_loss)
        maybe_set("gradient_checkpointing", self.config.training.gradient_checkpointing)
        maybe_set("bf16", self.config.training.bf16)
        maybe_set("fp16", self.config.training.fp16)
        maybe_set("dataloader_num_workers", self.config.training.dataloader_num_workers)
        maybe_set("report_to", report_value)
        maybe_set("seed", self.config.training.seed)
        maybe_set("remove_unused_columns", False)

        strategy_key = "eval_strategy" if "eval_strategy" in signature.parameters else "evaluation_strategy"
        if has_eval:
            maybe_set(strategy_key, "steps")
            maybe_set("eval_steps", self.config.training.eval_steps)
        else:
            maybe_set(strategy_key, "no")
        maybe_set("save_strategy", "steps")
        maybe_set("logging_strategy", "steps")
        return sft_config_cls(**kwargs)

    def _resolve_torch_dtype(self):
        dtype_name = self.config.model.torch_dtype.strip().lower()
        if not dtype_name or dtype_name == "auto":
            return None

        mapping = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if dtype_name not in mapping:
            raise ValueError(f"Unsupported torch_dtype: {self.config.model.torch_dtype}")
        return mapping[dtype_name]
