"""I/O helpers for TRL fine-tuning datasets."""

from __future__ import annotations

import json
from pathlib import Path

from .models import PreparedFinetuneSample, RawFinetuneSample


def load_raw_finetune_samples(input_path: str) -> list[RawFinetuneSample]:
    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"Fine-tune input JSONL file not found: {path}")
    rows: list[RawFinetuneSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
            try:
                rows.append(RawFinetuneSample.from_dict(payload))
            except ValueError as exc:
                raise ValueError(f"Invalid fine-tune sample on line {line_number} of {path}: {exc}") from exc
    return rows


def write_raw_finetune_samples(output_path: str, rows: list[RawFinetuneSample]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), ensure_ascii=False))
            handle.write("\n")


def load_prepared_finetune_samples(input_path: str) -> list[PreparedFinetuneSample]:
    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"Prepared fine-tune JSONL file not found: {path}")
    rows: list[PreparedFinetuneSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
            try:
                rows.append(PreparedFinetuneSample.from_dict(payload))
            except ValueError as exc:
                raise ValueError(f"Invalid prepared sample on line {line_number} of {path}: {exc}") from exc
    return rows


def write_prepared_finetune_samples(output_path: str, rows: list[PreparedFinetuneSample]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), ensure_ascii=False))
            handle.write("\n")
