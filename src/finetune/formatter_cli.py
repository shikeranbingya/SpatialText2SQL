"""CLI for formatting nl2sql rows into Alpaca-style fine-tune JSONL."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import DEFAULT_TRL_FINETUNE_CONFIG_PATH, load_trl_finetune_config, override_trl_finetune_config
from .formatter import NL2SQLAlpacaFormatter
from .io import load_raw_finetune_samples, write_raw_finetune_samples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Format nl2sql JSONL into Alpaca-style fine-tune JSONL.")
    parser.add_argument("--config", default=str(DEFAULT_TRL_FINETUNE_CONFIG_PATH))
    parser.add_argument("--input")
    parser.add_argument("--alpaca-output")
    parser.add_argument("--log-level")
    parser.add_argument("--log-path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    config = load_trl_finetune_config(args.config)
    config = override_trl_finetune_config(
        config,
        data={key: value for key, value in {
            "input_path": args.input,
            "alpaca_output_path": args.alpaca_output,
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
        "Formatting nl2sql into Alpaca JSONL | input=%s | output=%s",
        config.data.input_path,
        config.data.alpaca_output_path,
    )

    raw_rows = load_raw_finetune_samples(config.data.input_path)
    formatter = NL2SQLAlpacaFormatter(data_config=config.data)
    alpaca_rows = formatter.format_samples(raw_rows)
    write_raw_finetune_samples(config.data.alpaca_output_path, alpaca_rows)

    logging.info(
        "Alpaca formatting completed | input_rows=%s | output_rows=%s | output=%s",
        len(raw_rows),
        len(alpaca_rows),
        config.data.alpaca_output_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
