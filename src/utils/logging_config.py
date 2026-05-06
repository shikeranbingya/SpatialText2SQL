"""Unified logging helpers for the src package and spatial import utilities."""
from __future__ import annotations

import logging
from typing import Optional

from tqdm import tqdm

_TOP_NAME = __name__.split(".")[0]
_SPATIAL_LOGGER_NAME = "spatial_importer"
_PBF_LOGGER_NAME = "osm_pbf_importer"


class TqdmLoggingHandler(logging.Handler):
    """Logging handler that writes through tqdm when progress bars are active."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logging(
    level: int = logging.INFO,
    fmt: Optional[str] = None,
    force: bool = False,
) -> logging.Logger:
    """Configure the top-level src package logger and return it."""

    root_logger = logging.getLogger(_TOP_NAME)
    if root_logger.handlers and not force:
        root_logger.setLevel(level)
        return root_logger

    if force:
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)

    handler = logging.StreamHandler()
    fmt = fmt or (
        "%(asctime)s - %(name)s - %(levelname)s - "
        "[%(filename)s:%(lineno)s] - %(message)s"
    )
    handler.setFormatter(logging.Formatter(fmt))
    handler.setLevel(level)

    root_logger.addHandler(handler)
    root_logger.setLevel(level)
    root_logger.propagate = False
    return root_logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger under the src package namespace."""

    if not name:
        return logging.getLogger(_TOP_NAME)
    if isinstance(name, str) and name.startswith(f"{_TOP_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_TOP_NAME}.{name}")


def _configure_named_logger(
    name: str,
    use_tqdm: bool = False,
    level: int = logging.INFO,
) -> logging.Logger:
    log = logging.getLogger(name)
    log.handlers.clear()
    log.setLevel(level)

    handler: logging.Handler
    if use_tqdm:
        handler = TqdmLoggingHandler()
    else:
        handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    log.addHandler(handler)
    log.propagate = False
    return log


def init_spatial_logging(use_tqdm: bool = False) -> logging.Logger:
    """Configure the logger used by SHP/GeoJSON import flows."""

    return _configure_named_logger(_SPATIAL_LOGGER_NAME, use_tqdm=use_tqdm)


def init_pbf_logging(use_tqdm: bool = False) -> logging.Logger:
    """Configure the logger used by PBF import flows."""

    return _configure_named_logger(_PBF_LOGGER_NAME, use_tqdm=use_tqdm)


logger: logging.Logger = setup_logging()
spatial_logger: logging.Logger = init_spatial_logging(use_tqdm=False)
pbf_logger: logging.Logger = init_pbf_logging(use_tqdm=False)


__all__ = [
    "TqdmLoggingHandler",
    "get_logger",
    "init_pbf_logging",
    "init_spatial_logging",
    "logger",
    "pbf_logger",
    "setup_logging",
    "spatial_logger",
]
