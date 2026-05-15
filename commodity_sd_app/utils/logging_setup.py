"""Lightweight logging setup used across the app."""

from __future__ import annotations

import logging
import os


def get_logger(name: str = "commodity_sd") -> logging.Logger:
    """Return a configured logger.  Idempotent across reloads."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level = os.environ.get("COMMODITY_SD_LOG", "INFO").upper()
    logger.setLevel(level)

    handler = logging.StreamHandler()
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
