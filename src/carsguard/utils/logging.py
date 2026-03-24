from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def get_logger(name: str = "carsguard", level: int = logging.INFO) -> logging.Logger:
    """
    Create or retrieve a console logger.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(level)

        handler = logging.StreamHandler()
        handler.setLevel(level)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False
    return logger


def add_file_handler(
    logger: logging.Logger,
    file_path: str | Path,
    level: Optional[int] = None,
) -> None:
    """
    Add a file handler to an existing logger.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(file_path, encoding="utf-8")
    handler.setLevel(level or logger.level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)