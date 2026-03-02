"""
Structured logging utilities for MedQCNN.

Uses Rich for beautiful console output during training
and experiment tracking.
"""

from __future__ import annotations

import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from medqcnn.config.constants import LOG_DIR

# Global console instance
console = Console()


def setup_logger(
    name: str = "medqcnn",
    level: int = logging.INFO,
    log_file: str | None = None,
) -> logging.Logger:
    """Set up a structured logger with Rich formatting.

    Args:
        name: Logger name.
        level: Logging level.
        log_file: Optional file path for log output.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Rich console handler
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        markup=True,
    )
    rich_handler.setLevel(level)
    logger.addHandler(rich_handler)

    # Optional file handler
    if log_file is not None:
        log_path = Path(LOG_DIR) / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path))
        file_handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
