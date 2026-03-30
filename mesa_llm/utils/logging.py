"""Logging configuration for Mesa LLM Assistant."""

import logging
import sys

from .config import config


def setup_logging(
    level: str | None = None,
    log_file: str | None = None,
    format_string: str | None = None,
) -> logging.Logger:
    """Set up logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file to write logs to
        format_string: Custom format string for log messages

    Returns:
        Configured logger instance
    """
    level = level or config.log_level
    log_file = log_file or config.log_file

    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(message)s"
        )

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()), format=format_string, handlers=[]
    )

    logger = logging.getLogger("mesa_llm")
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_formatter = logging.Formatter(format_string)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    logger.setLevel(getattr(logging, level.upper()))
    return logger


# Global logger instance
logger = setup_logging()
