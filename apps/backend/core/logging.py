"""
Libra Backend - Structured Logging Setup
"""

import logging
import sys

from apps.backend.core.config import settings


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("libra")
    level = getattr(logging, settings.libra_log_level.upper(), logging.INFO)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = setup_logging()
