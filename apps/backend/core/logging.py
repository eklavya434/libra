"""
Libra Backend - Structured Logging Setup
"""

import logging
import re
import sys

from apps.backend.core.config import settings

# Redact secrets that could otherwise end up in logs via %s formatting of
# provider objects, headers, or error payloads.
_REDACT_PATTERNS = (
    # key/value forms: "api_key": "x", api_key=x, Authorization: Bearer x,
    # "Bearer x" — the key word is followed by any short non-alphanumeric
    # separator (including JSON quotes) before the value.
    re.compile(
        r"\b(?:api[_-]?key|token|password|secret|authorization|bearer)\b"
        r"[^A-Za-z0-9]{0,16}?[\"']?[A-Za-z0-9._+/=\-]{8,}[\"']?",
        re.IGNORECASE,
    ),
    # Bare high-entropy credentials: long base64 / hex runs without a label.
    re.compile(r"[A-Za-z0-9+/]{40,}={0,2}"),
)


class RedactionFormatter(logging.Formatter):
    """Formatter that scrubs secrets from every rendered log line."""

    _MASK = "********"

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        for pattern in _REDACT_PATTERNS:
            message = pattern.sub(self._MASK, message)
        return message


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("libra")
    level = getattr(logging, settings.libra_log_level.upper(), logging.INFO)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = RedactionFormatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = setup_logging()
