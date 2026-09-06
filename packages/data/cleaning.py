"""
Libra Data - Text Cleaning & Normalization
Sanitizes raw text: removes control characters, collapses whitespace, and standardizes Unicode.
"""

import re
import unicodedata
from typing import Any

from packages.data.config import DataPipelineConfig


def remove_control_characters(s: str) -> str:
    """Removes non-printable control characters while preserving standard newlines and tabs."""
    return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C" or ch in "\n\t\r")


def clean_text(text: str, config: DataPipelineConfig) -> tuple[str, dict[str, Any]]:
    """Cleans and sanitizes raw text according to configuration rules while preserving paragraph breaks."""
    chars_before = len(text)
    lines_before = len(text.splitlines())

    # 1. Unicode NFC Normalization
    if config.normalize_unicode:
        text = unicodedata.normalize("NFC", text)

    # 2. Control character removal
    text = remove_control_characters(text)

    # 3. Standardize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 4. Process paragraph by paragraph to preserve structural breaks
    raw_paragraphs = re.split(r"\n\s*\n", text)
    cleaned_paragraphs = []

    for para in raw_paragraphs:
        para_lines = []
        for line in para.splitlines():
            if config.clean_whitespace:
                line = re.sub(r"[ \t]+", " ", line).strip()
            if len(line) >= config.min_line_length:
                para_lines.append(line)
        if para_lines:
            cleaned_paragraphs.append("\n".join(para_lines))

    # 5. Join paragraphs with double newlines
    cleaned_text = "\n\n".join(cleaned_paragraphs)
    chars_after = len(cleaned_text)
    lines_after = len(cleaned_text.splitlines())
    reduction = round((1.0 - (chars_after / max(1, chars_before))) * 100, 2)

    stats = {
        "chars_before": chars_before,
        "chars_after": chars_after,
        "lines_before": lines_before,
        "lines_after": lines_after,
        "lines_removed": max(0, lines_before - lines_after),
        "reduction_pct": reduction,
    }
    return cleaned_text, stats
