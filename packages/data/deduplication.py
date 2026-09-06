"""
Libra Data - Deduplication Engine
Removes redundant documents and paragraphs using SHA-256 cryptographic hashing.
"""

import hashlib
from typing import Any


def deduplicate_paragraphs(text: str, delimiter: str = "\n\n") -> tuple[str, dict[str, Any]]:
    """Deduplicates text blocks (paragraphs or documents) using SHA-256 hashes."""
    paragraphs = [p.strip() for p in text.split(delimiter) if p.strip()]
    total_count = len(paragraphs)

    seen_hashes = set()
    unique_paragraphs: list[str] = []
    duplicate_count = 0

    for p in paragraphs:
        # Compute normalized hash (lowercase without extra whitespace for robust comparison)
        normalized = " ".join(p.lower().split())
        h = hashlib.sha256(normalized.encode("utf-8")).hexdigest()

        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_paragraphs.append(p)
        else:
            duplicate_count += 1

    deduplicated_text = delimiter.join(unique_paragraphs)
    duplication_rate = round((duplicate_count / max(1, total_count)) * 100, 2)

    stats = {
        "total_paragraphs": total_count,
        "unique_paragraphs": len(unique_paragraphs),
        "duplicates_removed": duplicate_count,
        "duplication_rate_pct": duplication_rate,
    }
    return deduplicated_text, stats
