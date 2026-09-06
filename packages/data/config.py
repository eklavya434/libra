"""
Libra Data - Pipeline Configuration
"""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class DataPipelineConfig:
    source_type: str = "local"  # "local", "memory", "huggingface"
    file_path: str = ""
    train_ratio: float = 0.9  # Train vs Validation split
    min_line_length: int = 10  # Minimum characters to retain a line
    deduplicate: bool = True  # Enable SHA-256 deduplication
    clean_whitespace: bool = True  # Normalize tabs, spaces, redundant newlines
    normalize_unicode: bool = True  # NFC Unicode normalization
    output_dir: str = "data/processed"  # Destination for preprocessed binary shards
    shard_size_tokens: int = 50_000  # Tokens per binary shard file
    max_ingest_mb: float = 25.0  # Guardrail against accidental large files

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
