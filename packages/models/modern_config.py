"""
Libra Models - Modern Transformer Configuration (YAML Support)
"""

import os
from dataclasses import asdict, dataclass
from typing import Any

import yaml


@dataclass
class ModernTransformerConfig:
    vocab_size: int = 512
    max_context_length: int = 256
    d_model: int = 128
    n_heads: int = 4
    n_kv_heads: int | None = None  # Number of Key/Value heads for GQA/MQA (defaults to n_heads)
    n_layers: int = 2
    hidden_dim: int | None = None  # SwiGLU inner dimension (defaults to 8/3 * d_model)
    rope_theta_base: float = 10000.0
    norm_eps: float = 1e-6
    dropout: float = 0.0
    tie_weights: bool = False  # Weight tying: output_head.weight = tok_emb.weight
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
            )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def num_queries_per_kv(self) -> int:
        """Ratio of query heads to each key/value head (GQA expansion factor)."""
        assert self.n_kv_heads is not None
        return self.n_heads // self.n_kv_heads

    @property
    def kv_dim(self) -> int:
        """Total dimension of key or value states."""
        assert self.n_kv_heads is not None
        return self.n_kv_heads * self.head_dim

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str) -> "ModernTransformerConfig":
        """Loads configuration from a YAML file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls(**data)

    def to_yaml(self, path: str) -> None:
        """Saves configuration to a YAML file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
