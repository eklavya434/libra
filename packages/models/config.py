"""
Libra Models - Tiny Transformer Configuration
Holds the hyperparameters for our educational decoder-only language model.
"""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class TinyTransformerConfig:
    vocab_size: int = 256          # Character / byte-level vocabulary (0-255)
    max_context_length: int = 128  # Maximum sequence length (tokens)
    d_model: int = 128             # Hidden embedding dimension
    n_heads: int = 4               # Number of attention heads (each head_dim = d_model // n_heads = 32)
    n_layers: int = 2              # Number of stacked Transformer blocks
    mlp_ratio: int = 4             # Multiplier for FeedForward inner dimension (4 * 128 = 512)
    dropout: float = 0.1           # Regularization dropout probability
    device: str = "cpu"            # Target device ("cpu" for our target hardware)

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)