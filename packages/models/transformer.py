"""
Libra Models - Educational Decoder-Only Transformer Language Model
Built from mathematical first principles in PyTorch.

Pipeline:
  Token IDs (B, T)
  -> Token Embedding + Positional Embedding (B, T, d_model)
  -> Transformer Blocks (LayerNorm -> Causal Multi-Head Attention -> Residual -> LayerNorm -> MLP -> Residual)
  -> Final LayerNorm (B, T, d_model)
  -> Output Head Projection (B, T, vocab_size) -> Logits
  -> Cross-Entropy Loss against Target IDs
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from packages.models.config import TinyTransformerConfig


class TokenEmbedding(nn.Module):
    """Maps discrete integer token IDs into continuous embedding vectors."""

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (batch_size, sequence_length)
        # Output: (batch_size, sequence_length, d_model)
        return self.embedding(x)


class PositionalEmbedding(nn.Module):
    """Learned positional representations for each sequence index (0 to max_context_length - 1)."""

    def __init__(self, max_context_length: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_context_length, d_model)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        # Input: (sequence_length,)
        # Output: (sequence_length, d_model)
        return self.embedding(positions)


class CausalSelfAttention(nn.Module):
    """Multi-Head Causal Self-Attention mechanism.

    Ensures that for any token at position t, it can only attend to tokens
    at positions <= t, preventing future information leakage.
    """

    def __init__(self, config: TinyTransformerConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim

        # Projections for Query, Key, and Value
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        # Output projection back to d_model
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Lower-triangular causal mask buffer (registered so it is moved to device with module)
        mask = torch.tril(torch.ones(config.max_context_length, config.max_context_length))
        self.register_buffer(
            "causal_mask", mask.view(1, 1, config.max_context_length, config.max_context_length)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # Batch, Sequence Length, Channels (d_model)

        # 1. Project into Queries, Keys, Values
        # Shape: (B, T, d_model) -> (B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim)
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # 2. Scaled Dot-Product Attention scores
        # Q * K^T / sqrt(d_k) -> shape: (B, n_heads, T, T)
        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # 3. Apply Causal Mask: future positions become -infinity
        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))

        # 4. Softmax turns raw scores into probability distribution summing to 1 across columns
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # 5. Weighted sum of Values: (B, n_heads, T, T) @ (B, n_heads, T, head_dim) -> (B, n_heads, T, head_dim)
        out = attn_weights @ v

        # 6. Concatenate all heads back together: (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # 7. Final linear projection
        return self.resid_dropout(self.out_proj(out))


class FeedForward(nn.Module):
    """Two-layer Multi-Layer Perceptron (MLP) with GELU activation."""

    def __init__(self, config: TinyTransformerConfig) -> None:
        super().__init__()
        hidden_dim = config.mlp_ratio * config.d_model
        self.fc1 = nn.Linear(config.d_model, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, d_model) -> (B, T, 4*d_model) -> (B, T, d_model)
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Standard Pre-LayerNorm Transformer Block."""

    def __init__(self, config: TinyTransformerConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN: normalize before attention, add residual connection
        x = x + self.attn(self.ln1(x))
        # Pre-LN: normalize before MLP, add residual connection
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformerLM(nn.Module):
    """Complete Educational Decoder-Only Autoregressive Language Model."""

    def __init__(self, config: TinyTransformerConfig) -> None:
        super().__init__()
        self.config = config

        self.tok_emb = TokenEmbedding(config.vocab_size, config.d_model)
        self.pos_emb = PositionalEmbedding(config.max_context_length, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights with standard normal (std = 0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, T = idx.shape
        if T > self.config.max_context_length:
            raise ValueError(
                f"Sequence length ({T}) exceeds maximum context length ({self.config.max_context_length})"
            )

        # Create position indices [0, 1, 2, ..., T-1]
        device = idx.device
        positions = torch.arange(0, T, dtype=torch.long, device=device)

        # Combine token embedding + positional embedding
        x = self.tok_emb(idx) + self.pos_emb(positions)
        x = self.drop(x)

        # Pass through sequential transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final LayerNorm
        x = self.ln_f(x)

        # Project to vocabulary logits: (B, T, vocab_size)
        logits = self.head(x)

        # Calculate cross-entropy loss if targets provided
        loss = None
        if targets is not None:
            # Flatten B and T to compute cross entropy over all tokens in the batch
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
