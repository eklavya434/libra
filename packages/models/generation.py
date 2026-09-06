"""
Libra Models - Text Generation & Sampling
Generates new tokens autoregressively from a trained TinyTransformerLM.
"""

from typing import List
import torch
import torch.nn.functional as F

from packages.models.transformer import TinyTransformerLM


def generate(
    model: TinyTransformerLM,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    """Autoregressively generates `max_new_tokens` given a conditioning sequence `idx`.

    Args:
        model: The trained TinyTransformerLM model.
        idx: Conditioning token indices of shape (B, T).
        max_new_tokens: Number of subsequent tokens to generate.
        temperature: Scaling factor for logits (1.0 = normal, <1.0 = more confident/greedy, >1.0 = more random).
        top_k: Optional int to truncate to top-k highest probability tokens.

    Returns:
        Tensor of shape (B, T + max_new_tokens) containing the full generated sequence.
    """
    model.eval()

    for _ in range(max_new_tokens):
        # Crop context to model max_context_length if needed
        idx_cond = idx if idx.size(1) <= model.config.max_context_length else idx[:, -model.config.max_context_length:]

        with torch.no_grad():
            logits, _ = model(idx_cond)

        # Focus only on the last time step: (B, vocab_size)
        logits = logits[:, -1, :]

        if temperature <= 0.0:
            # Greedy decoding: pick highest logit directly
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            # Apply temperature scaling
            logits = logits / temperature

            # Optionally crop probabilities to top-k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            # Sample next token from the probability distribution
            idx_next = torch.multinomial(probs, num_samples=1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def encode_string(text: str) -> List[int]:
    """Encodes a string into byte/ASCII token IDs (0-255)."""
    return list(text.encode("utf-8"))


def decode_tokens(tokens: List[int]) -> str:
    """Decodes byte/ASCII token IDs back into a human-readable string."""
    return bytes(tokens).decode("utf-8", errors="replace")