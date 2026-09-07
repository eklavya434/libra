"""
Libra Models - Text Generation & Sampling
Generates new tokens autoregressively from a trained TinyTransformerLM.
"""

from typing import Any

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
        idx_cond = (
            idx
            if idx.size(1) <= model.config.max_context_length
            else idx[:, -model.config.max_context_length :]
        )

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


def encode_string(text: str) -> list[int]:
    """Encodes a string into byte/ASCII token IDs (0-255)."""
    return list(text.encode("utf-8"))


def decode_tokens(tokens: list[int]) -> str:
    """Decodes byte/ASCII token IDs back into a human-readable string."""
    return bytes(tokens).decode("utf-8", errors="replace")


def generate_with_cache(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Generates new tokens autoregressively utilizing the Key-Value (KV) cache.

    Transforms per-token decode steps from O(T) sequence evaluations down to O(1).
    """
    import time

    from packages.models.components.kv_cache import KVCache

    model.eval()
    start_time = time.perf_counter()
    device = next(model.parameters()).device
    idx = idx.to(device)
    prompt_len = idx.size(1)

    n_layers = len(getattr(model, "blocks", []))
    kv_cache = KVCache(n_layers=n_layers)

    # 1. Prefill Phase: Process prompt tokens and populate cache
    with torch.no_grad():
        logits, _ = model(idx, kv_cache=kv_cache, start_pos=0)
    logits = logits[:, -1, :]

    if temperature <= 0.0:
        next_tok = torch.argmax(logits, dim=-1, keepdim=True)
    else:
        logits = logits / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)

    idx = torch.cat((idx, next_tok), dim=1)

    # 2. Decode Phase: Pass only the single latest token (O(1) complexity per step)
    for _ in range(1, max_new_tokens):
        start_pos = kv_cache.get_seq_len()
        with torch.no_grad():
            logits, _ = model(next_tok, kv_cache=kv_cache, start_pos=start_pos)
        logits = logits[:, -1, :]

        if temperature <= 0.0:
            next_tok = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, next_tok), dim=1)

    elapsed = time.perf_counter() - start_time
    gen_tokens = idx[0, prompt_len:].tolist()

    telemetry = {
        "generated_tokens": gen_tokens,
        "tokens_generated": len(gen_tokens),
        "prefill_tokens": prompt_len,
        "elapsed_time_sec": round(elapsed, 4),
        "tokens_per_sec": round(len(gen_tokens) / max(1e-6, elapsed), 2),
        "cache_memory_bytes": kv_cache.memory_bytes(),
    }

    return idx, telemetry
