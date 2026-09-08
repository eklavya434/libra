"""
Libra Models - Grammar-Constrained Text Generation
Autoregressively decodes tokens while applying GrammarLogitsProcessor masking at each step.
"""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn.functional as F

from packages.core.grammar.grammar_processor import GrammarLogitsProcessor


def generate_with_grammar(
    model: torch.nn.Module,
    idx: torch.Tensor,
    processor: GrammarLogitsProcessor,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_token_id: int | None = None,
) -> tuple[torch.Tensor, str, dict[str, Any]]:
    """Autoregressively generates tokens constrained by a grammar logits processor.

    Args:
        model: Autoregressive language model.
        idx: Conditioning token indices (1, T).
        processor: GrammarLogitsProcessor to mask invalid tokens at each step.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature.
        top_k: Optional top-k sampling limit.
        eos_token_id: Optional EOS token id to stop generation early when grammar completes.

    Returns:
        tuple of (full_idx_tensor, generated_text, telemetry_dict)
    """
    model.eval()
    device = next(model.parameters()).device
    idx = idx.to(device)

    start_len = idx.size(1)
    start_time = time.perf_counter()
    max_ctx = getattr(model.config, "max_context_length", 512)

    generated_tokens: list[int] = []

    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= max_ctx else idx[:, -max_ctx:]

        with torch.no_grad():
            logits, _ = model(idx_cond)

        # Focus on the last token position
        step_logits = logits[0, -1, :]

        # Apply grammar mask (offset by start_len)
        step_logits = processor(idx[0], step_logits, prefix_offset=start_len)

        if temperature <= 0.0:
            next_token = int(torch.argmax(step_logits, dim=-1).item())
        else:
            scaled = step_logits / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
                scaled = scaled.clone()
                scaled[scaled < v[-1]] = float("-inf")
            probs = F.softmax(scaled, dim=-1)
            # Safeguard if all probabilities are zero
            if torch.isnan(probs).any() or probs.sum() <= 0:
                next_token = int(torch.argmax(step_logits, dim=-1).item())
            else:
                next_token = int(torch.multinomial(probs, num_samples=1).item())

        # Check for EOS termination
        if eos_token_id is not None and next_token == eos_token_id:
            break

        generated_tokens.append(next_token)
        next_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
        idx = torch.cat((idx, next_tensor), dim=1)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    generated_text = processor.decode_fn(generated_tokens) if generated_tokens else ""

    telemetry = {
        "generated_tokens": generated_tokens,
        "token_count": len(generated_tokens),
        "duration_ms": round(elapsed_ms, 2),
        "tokens_per_sec": round(len(generated_tokens) / max(1e-6, elapsed_ms / 1000.0), 2),
    }

    return idx, generated_text, telemetry
