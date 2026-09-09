"""
Libra Evaluation - Medusa Speculative Decoding Benchmark & Telemetry
Measures speculative speedup, acceptance length E[alpha], and head accuracy (Cai et al., 2024).
"""

from __future__ import annotations

import time
from typing import Any

import torch

from packages.models.components.medusa import MedusaModel


class MedusaEvaluator:
    """Evaluates speculative decoding acceleration and multi-head prediction fidelity."""

    @classmethod
    def benchmark_speculative_speedup(
        cls,
        medusa_model: MedusaModel,
        test_prompts: list[torch.Tensor],
        max_new_tokens: int = 15,
    ) -> dict[str, Any]:
        """Compare standard single-token autoregressive decoding against Medusa multi-token speculative decoding."""
        medusa_model.eval()
        base_model = medusa_model.base_model

        total_ar_time = 0.0
        total_ar_tokens = 0

        total_medusa_time = 0.0
        total_medusa_tokens = 0
        total_forward_passes_medusa = 0
        total_accepted_per_step = []

        for prompt in test_prompts:
            # 1. Standard Autoregressive Baseline
            curr_ar = prompt.clone()
            start_ar = time.perf_counter()
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    logits, _ = base_model(curr_ar)
                    next_tok = int(torch.argmax(logits[0, -1, :]).item())
                    next_tensor = torch.tensor([[next_tok]], device=prompt.device, dtype=torch.long)
                    curr_ar = torch.cat([curr_ar, next_tensor], dim=1)
            total_ar_time += time.perf_counter() - start_ar
            total_ar_tokens += max_new_tokens

            # 2. Medusa Speculative Decoding
            start_medusa = time.perf_counter()
            res = medusa_model.medusa_generate(
                prompt, max_new_tokens=max_new_tokens, temperature=0.0
            )
            total_medusa_time += time.perf_counter() - start_medusa
            total_medusa_tokens += res["generated_tokens_count"]
            total_forward_passes_medusa += res["total_forward_passes"]
            total_accepted_per_step.append(res["avg_accepted_per_step"])

        ar_tok_per_sec = round(total_ar_tokens / max(1e-5, total_ar_time), 1)
        medusa_tok_per_sec = round(total_medusa_tokens / max(1e-5, total_medusa_time), 1)

        speedup_latency = round(total_ar_time / max(1e-5, total_medusa_time), 2)
        avg_alpha = round(sum(total_accepted_per_step) / max(1, len(total_accepted_per_step)), 2)
        forward_pass_reduction_pct = round(
            (1.0 - (total_forward_passes_medusa / max(1, total_ar_tokens))) * 100.0, 2
        )

        return {
            "test_prompts_count": len(test_prompts),
            "tokens_generated_per_prompt": max_new_tokens,
            "autoregressive_time_ms": round(total_ar_time * 1000.0, 2),
            "medusa_time_ms": round(total_medusa_time * 1000.0, 2),
            "autoregressive_tokens_per_sec": ar_tok_per_sec,
            "medusa_tokens_per_sec": medusa_tok_per_sec,
            "speedup_ratio": speedup_latency,
            "avg_accepted_tokens_per_step": avg_alpha,
            "forward_pass_reduction_pct": forward_pass_reduction_pct,
        }

    @classmethod
    def analyze_head_accuracies(
        cls,
        medusa_model: MedusaModel,
        sequences: list[torch.Tensor],
    ) -> dict[str, Any]:
        """Compute top-1 prediction accuracy for each speculative head on validation sequences."""
        medusa_model.eval()
        num_heads = medusa_model.num_heads

        head_matches = [0] * num_heads
        head_totals = [0] * num_heads

        with torch.no_grad():
            for seq in sequences:
                if seq.shape[1] < num_heads + 2:
                    continue

                _, medusa_logits, _ = medusa_model.forward_with_medusa(seq)

                for k, logits_k in enumerate(medusa_logits):
                    offset = k + 1
                    target_slice = seq[:, offset:]
                    pred_slice = torch.argmax(logits_k[:, :-offset, :], dim=-1)

                    matches = (pred_slice == target_slice).sum().item()
                    head_matches[k] += matches
                    head_totals[k] += target_slice.numel()

        accuracies = [
            round((head_matches[k] / max(1, head_totals[k])) * 100.0, 2) for k in range(num_heads)
        ]

        return {
            "num_heads": num_heads,
            "per_head_accuracy_pct": accuracies,
            "per_head_eval_tokens": head_totals,
        }
