"""
Libra Evaluation - Mixture of Experts (MoE) Routing & Utilization Evaluator
Analyzes token-to-expert routing assignments, load balancing distribution,
and detects expert starvation across MoE layers.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from packages.models.moe_transformer import MoETransformerLM


class MoEEvaluator:
    """Diagnostic evaluator for Sparse Mixture of Experts architectures."""

    @staticmethod
    def compute_distribution_entropy(probs: list[float]) -> float:
        """Compute Shannon entropy in nats to measure load uniformity across experts."""
        eps = 1e-12
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * math.log(p + eps)
        return round(entropy, 4)

    @classmethod
    def analyze_token_routing(
        cls,
        model: MoETransformerLM,
        input_ids: torch.Tensor,
        vocab_tokens: list[str] | None = None,
    ) -> dict[str, Any]:
        """Trace token-level expert selection and gating weights across all MoE blocks."""
        model.eval()
        with torch.no_grad():
            logits, _, _, aux_loss, routing_info = model(input_ids)

        _B, T = input_ids.shape
        num_experts = model.config.num_experts
        top_k = model.config.num_experts_per_tok

        # Build per-token tracing
        tokens_trace = []
        for t_idx in range(T):
            token_id = int(input_ids[0, t_idx].item())
            token_str = (
                vocab_tokens[token_id]
                if vocab_tokens and token_id < len(vocab_tokens)
                else f"id_{token_id}"
            )

            layer_data = []
            for layer_dict in routing_info:
                l_idx = layer_dict["layer_idx"]
                indices = layer_dict["routing"]["topk_indices"][0, t_idx]  # (top_k,)
                weights = layer_dict["routing"]["topk_weights"][0, t_idx]  # (top_k,)

                layer_data.append(
                    {
                        "layer_idx": l_idx,
                        "selected_experts": [
                            {
                                "expert_id": int(indices[k].item()),
                                "weight": round(float(weights[k].item()), 4),
                            }
                            for k in range(top_k)
                        ],
                    }
                )

            tokens_trace.append(
                {
                    "position": t_idx,
                    "token_id": token_id,
                    "token_str": token_str,
                    "layers": layer_data,
                }
            )

        # Aggregate expert load statistics across all layers and tokens
        layer_stats = []
        for layer_dict in routing_info:
            l_idx = layer_dict["layer_idx"]
            indices = layer_dict["routing"]["topk_indices"]  # (B, T, top_k)
            total_dispatches = indices.numel()

            counts = [0] * num_experts
            for exp_id in range(num_experts):
                counts[exp_id] = int((indices == exp_id).sum().item())

            fractions = [round(c / max(1, total_dispatches), 4) for c in counts]
            entropy = cls.compute_distribution_entropy(fractions)
            max_entropy = round(math.log(num_experts), 4)

            layer_stats.append(
                {
                    "layer_idx": l_idx,
                    "expert_counts": counts,
                    "expert_fractions": fractions,
                    "entropy": entropy,
                    "max_possible_entropy": max_entropy,
                    "uniformity_score_pct": round((entropy / max(1e-5, max_entropy)) * 100.0, 2),
                }
            )

        return {
            "prompt_tokens_count": T,
            "num_experts": num_experts,
            "top_k": top_k,
            "aux_loss": round(float(aux_loss.item()), 5),
            "tokens": tokens_trace,
            "layer_stats": layer_stats,
        }

    @classmethod
    def compute_expert_utilization(
        cls,
        model: MoETransformerLM,
        sequences: list[torch.Tensor],
    ) -> dict[str, Any]:
        """Assess expert load balance and starvation across validation sequences."""
        model.eval()
        num_experts = model.config.num_experts

        total_expert_dispatches = [0] * num_experts
        total_tokens = sum(seq.numel() for seq in sequences)

        with torch.no_grad():
            for seq in sequences:
                _, _, _, _, routing_info = model(seq)
                for layer_dict in routing_info:
                    indices = layer_dict["routing"]["topk_indices"]
                    for exp_id in range(num_experts):
                        total_expert_dispatches[exp_id] += int((indices == exp_id).sum().item())

        sum_dispatches = sum(total_expert_dispatches)
        fractions = [round(count / max(1, sum_dispatches), 4) for count in total_expert_dispatches]

        mean_val = sum(fractions) / max(1, len(fractions))
        variance = sum((f - mean_val) ** 2 for f in fractions) / max(1, len(fractions))
        std_dev = math.sqrt(variance)
        cv = round(std_dev / max(1e-5, mean_val), 4)

        # Starved experts: receiving less than 5% of tokens
        starved_experts = [exp_id for exp_id, f in enumerate(fractions) if f < 0.05]

        param_stats = model.count_parameters()

        return {
            "total_tokens_evaluated": total_tokens,
            "total_expert_dispatches": sum_dispatches,
            "expert_counts": total_expert_dispatches,
            "expert_fractions": fractions,
            "coefficient_of_variation": cv,
            "is_balanced": cv < 0.35,
            "starved_experts": starved_experts,
            "parameter_efficiency": param_stats,
        }
