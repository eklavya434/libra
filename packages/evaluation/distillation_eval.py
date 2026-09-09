"""
Libra Evaluation - Knowledge Distillation Evaluator & Dark Knowledge Analyzer
Analyzes temperature effects, probability softening, KL/JS divergence,
and CPU latency speedup between teacher and student transformers.
"""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn.functional as F

from packages.models.modern_transformer import ModernTransformerLM


class DistillationEvaluator:
    """Evaluates knowledge transfer fidelity, dark knowledge distributions, and CPU inference speedup."""

    @staticmethod
    def compute_shannon_entropy(probs: torch.Tensor) -> float:
        """Compute Shannon entropy H(P) = -sum(p * log(p + eps)) in nats."""
        eps = 1e-12
        entropy = -torch.sum(probs * torch.log(probs + eps), dim=-1)
        return float(entropy.mean().item())

    @staticmethod
    def compute_js_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
        """Compute Jensen-Shannon divergence JS(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M) where M = 0.5*(P+Q)."""
        eps = 1e-12
        p_safe = torch.clamp(p, min=eps)
        q_safe = torch.clamp(q, min=eps)
        m = 0.5 * (p_safe + q_safe)

        kl_pm = F.kl_div(torch.log(m), p_safe, reduction="batchmean")
        kl_qm = F.kl_div(torch.log(m), q_safe, reduction="batchmean")
        js = 0.5 * (kl_pm + kl_qm)
        return float(js.item())

    @classmethod
    def analyze_dark_knowledge(
        cls,
        teacher_model: ModernTransformerLM,
        student_model: ModernTransformerLM,
        input_ids: torch.Tensor,
        temperatures: list[float] | None = None,
        top_k: int = 5,
        vocab_tokens: list[str] | None = None,
    ) -> dict[str, Any]:
        """Examine token probability distributions at the last sequence position across varying temperatures."""
        teacher_model.eval()
        student_model.eval()

        if temperatures is None:
            temperatures = [1.0, 2.0, 5.0, 10.0]

        with torch.no_grad():
            t_logits, _ = teacher_model(input_ids)
            s_logits, _ = student_model(input_ids)

            # Focus on next-token prediction at the final position
            t_last = t_logits[0, -1, :]  # (V,)
            s_last = s_logits[0, -1, :]  # (V,)

            temp_results = []
            for tau in temperatures:
                t_probs = F.softmax(t_last / tau, dim=-1)
                s_probs = F.softmax(s_last / tau, dim=-1)

                # KL divergence D_KL(teacher || student)
                s_log_probs = F.log_softmax(s_last / tau, dim=-1)
                kl_div = float(F.kl_div(s_log_probs, t_probs, reduction="sum").item())
                js_div = cls.compute_js_divergence(t_probs, s_probs)

                # Entropy (measures distribution flatness / darkness)
                t_entropy = cls.compute_shannon_entropy(t_probs)
                s_entropy = cls.compute_shannon_entropy(s_probs)

                # Top-K distributions
                t_top_vals, t_top_idx = torch.topk(t_probs, min(top_k, t_probs.shape[-1]))
                s_top_vals, s_top_idx = torch.topk(s_probs, min(top_k, s_probs.shape[-1]))

                t_top_list = [
                    {
                        "token_id": int(idx.item()),
                        "token_str": (
                            vocab_tokens[idx.item()]
                            if vocab_tokens and idx.item() < len(vocab_tokens)
                            else f"id_{idx.item()}"
                        ),
                        "prob": round(float(val.item()), 4),
                    }
                    for val, idx in zip(t_top_vals, t_top_idx)
                ]

                s_top_list = [
                    {
                        "token_id": int(idx.item()),
                        "token_str": (
                            vocab_tokens[idx.item()]
                            if vocab_tokens and idx.item() < len(vocab_tokens)
                            else f"id_{idx.item()}"
                        ),
                        "prob": round(float(val.item()), 4),
                    }
                    for val, idx in zip(s_top_vals, s_top_idx)
                ]

                temp_results.append(
                    {
                        "temperature": tau,
                        "kl_divergence": round(kl_div, 4),
                        "js_divergence": round(js_div, 4),
                        "teacher_entropy": round(t_entropy, 4),
                        "student_entropy": round(s_entropy, 4),
                        "teacher_top_tokens": t_top_list,
                        "student_top_tokens": s_top_list,
                    }
                )

        return {
            "prompt_length": input_ids.shape[1],
            "vocab_size": teacher_model.config.vocab_size,
            "temperature_analysis": temp_results,
        }

    @classmethod
    def benchmark_speed_and_agreement(
        cls,
        teacher_model: ModernTransformerLM,
        student_model: ModernTransformerLM,
        test_sequences: list[torch.Tensor],
        runs: int = 5,
    ) -> dict[str, Any]:
        """Measure inference latency on CPU, tokens/sec throughput, and token prediction agreement."""
        teacher_model.eval()
        student_model.eval()

        total_tokens = sum(seq.numel() for seq in test_sequences) * runs

        # 1. Benchmark Teacher
        with torch.no_grad():
            # Warmup
            _ = teacher_model(test_sequences[0])

            start_t = time.perf_counter()
            for _ in range(runs):
                for seq in test_sequences:
                    _ = teacher_model(seq)
            elapsed_t = time.perf_counter() - start_t
            teacher_latency_ms = (elapsed_t / (runs * len(test_sequences))) * 1000.0
            teacher_tokens_per_sec = total_tokens / max(1e-6, elapsed_t)

        # 2. Benchmark Student
        with torch.no_grad():
            # Warmup
            _ = student_model(test_sequences[0])

            start_s = time.perf_counter()
            for _ in range(runs):
                for seq in test_sequences:
                    _ = student_model(seq)
            elapsed_s = time.perf_counter() - start_s
            student_latency_ms = (elapsed_s / (runs * len(test_sequences))) * 1000.0
            student_tokens_per_sec = total_tokens / max(1e-6, elapsed_s)

        speedup_ratio = teacher_latency_ms / max(1e-6, student_latency_ms)

        # 3. Compute Top-1 and Top-5 Agreement
        total_positions = 0
        top1_matches = 0
        top5_matches = 0

        with torch.no_grad():
            for seq in test_sequences:
                t_logits, _ = teacher_model(seq)
                s_logits, _ = student_model(seq)

                t_top1 = torch.argmax(t_logits, dim=-1)
                s_top1 = torch.argmax(s_logits, dim=-1)
                t_top5 = torch.topk(t_logits, min(5, t_logits.shape[-1]), dim=-1).indices

                matches = (s_top1 == t_top1).sum().item()
                top1_matches += matches

                # Check if student top1 is in teacher top5
                s_top1_expanded = s_top1.unsqueeze(-1)  # (B, T, 1)
                in_top5 = (s_top1_expanded == t_top5).any(dim=-1).sum().item()
                top5_matches += in_top5

                total_positions += seq.numel()

        top1_agreement_rate = top1_matches / max(1, total_positions)
        top5_agreement_rate = top5_matches / max(1, total_positions)

        return {
            "total_sequences": len(test_sequences),
            "total_tokens_evaluated": total_tokens // runs,
            "teacher_latency_ms": round(teacher_latency_ms, 3),
            "student_latency_ms": round(student_latency_ms, 3),
            "speedup_ratio": round(speedup_ratio, 2),
            "teacher_throughput_tok_per_sec": round(teacher_tokens_per_sec, 1),
            "student_throughput_tok_per_sec": round(student_tokens_per_sec, 1),
            "top1_agreement_pct": round(top1_agreement_rate * 100.0, 2),
            "top5_agreement_pct": round(top5_agreement_rate * 100.0, 2),
        }
