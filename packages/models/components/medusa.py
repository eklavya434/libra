"""
Libra Models - Medusa Multi-Head Speculative Decoding Engine
Implements parallel speculative decoding heads on final transformer hidden states,
tree candidate generation, and single-pass parallel prefix verification (Cai et al., 2024).
"""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from packages.models.modern_transformer import ModernTransformerLM


class MedusaHead(nn.Module):
    """A single speculative decoding head with a residual MLP and vocabulary projection."""

    def __init__(self, d_model: int, vocab_size: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        inner_dim = hidden_dim or d_model
        # Residual projection block
        self.mlp_in = nn.Linear(d_model, inner_dim, bias=False)
        self.act = nn.SiLU()
        self.mlp_out = nn.Linear(inner_dim, d_model, bias=False)
        # Final prediction head
        self.proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Compute speculative logits for a future token position."""
        residual = h
        h = h + self.mlp_out(self.act(self.mlp_in(h)))
        return self.proj(h)


class MedusaModel(nn.Module):
    """Wraps a base autoregressive transformer with M parallel speculative decoding heads."""

    def __init__(
        self,
        base_model: ModernTransformerLM,
        num_heads: int = 3,
        head_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.num_heads = num_heads
        d_model = base_model.config.d_model
        vocab_size = base_model.config.vocab_size

        # M speculative heads (Head 0 is base model's LM head for t+1; Medusa heads predict t+2, t+3, ...)
        self.medusa_heads = nn.ModuleList(
            [
                MedusaHead(d_model=d_model, vocab_size=vocab_size, hidden_dim=head_hidden_dim)
                for _ in range(num_heads)
            ]
        )

    def forward_hidden(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward through base model to extract normalized final hidden state and base logits."""
        _B, T = idx.shape
        if T > self.base_model.config.max_context_length:
            raise ValueError(
                f"Sequence length ({T}) exceeds maximum context length ({self.base_model.config.max_context_length})"
            )

        x = self.base_model.drop(self.base_model.tok_emb(idx))
        for layer_idx, block in enumerate(self.base_model.blocks):
            x = block(x, kv_cache=None, layer_idx=layer_idx, start_pos=0)

        normed_x = self.base_model.norm_f(x)
        base_logits = self.base_model.output_head(normed_x)
        return normed_x, base_logits

    def forward_with_medusa(
        self, idx: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """Compute base model logits and all Medusa speculative logits simultaneously.

        Returns:
            base_logits: (B, T, vocab_size) predicting t+1
            medusa_logits: List of M tensors, each (B, T, vocab_size) predicting t+2, t+3, ...
            normed_hidden: (B, T, d_model) final normalized hidden states
        """
        normed_x, base_logits = self.forward_hidden(idx)
        medusa_logits = [head(normed_x) for head in self.medusa_heads]
        return base_logits, medusa_logits, normed_x

    def compute_medusa_loss(
        self,
        medusa_logits: list[torch.Tensor],
        targets: torch.Tensor,
        decay: float = 0.8,
    ) -> tuple[torch.Tensor, list[float]]:
        """Compute discounted cross-entropy loss for training speculative heads.

        Medusa head k (0-indexed) predicts token at position t + k + 2.
        """
        total_loss = torch.tensor(0.0, device=targets.device)
        per_head_losses: list[float] = []

        for k, head_logits in enumerate(medusa_logits):
            # Target offset: head k predicts t + k + 2
            # So for target tokens, offset is k + 1
            offset = k + 1
            if targets.shape[1] > offset:
                target_slice = targets[:, offset:]  # (B, T - offset)
                logits_slice = head_logits[:, :-offset, :]  # (B, T - offset, V)

                loss_k = F.cross_entropy(
                    logits_slice.reshape(-1, logits_slice.size(-1)),
                    target_slice.reshape(-1),
                    ignore_index=-100,
                )
                discount = decay**k
                total_loss = total_loss + discount * loss_k
                per_head_losses.append(round(float(loss_k.item()), 4))
            else:
                per_head_losses.append(0.0)

        return total_loss, per_head_losses

    def medusa_generate(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
    ) -> dict[str, Any]:
        """Speculative decoding loop generating multiple tokens per forward pass.

        At each step:
        1. Forward pass emits base prediction (t+1) and M Medusa predictions (t+2, ..., t+M+1).
        2. Draft candidate sequence is formed: [s_0, s_1, ..., s_M].
        3. Base model verifies candidate tokens in a single parallel forward pass.
        4. Greedily accepts matching prefix.
        """
        self.eval()
        device = prompt_tokens.device
        current_tokens = prompt_tokens.clone()

        generated_tokens: list[int] = []
        step_telemetry: list[dict[str, Any]] = []
        total_forward_passes = 0

        start_time = time.perf_counter()

        while len(generated_tokens) < max_new_tokens:
            if current_tokens.shape[1] >= self.base_model.config.max_context_length - (
                self.num_heads + 2
            ):
                break

            total_forward_passes += 1

            # 1. Propose drafts from current position
            with torch.no_grad():
                base_logits, medusa_logits, _ = self.forward_with_medusa(current_tokens)

                # Base model prediction for t+1
                last_base_logits = base_logits[0, -1, :]
                if temperature > 0:
                    base_next = int(
                        torch.argmax(F.softmax(last_base_logits / temperature, dim=-1)).item()
                    )
                else:
                    base_next = int(torch.argmax(last_base_logits).item())

                # Medusa proposals for future steps
                draft_candidates = [base_next]
                proposals_by_head = []
                for k, head_logits in enumerate(medusa_logits):
                    last_h_logits = head_logits[0, -1, :]
                    cand = int(torch.argmax(last_h_logits).item())
                    draft_candidates.append(cand)
                    proposals_by_head.append(
                        {
                            "head_idx": k + 1,
                            "candidate_token_id": cand,
                        }
                    )

            # 2. Speculative Verification Pass
            # Form candidate evaluation sequence: current_tokens + draft_candidates[:-1]
            draft_tensor = torch.tensor([draft_candidates], device=device, dtype=torch.long)
            verify_seq = torch.cat([current_tokens, draft_tensor[:, :-1]], dim=1)

            total_forward_passes += 1
            with torch.no_grad():
                verify_logits, _ = self.base_model(verify_seq)

            # 3. Verify candidates greedily
            # Candidate 0 (base_next) was generated by base model at current_tokens[-1], so it's always accepted!
            accepted_tokens = [base_next]
            acceptance_mask = [True]

            # Check remaining drafted candidates against verify_logits
            for i in range(1, len(draft_candidates)):
                # Position in verify_seq corresponding to candidate i-1
                pos_in_verify = current_tokens.shape[1] - 1 + (i - 1)
                expected_next = int(torch.argmax(verify_logits[0, pos_in_verify, :]).item())
                drafted_next = draft_candidates[i]

                if (
                    drafted_next == expected_next
                    and len(generated_tokens) + len(accepted_tokens) < max_new_tokens
                ):
                    accepted_tokens.append(drafted_next)
                    acceptance_mask.append(True)
                else:
                    acceptance_mask.append(False)
                    break

            # Update current tokens with all accepted tokens
            acc_tensor = torch.tensor([accepted_tokens], device=device, dtype=torch.long)
            current_tokens = torch.cat([current_tokens, acc_tensor], dim=1)
            generated_tokens.extend(accepted_tokens)

            step_telemetry.append(
                {
                    "step": len(step_telemetry) + 1,
                    "drafted_tokens": draft_candidates,
                    "accepted_count": len(accepted_tokens),
                    "accepted_tokens": accepted_tokens,
                    "acceptance_mask": acceptance_mask,
                    "proposals": proposals_by_head,
                }
            )

        elapsed = max(1e-5, time.perf_counter() - start_time)
        avg_acceptance = sum(s["accepted_count"] for s in step_telemetry) / max(
            1, len(step_telemetry)
        )
        speedup_ratio = round(len(generated_tokens) / max(1, total_forward_passes), 2)
        tokens_per_sec = round(len(generated_tokens) / elapsed, 1)

        return {
            "prompt_length": prompt_tokens.shape[1],
            "generated_tokens_count": len(generated_tokens),
            "generated_tokens": generated_tokens,
            "total_forward_passes": total_forward_passes,
            "speedup_ratio": speedup_ratio,
            "avg_accepted_per_step": round(avg_acceptance, 2),
            "tokens_per_second": tokens_per_sec,
            "elapsed_seconds": round(elapsed, 4),
            "steps": step_telemetry,
        }
