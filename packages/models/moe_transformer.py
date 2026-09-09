"""
Libra Models - Sparse Mixture of Experts (MoE) Transformer
Autoregressive language model with sparse expert routing, load-balancing loss,
and constant compute FLOPs decoupled from total parameter capacity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from packages.models.components.kv_cache import KVCache
from packages.models.components.moe import SparseMoEBlock
from packages.models.components.rmsnorm import RMSNorm
from packages.models.components.swiglu import SwiGLU
from packages.models.modern_config import ModernTransformerConfig
from packages.models.modern_transformer import ModernCausalAttention


@dataclass
class MoETransformerConfig(ModernTransformerConfig):
    """Configuration for Sparse Mixture of Experts (MoE) Language Model."""

    num_experts: int = 4  # Total number of expert networks per MoE layer
    num_experts_per_tok: int = 2  # Top-K active experts selected per token
    aux_loss_coef: float = 0.01  # Switch/Shazeer load balancing auxiliary loss coefficient
    noisy_gating: bool = True  # Add exploration noise to router logits during training
    moe_layers: list[int] | None = None  # Indices of layers that use MoE; None means all layers


class MoETransformerBlock(nn.Module):
    """Pre-RMSNorm Transformer Block with RoPE Attention and Sparse MoE Feed-Forward."""

    def __init__(self, config: MoETransformerConfig, is_moe_layer: bool = True) -> None:
        super().__init__()
        self.is_moe_layer = is_moe_layer

        self.norm1 = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = ModernCausalAttention(config)
        self.norm2 = RMSNorm(config.d_model, eps=config.norm_eps)

        if is_moe_layer:
            self.ffn = SparseMoEBlock(
                d_model=config.d_model,
                num_experts=config.num_experts,
                top_k=config.num_experts_per_tok,
                hidden_dim=config.hidden_dim,
                dropout=config.dropout,
                aux_loss_coef=config.aux_loss_coef,
                noisy_gating=config.noisy_gating,
            )
        else:
            self.ffn = SwiGLU(
                d_model=config.d_model,
                hidden_dim=config.hidden_dim,
                dropout=config.dropout,
            )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: KVCache | None = None,
        layer_idx: int = 0,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any] | None]:
        # 1. Attention with residual
        x = x + self.attn(
            self.norm1(x), kv_cache=kv_cache, layer_idx=layer_idx, start_pos=start_pos
        )

        # 2. Feed-forward (MoE or Dense)
        normed = self.norm2(x)
        if self.is_moe_layer:
            ffn_out, aux_loss, routing_info = self.ffn(normed)
        else:
            ffn_out = self.ffn(normed)
            aux_loss = torch.tensor(0.0, device=x.device)
            routing_info = None

        x = x + ffn_out
        return x, aux_loss, routing_info


class MoETransformerLM(nn.Module):
    """Modern Decoder-Only Transformer with Sparse Mixture of Experts routing."""

    def __init__(self, config: MoETransformerConfig) -> None:
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        moe_layer_set = (
            set(config.moe_layers) if config.moe_layers is not None else set(range(config.n_layers))
        )

        self.blocks = nn.ModuleList(
            [
                MoETransformerBlock(config, is_moe_layer=(i in moe_layer_set))
                for i in range(config.n_layers)
            ]
        )
        self.norm_f = RMSNorm(config.d_model, eps=config.norm_eps)
        self.output_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_weights:
            self.output_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_parameters(self) -> dict[str, Any]:
        """Calculate total model parameters vs active parameters executed per token."""
        total_params = sum(p.numel() for p in self.parameters())

        # Base parameters (embedding, attention, norms, output head)
        emb_params = self.tok_emb.weight.numel()
        norm_params = self.norm_f.weight.numel()
        head_params = 0 if self.config.tie_weights else self.output_head.weight.numel()

        block_attn_params = 0
        expert_params_per_block = 0
        router_params_per_block = 0

        for block in self.blocks:
            attn_p = sum(p.numel() for p in block.attn.parameters())
            norm1_p = block.norm1.weight.numel()
            norm2_p = block.norm2.weight.numel()
            block_attn_params += attn_p + norm1_p + norm2_p

            if block.is_moe_layer:
                router_p = sum(p.numel() for p in block.ffn.router.parameters())
                # Single expert param count
                single_exp_p = sum(p.numel() for p in block.ffn.experts[0].parameters())
                router_params_per_block += router_p
                # Active per token is k * single_exp_p
                expert_params_per_block += self.config.num_experts_per_tok * single_exp_p
            else:
                dense_ffn_p = sum(p.numel() for p in block.ffn.parameters())
                expert_params_per_block += dense_ffn_p

        active_params = (
            emb_params
            + norm_params
            + head_params
            + block_attn_params
            + router_params_per_block
            + expert_params_per_block
        )

        sparsity_ratio = (
            round(total_params / max(1, active_params), 2) if active_params > 0 else 1.0
        )
        savings_pct = (
            round((1.0 - active_params / total_params) * 100.0, 2) if total_params > 0 else 0.0
        )

        return {
            "total_parameters": total_params,
            "active_parameters_per_token": active_params,
            "sparsity_ratio": sparsity_ratio,
            "compute_savings_pct": savings_pct,
            "num_experts": self.config.num_experts,
            "num_experts_per_tok": self.config.num_experts_per_tok,
            "n_layers": self.config.n_layers,
            "d_model": self.config.d_model,
        }

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
        start_pos: int = 0,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
        list[dict[str, Any]],
    ]:
        """Forward pass through MoE transformer.

        Returns:
            logits: (B, T, vocab_size)
            total_loss: task_loss + total_aux_loss (or None)
            task_loss: Cross-entropy loss on target tokens (or None)
            aux_loss: Combined load-balancing auxiliary loss across all MoE layers
            layer_routing_info: Routing diagnostics per MoE block
        """
        _B, T = idx.shape
        total_len = start_pos + T
        if total_len > self.config.max_context_length:
            raise ValueError(
                f"Sequence length ({total_len}) exceeds maximum context length ({self.config.max_context_length})"
            )

        x = self.drop(self.tok_emb(idx))
        total_aux_loss = torch.tensor(0.0, device=idx.device)
        layer_routing_info: list[dict[str, Any]] = []

        for layer_idx, block in enumerate(self.blocks):
            x, aux_loss, routing = block(
                x,
                kv_cache=kv_cache,
                layer_idx=layer_idx,
                start_pos=start_pos,
            )
            total_aux_loss = total_aux_loss + aux_loss
            if routing is not None:
                layer_routing_info.append(
                    {
                        "layer_idx": layer_idx,
                        "routing": routing,
                    }
                )

        x = self.norm_f(x)
        logits = self.output_head(x)

        task_loss = None
        total_loss = None
        if targets is not None:
            task_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )
            total_loss = task_loss + total_aux_loss

        return logits, total_loss, task_loss, total_aux_loss, layer_routing_info
