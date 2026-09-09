"""
Libra Models - Knowledge Distillation Model Shrinker
Supports layer dropping, sub-network extraction, and compression statistics.
"""

from __future__ import annotations

from typing import Any

import torch

from packages.models.modern_config import ModernTransformerConfig
from packages.models.modern_transformer import ModernTransformerLM


class ModelShrinker:
    """Utilities for shrinking teacher transformer architectures into compact students."""

    @staticmethod
    def create_student_config(
        teacher_config: ModernTransformerConfig,
        n_layers: int | None = None,
        d_model: int | None = None,
        n_heads: int | None = None,
        hidden_dim: int | None = None,
    ) -> ModernTransformerConfig:
        """Derive a valid student configuration from a teacher configuration."""
        student_n_layers = (
            n_layers if n_layers is not None else max(1, teacher_config.n_layers // 2)
        )
        student_d_model = d_model if d_model is not None else teacher_config.d_model
        student_n_heads = n_heads if n_heads is not None else teacher_config.n_heads

        # Adjust n_heads if d_model changed and n_heads is incompatible
        if student_d_model % student_n_heads != 0:
            divisors = [h for h in range(1, student_d_model + 1) if student_d_model % h == 0]
            student_n_heads = max(h for h in divisors if h <= student_n_heads)

        student_n_kv_heads = min(student_n_heads, teacher_config.n_kv_heads or student_n_heads)
        while student_n_heads % student_n_kv_heads != 0:
            student_n_kv_heads -= 1

        student_hidden_dim = hidden_dim if hidden_dim is not None else teacher_config.hidden_dim

        cfg_dict = teacher_config.to_dict()
        cfg_dict.update(
            {
                "n_layers": student_n_layers,
                "d_model": student_d_model,
                "n_heads": student_n_heads,
                "n_kv_heads": student_n_kv_heads,
                "hidden_dim": student_hidden_dim,
            }
        )
        return ModernTransformerConfig(**cfg_dict)

    @staticmethod
    def select_layer_indices(teacher_layers: int, student_layers: int) -> list[int]:
        """Evenly select which teacher layers to retain in the student model.

        e.g. teacher=4, student=2 -> [0, 2]
        teacher=6, student=3 -> [0, 2, 4]
        """
        if student_layers >= teacher_layers:
            return list(range(student_layers))
        if student_layers == 1:
            return [0]

        indices = [
            round(i * (teacher_layers - 1) / (student_layers - 1)) for i in range(student_layers)
        ]
        return sorted(list(set(indices)))

    @classmethod
    def shrink_layers(
        cls,
        teacher_model: ModernTransformerLM,
        target_layer_indices: list[int] | None = None,
        student_config: ModernTransformerConfig | None = None,
    ) -> ModernTransformerLM:
        """Create an initialized student model by dropping layers and inheriting weights from teacher."""
        teacher_cfg = teacher_model.config
        if student_config is None:
            student_n_layers = max(1, teacher_cfg.n_layers // 2)
            student_config = cls.create_student_config(teacher_cfg, n_layers=student_n_layers)

        if target_layer_indices is None:
            target_layer_indices = cls.select_layer_indices(
                teacher_cfg.n_layers, student_config.n_layers
            )

        if len(target_layer_indices) != student_config.n_layers:
            raise ValueError(
                f"target_layer_indices length ({len(target_layer_indices)}) must match "
                f"student n_layers ({student_config.n_layers})"
            )

        student_model = ModernTransformerLM(student_config)

        # 1. Copy token embedding if dimensions match
        if student_model.tok_emb.weight.shape == teacher_model.tok_emb.weight.shape:
            student_model.tok_emb.weight.data.copy_(teacher_model.tok_emb.weight.data)

        # 2. Copy final norm if dimensions match
        if (
            hasattr(student_model, "norm_f")
            and hasattr(teacher_model, "norm_f")
            and student_model.norm_f.weight.shape == teacher_model.norm_f.weight.shape
        ):
            student_model.norm_f.weight.data.copy_(teacher_model.norm_f.weight.data)

        # 3. Copy output head if dimensions match
        if (
            not student_config.tie_weights
            and student_model.output_head.weight.shape == teacher_model.output_head.weight.shape
        ):
            student_model.output_head.weight.data.copy_(teacher_model.output_head.weight.data)

        # 4. Copy selected transformer blocks
        for s_idx, t_idx in enumerate(target_layer_indices):
            t_block = teacher_model.blocks[t_idx]
            s_block = student_model.blocks[s_idx]

            t_state = t_block.state_dict()
            s_state = s_block.state_dict()
            compatible = True
            for k in s_state:
                if k not in t_state or s_state[k].shape != t_state[k].shape:
                    compatible = False
                    break
            if compatible:
                s_block.load_state_dict(t_state)

        return student_model

    @staticmethod
    def forward_with_hidden_states(
        model: ModernTransformerLM,
        idx: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """Execute forward pass and collect intermediate hidden states after each block.

        Returns:
            (logits, block_hidden_states, final_hidden_state)
        """
        _B, T = idx.shape
        if T > model.config.max_context_length:
            raise ValueError(
                f"Sequence length ({T}) exceeds maximum context length ({model.config.max_context_length})"
            )

        x = model.drop(model.tok_emb(idx))
        hidden_states = []

        for layer_idx, block in enumerate(model.blocks):
            x = block(x, kv_cache=None, layer_idx=layer_idx, start_pos=0)
            hidden_states.append(x)

        normed_x = model.norm_f(x)
        logits = model.output_head(normed_x)

        return logits, hidden_states, normed_x

    @staticmethod
    def compute_compression_stats(
        teacher_model: ModernTransformerLM,
        student_model: ModernTransformerLM,
    ) -> dict[str, Any]:
        """Compute parameter and architectural compression metrics between teacher and student."""
        teacher_params = sum(p.numel() for p in teacher_model.parameters())
        student_params = sum(p.numel() for p in student_model.parameters())

        param_reduction_pct = (
            ((teacher_params - student_params) / teacher_params * 100.0)
            if teacher_params > 0
            else 0.0
        )
        compression_ratio = teacher_params / max(1, student_params)

        teacher_mb = (teacher_params * 4) / (1024 * 1024)
        student_mb = (student_params * 4) / (1024 * 1024)
        memory_savings_mb = teacher_mb - student_mb

        return {
            "teacher_params": teacher_params,
            "student_params": student_params,
            "compression_ratio": round(compression_ratio, 2),
            "param_reduction_pct": round(param_reduction_pct, 2),
            "teacher_memory_mb": round(teacher_mb, 3),
            "student_memory_mb": round(student_mb, 3),
            "memory_savings_mb": round(memory_savings_mb, 3),
            "teacher_layers": teacher_model.config.n_layers,
            "student_layers": student_model.config.n_layers,
            "teacher_d_model": teacher_model.config.d_model,
            "student_d_model": student_model.config.d_model,
        }
