"""
Libra Vision - Multi-Modal Projection Adapter
References:
- Liu et al., 2023 ("Visual Instruction Tuning", LLaVA)
- Alayrac et al., 2022 ("Flamingo: a Visual Language Model for Few-Shot Learning")

Projects visual patch representations from vision embedding dimension (d_vision)
into the language model's continuous token embedding space (d_model).
Supports:
1. Linear Projector: Single linear layer
2. MLP Projector: 2-layer MLP with GELU non-linearity (LLaVA-style)
3. Cross-Attention Perceiver Resampler: Latent query tokens cross-attending to visual patches
"""

from __future__ import annotations

import math
from enum import Enum
import torch
import torch.nn.functional as F
from torch import nn


class ProjectorType(str, Enum):
    LINEAR = "linear"
    MLP = "mlp"
    CROSS_ATTENTION = "cross_attention"


class VisionLanguageAdapter(nn.Module):
    """Aligns vision features with language token embeddings."""

    def __init__(
        self,
        vision_dim: int,
        llm_dim: int,
        projector_type: ProjectorType | str = ProjectorType.MLP,
        num_latents: int = 8,
    ) -> None:
        super().__init__()
        self.vision_dim = vision_dim
        self.llm_dim = llm_dim
        if isinstance(projector_type, ProjectorType):
            self.projector_type = projector_type
        else:
            self.projector_type = ProjectorType(str(projector_type).lower())
        self.num_latents = num_latents

        if self.projector_type == ProjectorType.LINEAR:
            self.net = nn.Linear(vision_dim, llm_dim)

        elif self.projector_type == ProjectorType.MLP:
            self.net = nn.Sequential(
                nn.Linear(vision_dim, llm_dim),
                nn.GELU(),
                nn.Linear(llm_dim, llm_dim),
            )

        elif self.projector_type == ProjectorType.CROSS_ATTENTION:
            # Perceiver-style cross attention
            self.latents = nn.Parameter(torch.randn(1, num_latents, llm_dim) * 0.02)
            self.k_proj = nn.Linear(vision_dim, llm_dim)
            self.v_proj = nn.Linear(vision_dim, llm_dim)
            self.out_proj = nn.Linear(llm_dim, llm_dim)

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_features: Tensor of shape (B, num_patches, vision_dim)

        Returns:
            Projected visual tokens of shape (B, num_visual_tokens, llm_dim)
        """
        B, N, D = visual_features.shape

        if self.projector_type in (ProjectorType.LINEAR, ProjectorType.MLP):
            return self.net(visual_features)  # (B, N, llm_dim)

        # Cross-attention Perceiver Resampler
        q = self.latents.expand(B, -1, -1)  # (B, num_latents, llm_dim)
        k = self.k_proj(visual_features)     # (B, N, llm_dim)
        v = self.v_proj(visual_features)     # (B, N, llm_dim)

        scale = 1.0 / math.sqrt(self.llm_dim)
        scores = torch.bmm(q, k.transpose(1, 2)) * scale  # (B, num_latents, N)
        attn = F.softmax(scores, dim=-1)
        out = torch.bmm(attn, v)  # (B, num_latents, llm_dim)
        return self.out_proj(out)
