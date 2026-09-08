"""
Libra Vision - End-to-End Vision-Language Model (LibraVLM)
Integrates:
1. ImagePatchEmbedder: Raw pixels -> Visual patch embeddings
2. VisionLanguageAdapter: Visual patch embeddings -> LLM continuous token embeddings
3. ModernTransformerLM: Causal decoder processing interleaved [visual_tokens; text_tokens]
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from packages.models.modern_config import ModernTransformerConfig
from packages.models.modern_transformer import ModernTransformerLM
from packages.models.vision.patch_embed import ImagePatchEmbedder
from packages.models.vision.vlm_projector import ProjectorType, VisionLanguageAdapter


class LibraVLM(nn.Module):
    """Educational Vision-Language Model."""

    def __init__(
        self,
        llm_config: ModernTransformerConfig,
        image_size: int = 32,
        patch_size: int = 8,
        in_channels: int = 3,
        vision_dim: int = 64,
        projector_type: ProjectorType | str = ProjectorType.MLP,
    ) -> None:
        super().__init__()
        self.llm_config = llm_config

        # 1. Vision Patch Embedder
        self.vision_encoder = ImagePatchEmbedder(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=vision_dim,
        )

        # 2. Multi-Modal Alignment Adapter
        self.projector = VisionLanguageAdapter(
            vision_dim=vision_dim,
            llm_dim=llm_config.d_model,
            projector_type=projector_type,
        )

        # 3. Base Language Model Backbone
        self.language_model = ModernTransformerLM(llm_config)

    def forward_embeddings(
        self,
        images: torch.Tensor | None,
        text_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Constructs interleaved multi-modal input embeddings."""
        # Text embeddings: (B, T_text, d_model)
        text_emb = self.language_model.tok_emb(text_ids)

        if images is not None:
            # Visual embeddings: (B, N_patches, vision_dim) -> (B, N_patches, d_model)
            vision_features = self.vision_encoder(images)
            visual_tokens = self.projector(vision_features)

            # Interleaved prefix sequence: [visual_tokens, text_tokens]
            full_emb = torch.cat([visual_tokens, text_emb], dim=1)
            return full_emb

        return text_emb

    def forward(
        self,
        images: torch.Tensor | None,
        text_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass through the multimodal transformer.

        Args:
            images: Tensor of shape (B, C, H, W) or None
            text_ids: Text token tensor of shape (B, T_text)
            targets: Optional text targets of shape (B, T_text) or full sequence (B, T_total)
        """
        x = self.forward_embeddings(images, text_ids)
        x = self.language_model.drop(x)

        for layer_idx, block in enumerate(self.language_model.blocks):
            x = block(x, layer_idx=layer_idx, start_pos=0)

        x = self.language_model.norm_f(x)
        logits = self.language_model.output_head(x)  # (B, T_total, vocab_size)

        loss = None
        if targets is not None:
            # If targets match text sequence length, align with tail of logits
            t_text = text_ids.size(1)
            text_logits = logits[:, -t_text:, :]
            loss = F.cross_entropy(text_logits.reshape(-1, text_logits.size(-1)), targets.reshape(-1))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor | None,
        prompt_text_ids: torch.Tensor,
        max_new_tokens: int = 8,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        """Autoregressively generates text tokens conditioned on image and prompt."""
        generated = prompt_text_ids.clone()

        for _ in range(max_new_tokens):
            logits, _ = self.forward(images, generated)
            next_token_logits = logits[:, -1, :]

            if temperature == 0.0:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

        return generated[:, prompt_text_ids.size(1):]
