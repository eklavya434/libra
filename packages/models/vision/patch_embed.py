"""
Libra Vision - Image Patch Embedding Module
Reference: Dosovitskiy et al., 2020 ("An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale")

Extracts non-overlapping 2D image patches, flattens them into 1D vectors,
applies linear projection to vision embedding dimension, and adds learned spatial position embeddings.
"""

from __future__ import annotations

import torch
from torch import nn


class ImagePatchEmbedder(nn.Module):
    """Decomposes an image into flattened patch embeddings with spatial positional encoding."""

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 8,
        in_channels: int = 3,
        embed_dim: int = 64,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size ({image_size}) must be divisible by patch_size ({patch_size})"
            )

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        # Linear projection mapping flattened patch (P * P * C) -> embed_dim
        self.patch_dim = patch_size * patch_size * in_channels
        self.projection = nn.Linear(self.patch_dim, embed_dim)

        # Learned 1D/2D spatial positional embeddings for all patches
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image tensor of shape (B, C, H, W)

        Returns:
            Patch embedding tensor of shape (B, num_patches, embed_dim)
        """
        B, C, H, W = x.shape
        if H != self.image_size or W != self.image_size:
            raise ValueError(
                f"Input image resolution ({H}x{W}) does not match expected ({self.image_size}x{self.image_size})"
            )

        P = self.patch_size
        # Reshape (B, C, grid_h, P, grid_w, P) -> (B, grid_h * grid_w, P * P * C)
        x = x.view(B, C, self.grid_size, P, self.grid_size, P)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        x = x.view(B, self.num_patches, self.patch_dim)

        # Project flattened patches into continuous embedding space
        embeddings = self.projection(x)  # (B, num_patches, embed_dim)

        # Add spatial position embeddings
        embeddings = embeddings + self.pos_embedding
        return embeddings


def generate_synthetic_image(
    pattern: str = "checkerboard",
    image_size: int = 32,
    channels: int = 3,
) -> torch.Tensor:
    """Generates synthetic deterministic test images without external file downloads."""
    image = torch.zeros(1, channels, image_size, image_size, dtype=torch.float32)

    if pattern == "checkerboard":
        grid = image_size // 4
        for r in range(image_size):
            for c in range(image_size):
                if ((r // grid) + (c // grid)) % 2 == 0:
                    image[0, 0, r, c] = 1.0  # Red
                else:
                    image[0, 2, r, c] = 1.0  # Blue
    elif pattern == "horizontal_gradient":
        for c in range(image_size):
            image[0, :, :, c] = float(c) / float(image_size)
    else:  # solid
        image.fill_(0.5)

    return image
