"""
Unit Tests for Phase 32: Vision Patch Embedder & Projection Adapter
"""

import pytest
import torch

from packages.models.vision.patch_embed import ImagePatchEmbedder, generate_synthetic_image
from packages.models.vision.vlm_projector import ProjectorType, VisionLanguageAdapter


def test_image_patch_embedder_shapes():
    image_size = 32
    patch_size = 8
    in_channels = 3
    embed_dim = 64
    batch_size = 2

    embedder = ImagePatchEmbedder(
        image_size=image_size,
        patch_size=patch_size,
        in_channels=in_channels,
        embed_dim=embed_dim,
    )

    expected_patches = (image_size // patch_size) ** 2  # (32/8)^2 = 16
    assert embedder.num_patches == expected_patches
    assert embedder.patch_dim == patch_size * patch_size * in_channels  # 8*8*3 = 192

    img = torch.randn(batch_size, in_channels, image_size, image_size)
    patches = embedder(img)

    assert patches.shape == (batch_size, expected_patches, embed_dim)


def test_synthetic_image_patterns():
    checker = generate_synthetic_image("checkerboard", image_size=16)
    gradient = generate_synthetic_image("horizontal_gradient", image_size=16)

    assert checker.shape == (1, 3, 16, 16)
    assert gradient.shape == (1, 3, 16, 16)
    assert checker.max() <= 1.0 and checker.min() >= 0.0


def test_vision_language_adapter_mlp():
    vision_dim = 64
    llm_dim = 128
    batch_size = 2
    num_patches = 16

    adapter = VisionLanguageAdapter(
        vision_dim=vision_dim,
        llm_dim=llm_dim,
        projector_type=ProjectorType.MLP,
    )

    visual_features = torch.randn(batch_size, num_patches, vision_dim)
    projected = adapter(visual_features)

    assert projected.shape == (batch_size, num_patches, llm_dim)


def test_vision_language_adapter_cross_attention():
    vision_dim = 64
    llm_dim = 128
    batch_size = 2
    num_patches = 16
    num_latents = 4

    adapter = VisionLanguageAdapter(
        vision_dim=vision_dim,
        llm_dim=llm_dim,
        projector_type=ProjectorType.CROSS_ATTENTION,
        num_latents=num_latents,
    )

    visual_features = torch.randn(batch_size, num_patches, vision_dim)
    projected = adapter(visual_features)

    # Cross-attention resamples N patches into fixed number of latent tokens
    assert projected.shape == (batch_size, num_latents, llm_dim)
