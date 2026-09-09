"""
Unit Tests for Phase 32: End-to-End LibraVLM
"""

import torch

from packages.models.modern_config import ModernTransformerConfig
from packages.models.vision.patch_embed import generate_synthetic_image
from packages.models.vision.vlm_model import LibraVLM
from packages.models.vision.vlm_projector import ProjectorType


def test_vlm_forward_loss_and_gradients():
    llm_config = ModernTransformerConfig(
        vocab_size=100,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        n_layers=2,
        max_context_length=64,
    )

    vlm = LibraVLM(
        llm_config=llm_config,
        image_size=16,
        patch_size=8,
        vision_dim=32,
        projector_type=ProjectorType.MLP,
    )

    img = generate_synthetic_image("checkerboard", image_size=16)  # (1, 3, 16, 16) -> 4 patches
    text_ids = torch.randint(0, 100, (1, 8))
    targets = torch.randint(0, 100, (1, 8))

    logits, loss = vlm(img, text_ids, targets=targets)

    # 4 image patches + 8 text tokens = 12 total tokens
    assert logits.shape == (1, 12, 100)
    assert loss is not None
    assert loss.item() > 0.0

    # Test backpropagation
    loss.backward()
    assert vlm.projector.net[0].weight.grad is not None
    assert vlm.vision_encoder.projection.weight.grad is not None


def test_vlm_autoregressive_generate():
    llm_config = ModernTransformerConfig(
        vocab_size=100,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        n_layers=2,
        max_context_length=64,
    )

    vlm = LibraVLM(
        llm_config=llm_config,
        image_size=16,
        patch_size=8,
        vision_dim=32,
    )

    img = generate_synthetic_image("checkerboard", image_size=16)
    prompt_ids = torch.tensor([[5, 10, 15]])

    gen_tokens = vlm.generate(img, prompt_ids, max_new_tokens=4, temperature=0.0)
    assert gen_tokens.shape == (1, 4)
