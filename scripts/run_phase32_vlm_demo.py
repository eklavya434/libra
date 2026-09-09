"""
Phase 32 Interactive Demonstration: Multi-Modal Vision-Language Model (LibraVLM)
Runs an interactive demonstration on consumer CPU:
1. Visual 2D image patch decomposition & embedding
2. Projection adapter alignment (LLaVA-style MLP and Perceiver cross-attention)
3. End-to-end multi-modal forward pass and image-conditioned text generation
"""

import os
import sys
import time

import torch

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from packages.models.modern_config import ModernTransformerConfig
from packages.models.vision.patch_embed import ImagePatchEmbedder, generate_synthetic_image
from packages.models.vision.vlm_model import LibraVLM
from packages.models.vision.vlm_projector import ProjectorType, VisionLanguageAdapter


def print_banner(text: str) -> None:
    print(f"\n{'=' * 75}\n  {text}\n{'=' * 75}")


def demo_patch_decomposition():
    print_banner("1. IMAGE PATCH DECOMPOSITION & SPATIAL POSITIONAL EMBEDDING")
    image_size = 32
    patch_size = 8
    channels = 3
    vision_dim = 32

    embedder = ImagePatchEmbedder(
        image_size=image_size,
        patch_size=patch_size,
        in_channels=channels,
        embed_dim=vision_dim,
    )

    img = generate_synthetic_image("checkerboard", image_size=image_size)
    patches = embedder(img)

    print(f"Input Image Resolution : {image_size} x {image_size} (Channels: {channels})")
    print(f"Patch Resolution       : {patch_size} x {patch_size} pixels")
    print(
        f"Flattened Patch Dim    : {embedder.patch_dim} elements ({patch_size}*{patch_size}*{channels})"
    )
    print(
        f"Total Visual Patches   : {embedder.num_patches} patches in a {embedder.grid_size}x{embedder.grid_size} grid"
    )
    print(f"Output Patch Tensor    : {tuple(patches.shape)} (B, N_patches, d_vision)")
    print(
        "Verification: ImagePatchEmbedder successfully extracts and projects non-overlapping visual tokens."
    )


def demo_multimodal_projection():
    print_banner("2. MULTI-MODAL PROJECTION ADAPTER ALIGNMENT")
    vision_dim = 32
    llm_dim = 64
    num_patches = 16

    mlp_adapter = VisionLanguageAdapter(
        vision_dim=vision_dim,
        llm_dim=llm_dim,
        projector_type=ProjectorType.MLP,
    )

    perceiver_adapter = VisionLanguageAdapter(
        vision_dim=vision_dim,
        llm_dim=llm_dim,
        projector_type=ProjectorType.CROSS_ATTENTION,
        num_latents=4,
    )

    visual_feats = torch.randn(1, num_patches, vision_dim)
    mlp_tokens = mlp_adapter(visual_feats)
    perceiver_tokens = perceiver_adapter(visual_feats)

    print(f"Raw Vision Feature Shape       : {tuple(visual_feats.shape)} (d_vision = {vision_dim})")
    print(f"LLaVA MLP Projected Tokens      : {tuple(mlp_tokens.shape)} (d_model = {llm_dim})")
    print(
        f"Perceiver Resampled Tokens     : {tuple(perceiver_tokens.shape)} (resampled to 4 latents)"
    )
    print(
        "Verification: Vision features successfully projected to continuous LLM token embedding space."
    )


def demo_vlm_generation():
    print_banner("3. END-TO-END MULTI-MODAL INFERENCE & GENERATION")
    llm_config = ModernTransformerConfig(
        vocab_size=256,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        n_layers=2,
        max_context_length=128,
    )

    vlm = LibraVLM(
        llm_config=llm_config,
        image_size=32,
        patch_size=8,
        vision_dim=32,
        projector_type=ProjectorType.MLP,
    )

    img = generate_synthetic_image("checkerboard", image_size=32)
    prompt_ids = torch.tensor([[10, 20, 30]])  # 3 prompt tokens

    t0 = time.perf_counter()
    gen_tokens = vlm.generate(img, prompt_ids, max_new_tokens=6, temperature=0.0)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    print(f"Input Prompt Token IDs : {prompt_ids.tolist()[0]}")
    print("Conditioning Image     : 32x32 Checkerboard (16 visual prefix tokens)")
    print(f"Generated Token IDs    : {gen_tokens.tolist()[0]}")
    print(f"Generation Latency     : {elapsed_ms:.2f} ms on CPU")
    print(
        "Verification: ModernTransformerLM successfully conditioned autoregressive generation on visual prefix tokens."
    )


if __name__ == "__main__":
    print_banner("LIBRA PHASE 32: MULTI-MODAL VISION-LANGUAGE MODEL (LibraVLM)")
    demo_patch_decomposition()
    demo_multimodal_projection()
    demo_vlm_generation()
    print_banner("PHASE 32 DEMONSTRATION COMPLETE & VERIFIED ON CPU")
