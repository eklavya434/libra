"""
Libra Backend API - Multi-Modal (Vision-Language) Endpoints (Phase 32)
Provides image patch decomposition, embedding extraction, and multi-modal generation.
"""

from __future__ import annotations

import time
from typing import Any

import torch
from fastapi import APIRouter
from pydantic import BaseModel, Field

from packages.models.modern_config import ModernTransformerConfig
from packages.models.vision.patch_embed import ImagePatchEmbedder, generate_synthetic_image
from packages.models.vision.vlm_model import LibraVLM
from packages.models.vision.vlm_projector import ProjectorType, VisionLanguageAdapter

router = APIRouter(prefix="/multimodal", tags=["Multi-Modal & Vision-Language"])

# Global educational VLM instance on CPU
_llm_config = ModernTransformerConfig(
    vocab_size=256,
    d_model=64,
    n_heads=4,
    n_kv_heads=2,
    n_layers=2,
    max_context_length=128,
)
_global_vlm = LibraVLM(
    llm_config=_llm_config,
    image_size=32,
    patch_size=8,
    vision_dim=32,
    projector_type=ProjectorType.MLP,
)


class EmbedImageRequest(BaseModel):
    pattern: str = Field(
        default="checkerboard", description="checkerboard, horizontal_gradient, solid"
    )
    image_size: int = Field(default=32, ge=16, le=64)
    patch_size: int = Field(default=8, ge=4, le=16)
    vision_dim: int = Field(default=32)


class EmbedImageResponse(BaseModel):
    image_size: int
    patch_size: int
    num_patches: int
    patch_dim: int
    vision_dim: int
    projected_llm_dim: int
    patch_grid: list[dict[str, Any]]


class VLMGenerateRequest(BaseModel):
    prompt: str = Field(default="This image shows a", description="Prefix text prompt")
    pattern: str = Field(
        default="checkerboard", description="checkerboard, horizontal_gradient, solid"
    )
    max_new_tokens: int = Field(default=6, ge=1, le=16)


class VLMGenerateResponse(BaseModel):
    prompt: str
    image_pattern: str
    num_visual_tokens: int
    generated_tokens: list[int]
    generated_text: str
    latency_ms: float


@router.post("/embed_image", response_model=EmbedImageResponse)
async def embed_image_endpoint(req: EmbedImageRequest) -> EmbedImageResponse:
    """Decomposes an image into patches and reports projection metrics."""
    embedder = ImagePatchEmbedder(
        image_size=req.image_size,
        patch_size=req.patch_size,
        embed_dim=req.vision_dim,
    )
    adapter = VisionLanguageAdapter(
        vision_dim=req.vision_dim,
        llm_dim=_llm_config.d_model,
        projector_type=ProjectorType.MLP,
    )

    img = generate_synthetic_image(pattern=req.pattern, image_size=req.image_size)
    patches = embedder(img)
    projected = adapter(patches)

    grid_dim = req.image_size // req.patch_size
    patch_grid = []
    for r in range(grid_dim):
        for c in range(grid_dim):
            idx = r * grid_dim + c
            # Mean intensity of patch across channels
            patch_slice = img[
                0,
                :,
                r * req.patch_size : (r + 1) * req.patch_size,
                c * req.patch_size : (c + 1) * req.patch_size,
            ]
            mean_rgb = [round(patch_slice[ch].mean().item(), 3) for ch in range(3)]
            patch_grid.append(
                {
                    "patch_index": idx,
                    "row": r,
                    "col": c,
                    "mean_rgb": mean_rgb,
                    "norm": round(patches[0, idx].norm().item(), 4),
                }
            )

    return EmbedImageResponse(
        image_size=req.image_size,
        patch_size=req.patch_size,
        num_patches=embedder.num_patches,
        patch_dim=embedder.patch_dim,
        vision_dim=req.vision_dim,
        projected_llm_dim=_llm_config.d_model,
        patch_grid=patch_grid,
    )


@router.post("/generate", response_model=VLMGenerateResponse)
async def vlm_generate_endpoint(req: VLMGenerateRequest) -> VLMGenerateResponse:
    """Generates text conditioned on visual image tokens."""
    img = generate_synthetic_image(pattern=req.pattern, image_size=32)
    # Simple hash tokenization for prompt
    prompt_ids = torch.tensor([[hash(w) % 256 for w in req.prompt.split()]], dtype=torch.long)
    if prompt_ids.size(1) == 0:
        prompt_ids = torch.tensor([[1]], dtype=torch.long)

    t0 = time.perf_counter()
    gen_tokens = _global_vlm.generate(img, prompt_ids, max_new_tokens=req.max_new_tokens)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    gen_list = gen_tokens[0].tolist()
    # Mock vocabulary decode for demonstration
    mock_vocab = {
        "checkerboard": "high-contrast alternating geometric grid pattern.",
        "horizontal_gradient": "smooth horizontal color luminance gradient.",
        "solid": "uniform monochromatic tone.",
    }
    desc = mock_vocab.get(req.pattern, "visual pattern.")
    simulated_text = f"{req.prompt} {desc}"

    return VLMGenerateResponse(
        prompt=req.prompt,
        image_pattern=req.pattern,
        num_visual_tokens=_global_vlm.vision_encoder.num_patches,
        generated_tokens=gen_list,
        generated_text=simulated_text,
        latency_ms=round(latency_ms, 2),
    )
