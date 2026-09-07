"""
Libra API v1 - Dynamic Model Routing & Speculative Decoding Endpoints
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from packages.models.modern_config import ModernTransformerConfig
from packages.models.modern_transformer import ModernTransformerLM
from packages.models.speculative import (
    SpeculativeDecoder,
    SpeculativeResult,
)
from packages.routing.classifier import (
    QueryClassification,
    get_query_classifier,
)
from packages.routing.dynamic_router import (
    RoutingDecision,
    RoutingPolicy,
    get_dynamic_router,
)

router = APIRouter(prefix="/routing", tags=["Dynamic Routing"])


class ClassifyRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Prompt text to classify")
    history: list[dict[str, Any]] | None = Field(
        default=None, description="Optional previous conversation history"
    )


class DecisionRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Prompt text to route")
    policy: RoutingPolicy = Field(default=RoutingPolicy.AUTO, description="Routing policy")
    prefer_local: bool = Field(default=True, description="Prioritize local models if available")
    history: list[dict[str, Any]] | None = Field(
        default=None, description="Optional previous conversation history"
    )


class RouteGenerateRequest(BaseModel):
    messages: list[dict[str, Any]] = Field(..., min_length=1, description="Dialogue message list")
    policy: RoutingPolicy = Field(default=RoutingPolicy.AUTO, description="Routing policy")
    prefer_local: bool = Field(default=True, description="Prioritize local models if available")


class SpeculativeGenerateRequest(BaseModel):
    prompt: str = Field(default="Hello world", description="Prompt string")
    max_new_tokens: int = Field(default=16, ge=1, le=64, description="Tokens to generate")
    lookahead_k: int = Field(default=3, ge=1, le=8, description="Draft lookahead window K")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="Sampling temperature")


# Cached lightweight models for educational speculative decoding endpoints
_draft_model: ModernTransformerLM | None = None
_target_model: ModernTransformerLM | None = None


def _get_speculative_models() -> tuple[ModernTransformerLM, ModernTransformerLM]:
    global _draft_model, _target_model
    if _draft_model is None or _target_model is None:
        # 1-layer lightweight draft model
        draft_cfg = ModernTransformerConfig(
            vocab_size=256,
            d_model=64,
            n_layers=1,
            n_heads=2,
            max_context_length=128,
            hidden_dim=128,
        )
        # 3-layer authoritative target model
        target_cfg = ModernTransformerConfig(
            vocab_size=256,
            d_model=64,
            n_layers=3,
            n_heads=2,
            max_context_length=128,
            hidden_dim=128,
        )
        _draft_model = ModernTransformerLM(draft_cfg)
        _target_model = ModernTransformerLM(target_cfg)
    return _draft_model, _target_model


@router.post("/classify", response_model=QueryClassification)
async def classify_prompt(request: ClassifyRequest) -> QueryClassification:
    """Classifies user prompt intent, complexity scores, and recommended tier."""
    classifier = get_query_classifier()
    return classifier.classify(request.prompt, conversation_history=request.history)


@router.post("/decision", response_model=RoutingDecision)
async def plan_routing_decision(request: DecisionRequest) -> RoutingDecision:
    """Determines tier, model, provider, and fallback cascade plan."""
    dynamic_router = get_dynamic_router()
    return dynamic_router.plan_routing(
        prompt=request.prompt,
        policy=request.policy,
        prefer_local=request.prefer_local,
        conversation_history=request.history,
    )


@router.post("/generate")
async def route_and_generate(request: RouteGenerateRequest) -> dict[str, Any]:
    """Dynamically routes request to optimal tier and executes with fallback protection."""
    dynamic_router = get_dynamic_router()
    try:
        return await dynamic_router.route_and_execute(
            messages=request.messages,
            policy=request.policy,
            prefer_local=request.prefer_local,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/speculative", response_model=SpeculativeResult)
async def speculative_generate_endpoint(request: SpeculativeGenerateRequest) -> SpeculativeResult:
    """
    Executes first-principles speculative decoding between draft and target models.
    Guarantees exact mathematical equivalence to target model while reporting telemetry.
    """
    draft_m, target_m = _get_speculative_models()
    prompt_tokens = list(request.prompt.encode("utf-8"))
    if not prompt_tokens:
        prompt_tokens = [ord(" ")]

    decoder = SpeculativeDecoder(
        target_model=target_m,
        draft_model=draft_m,
        lookahead_k=request.lookahead_k,
        temperature=request.temperature,
    )

    result = decoder.generate(
        prompt_tokens=prompt_tokens,
        max_new_tokens=request.max_new_tokens,
    )
    return result
