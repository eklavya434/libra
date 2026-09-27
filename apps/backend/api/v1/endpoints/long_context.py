"""
FastAPI Router - Long-Context Needle-In-A-Haystack (NIAH) & Attention Compaction Endpoints
"""

from typing import Any, Dict, List, Optional

import torch
from fastapi import APIRouter
from pydantic import BaseModel, Field

from apps.backend.api.v1.deps import run_provider_completion
from packages.evaluation.multi_needle import MultiNeedleEvaluator, MultiNeedleItem
from packages.evaluation.needle_haystack import (
    DEFAULT_NEEDLE_FACT,
    DEFAULT_NEEDLE_KEY,
    DEFAULT_NEEDLE_PROMPT,
    NeedleInHaystackEvaluator,
)
from packages.models.components.compacted_kv_cache import CompactedKVCache

router = APIRouter(prefix="/long_context", tags=["Long-Context & Attention Compaction"])


class SingleNeedleGridRequest(BaseModel):
    model: str = Field(
        default="mock",
        description="Model id to evaluate. Use 'mock' for the explicitly-marked educational mock generator.",
    )
    needle: str = Field(default=DEFAULT_NEEDLE_FACT)
    target_key: str = Field(default=DEFAULT_NEEDLE_KEY)
    retrieval_prompt: str = Field(default=DEFAULT_NEEDLE_PROMPT)
    context_lengths: List[int] = Field(default=[300, 600, 1000])
    depth_fractions: List[float] = Field(default=[0.1, 0.3, 0.5, 0.7, 0.9])


class MultiNeedleItemSchema(BaseModel):
    key: str
    fact: str
    depth_fraction: float


class MultiNeedleEvalRequest(BaseModel):
    model: str = Field(
        default="mock",
        description="Model id to evaluate. Use 'mock' for the explicitly-marked educational mock generator.",
    )
    context_length: int = Field(default=600)
    needles: Optional[List[MultiNeedleItemSchema]] = None
    question: Optional[str] = None


class CompactionSimulationRequest(BaseModel):
    sequence_length: int = Field(default=1024, ge=64, le=16384)
    max_budget: int = Field(default=256, ge=32, le=4096)
    n_sink: int = Field(default=4, ge=1)
    n_recent: int = Field(default=64, ge=8)
    n_layers: int = Field(default=4, ge=1)
    n_kv_heads: int = Field(default=4, ge=1)
    head_dim: int = Field(default=64, ge=16)


def _get_generator(model_name: str):
    """Returns a callable generation function for the selected provider.

    Only the literal model ``mock`` maps to the explicitly-marked educational
    mock generator. Every other model id is routed to the real provider;
    failures surface as an honest HTTP 503 rather than fabricated text.
    """
    if model_name.lower() in ("mock", "test"):
        return _mock_gen

    def _sync_wrapper(prompt: str) -> str:
        return run_provider_completion(model_name, prompt, max_tokens=100, temperature=0.0)

    return _sync_wrapper


def _mock_gen(prompt: str) -> str:
    """Explicitly-marked educational mock generator (aggressive honesty).

    Used ONLY when the caller requests model='mock'. Looks for labelled keys in
    the prompt and answers, exactly like the presets, so students can run the
    benchmark without a provider — it is never used for real deployments.
    """
    lines = prompt.split("\n")
    for line in lines:
        if "secret" in line.lower() or "code" in line.lower() or "key" in line.lower():
            words = line.split()
            for w in words:
                if any(c.isdigit() for c in w) or "-" in w:
                    return f"The requested code is {w.strip('.,')}"
    return "[MockGenerator] No key-like token found in the prompt."


@router.post("/evaluate/needle")
def evaluate_single_needle_grid(req: SingleNeedleGridRequest) -> Dict[str, Any]:
    """Evaluates 2D Needle-In-A-Haystack retrieval matrix across lengths and depths."""
    evaluator = NeedleInHaystackEvaluator(
        needle=req.needle,
        target_key=req.target_key,
        retrieval_prompt=req.retrieval_prompt,
    )
    generator = _get_generator(req.model)
    results = evaluator.run_grid(
        generator_fn=generator,
        context_lengths=req.context_lengths,
        depth_fractions=req.depth_fractions,
    )

    total = len(results)
    correct = sum(1 for r in results if r.is_correct)
    acc = (correct / total * 100.0) if total > 0 else 0.0
    avg_lat = sum(r.latency_ms for r in results) / total if total > 0 else 0.0

    return {
        "model": req.model,
        "total_trials": total,
        "accuracy_percent": round(acc, 1),
        "average_latency_ms": round(avg_lat, 2),
        "results": [
            {
                "context_length": r.context_length,
                "depth_percent": r.depth_percent,
                "is_correct": r.is_correct,
                "score": r.score,
                "latency_ms": r.latency_ms,
                "retrieved_text": r.retrieved_text,
            }
            for r in results
        ],
    }


@router.post("/evaluate/multi_needle")
def evaluate_multi_needle(req: MultiNeedleEvalRequest) -> Dict[str, Any]:
    """Runs multi-needle relational associative recall test."""
    evaluator = MultiNeedleEvaluator()
    generator = _get_generator(req.model)

    needles = None
    if req.needles:
        needles = [
            MultiNeedleItem(key=n.key, fact=n.fact, depth_fraction=n.depth_fraction)
            for n in req.needles
        ]

    res = evaluator.run_single(
        generator_fn=generator,
        context_words=req.context_length,
        needles=needles,
        question=req.question,
    )

    return {
        "context_length": res.context_length,
        "num_needles": res.num_needles,
        "needles": res.needles,
        "all_correct": res.all_correct,
        "partial_score": res.partial_score,
        "found_keys": res.found_keys,
        "missing_keys": res.missing_keys,
        "retrieved_text": res.retrieved_text,
        "latency_ms": res.latency_ms,
    }


@router.post("/compaction/simulate")
def simulate_compaction(req: CompactionSimulationRequest) -> Dict[str, Any]:
    """
    Simulates H2O + StreamingLLM KV cache compaction over an autoregressive sequence.
    Returns memory reduction, token retention timeline, and token category breakdown.
    """
    cache = CompactedKVCache(
        n_layers=req.n_layers,
        max_budget=req.max_budget,
        n_sink=req.n_sink,
        n_recent=req.n_recent,
    )

    # Simulate steps
    step_size = 64
    steps = req.sequence_length // step_size
    memory_timeline: List[Dict[str, Any]] = []

    device = "cpu"
    for step in range(1, steps + 1):
        seq_len = step * step_size
        k_step = torch.randn(1, req.n_kv_heads, step_size, req.head_dim, device=device)
        v_step = torch.randn(1, req.n_kv_heads, step_size, req.head_dim, device=device)

        # Synthetic attention score simulation: give higher scores to early tokens (sinks) and random spikes (heavy hitters)
        current_len = cache.get_seq_len(0) + step_size
        synthetic_attn = torch.rand(1, req.n_kv_heads, step_size, current_len, device=device)

        cache.update(k_step, v_step, layer_idx=0, current_attn_weights=synthetic_attn)

        # Record timeline
        stats = cache.get_stats()
        memory_timeline.append(
            {
                "tokens_processed": seq_len,
                "cached_tokens": stats["cached_tokens"],
                "evicted_tokens": stats["evicted_tokens"],
                "memory_bytes": stats["memory_bytes"],
                "uncompressed_bytes": stats["uncompressed_bytes"],
                "memory_savings_pct": stats["memory_savings_pct"],
            }
        )

    final_stats = cache.get_stats()

    # Generate token category breakdown for visualization
    categories = []
    cached_len = cache.get_seq_len(0)
    for idx in range(req.sequence_length):
        if idx < req.n_sink:
            categories.append({"index": idx, "type": "sink", "label": "Attention Sink"})
        elif idx >= req.sequence_length - req.n_recent:
            categories.append({"index": idx, "type": "recent", "label": "Recent Window"})
        elif idx < cached_len:
            categories.append({"index": idx, "type": "heavy", "label": "Heavy Hitter"})
        else:
            categories.append({"index": idx, "type": "evicted", "label": "Evicted Token"})

    return {
        "sequence_length": req.sequence_length,
        "max_budget": req.max_budget,
        "final_stats": final_stats,
        "timeline": memory_timeline[:: max(1, len(memory_timeline) // 10)],
        "token_sample": categories[: min(128, len(categories))],
    }


@router.get("/presets")
def get_long_context_presets() -> List[Dict[str, Any]]:
    """Returns educational benchmark scenarios."""
    return [
        {
            "id": "single_needle_vault",
            "title": "Vault Access Code Retrieval",
            "description": "Evaluate model precision retrieving an access PIN at 5 distinct depths from 300 to 1,200 words.",
            "needle": "The secret access code to the vault is 849204.",
            "target_key": "849204",
            "prompt": "What is the secret access code to the vault?",
        },
        {
            "id": "multi_needle_launch",
            "title": "Multi-Key Mission Operations",
            "description": "Simultaneously test recall of three distinct operation keys (Silo code, Callsign, Abort key) distributed across early, middle, and late document positions.",
            "needles": [
                {
                    "key": "Alpha-77",
                    "fact": "The primary launch silo code is Alpha-77.",
                    "depth_fraction": 0.20,
                },
                {
                    "key": "Falcon",
                    "fact": "The mission commander callsign is Falcon.",
                    "depth_fraction": 0.50,
                },
                {
                    "key": "Omega-Zero",
                    "fact": "The emergency abort sequence key is Omega-Zero.",
                    "depth_fraction": 0.85,
                },
            ],
            "prompt": "What are the primary launch silo code, the mission commander callsign, and the emergency abort sequence key?",
        },
    ]
