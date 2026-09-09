"""
Libra API v1 - Step-Level Verifiable Search & PRM Reasoning Endpoints (Phase 48)

Provides REST endpoints for:
- Executing PRM-verified tree/beam reasoning search with early pruning and backtracking
- Benchmarking Greedy vs Best-of-N vs Verifiable Search
- Educational multi-step math and logic problem presets
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from packages.evaluation.verifiable_search_eval import (
    VerifiableSearchComparisonResult,
    VerifiableSearchEvaluator,
)
from packages.models.reasoning.prm import ProcessRewardModel
from packages.models.reasoning.verifiable_search import (
    VerifiableSearchConfig,
    VerifiableSearchEngine,
    VerifiableSearchResult,
)

router = APIRouter(prefix="/verifiable-search", tags=["Verifiable Search & PRM Reasoning"])


class SolveRequest(BaseModel):
    """Request payload for executing verifiable reasoning search on a prompt."""

    prompt: str = Field(..., min_length=3, description="Problem prompt")
    beam_width: int = Field(default=3, ge=1, le=5, description="Search beam width")
    max_depth: int = Field(default=5, ge=1, le=8, description="Maximum reasoning steps")
    step_prune_threshold: float = Field(
        default=0.55, ge=0.1, le=0.9, description="PRM correctness threshold for pruning"
    )
    branching_factor: int = Field(default=2, ge=1, le=4, description="Candidate steps per node")


class BenchmarkProblem(BaseModel):
    """Problem prompt with expected solution answer."""

    prompt: str
    expected_answer: str


class BenchmarkRequest(BaseModel):
    """Request payload for comparative search benchmarking."""

    problems: list[BenchmarkProblem] = Field(
        default=[
            BenchmarkProblem(
                prompt="Calculate (15 * 4) + (24 / 3) - 17",
                expected_answer="51",
            ),
            BenchmarkProblem(
                prompt="Sarah has 12 apples, buys 8 more, eats 3, divides rest among 2 friends",
                expected_answer="8.5",
            ),
        ],
        description="Problems to benchmark across search paradigms",
    )
    n_best_of_n: int = Field(default=3, ge=2, le=5, description="Number of Best-of-N rollouts")


# Cached engine instance for fast educational API responses
_search_engine: VerifiableSearchEngine | None = None


def _get_engine() -> VerifiableSearchEngine:
    global _search_engine
    if _search_engine is None:
        prm = ProcessRewardModel()
        cfg = VerifiableSearchConfig(
            beam_width=3,
            max_depth=5,
            step_prune_threshold=0.55,
            branching_factor=2,
        )
        _search_engine = VerifiableSearchEngine(prm=prm, config=cfg)
    return _search_engine


@router.post("/solve")
async def solve_verifiable_reasoning(request: SolveRequest) -> dict[str, Any]:
    """
    Executes PRM step-level verifiable search with early pruning and backtracking.
    """
    engine = _get_engine()
    engine.config.beam_width = request.beam_width
    engine.config.max_depth = request.max_depth
    engine.config.step_prune_threshold = request.step_prune_threshold
    engine.config.branching_factor = request.branching_factor

    res: VerifiableSearchResult = engine.search(request.prompt)

    return {
        "prompt": res.prompt,
        "optimal_trajectory": res.optimal_trajectory,
        "final_answer": res.final_answer,
        "is_verified": res.is_verified,
        "total_steps_explored": res.total_steps_explored,
        "pruned_branches_count": res.pruned_branches_count,
        "backtracks_count": res.backtracks_count,
        "compute_savings_pct": res.compute_savings_pct,
        "tree_nodes": res.tree_nodes,
        "tree_edges": res.tree_edges,
    }


@router.post("/benchmark", response_model=VerifiableSearchComparisonResult)
async def benchmark_search_paradigms(request: BenchmarkRequest) -> VerifiableSearchComparisonResult:
    """
    Compares Greedy, Best-of-N (Outcome-only), and Verifiable PRM Search.
    """
    engine = _get_engine()
    prompt_pairs = [(p.prompt, p.expected_answer) for p in request.problems]

    result = VerifiableSearchEvaluator.compare_search_methods(
        engine=engine,
        prompts_with_expected=prompt_pairs,
        n_best_of_n=request.n_best_of_n,
    )
    return result


@router.get("/presets")
async def get_verifiable_search_presets() -> dict[str, Any]:
    """Returns educational math and logic presets designed to test PRM verification and pruning."""
    return {
        "presets": [
            {
                "id": "compound_arithmetic",
                "name": "Compound Arithmetic Evaluation",
                "prompt": "Calculate (15 * 4) + (24 / 3) - 17",
                "expected_answer": "51",
                "description": "Multi-term arithmetic with multiplication, division, and addition. Demonstrates PRM pruning arithmetic mismatches early.",
                "beam_width": 3,
                "step_prune_threshold": 0.55,
            },
            {
                "id": "multi_step_word_problem",
                "name": "Sarah's Apples (Multi-Step Deduction)",
                "prompt": "Sarah has 12 apples, buys 8 more, eats 3, divides rest among 2 friends",
                "expected_answer": "8.5",
                "description": "Sequential real-world scenario testing state tracking and intermediate step verification.",
                "beam_width": 2,
                "step_prune_threshold": 0.60,
            },
            {
                "id": "logic_contradiction",
                "name": "Logical Contradiction Detection",
                "prompt": "Deduce whether 0 = 1 given standard field axioms.",
                "expected_answer": "Valid",
                "description": "Demonstrates PRM semantic pattern matching pruning logical contradictions and impossible deductions.",
                "beam_width": 3,
                "step_prune_threshold": 0.55,
            },
        ],
        "verifiable_search_theory": {
            "prm_vs_orm": "PRMs evaluate intermediate steps r(s_t), while ORMs only score final output r(s_T).",
            "early_pruning": "Pruning a depth-1 error saves exploring up to B^(D-1) subsequent doomed nodes.",
            "test_time_scaling": "Allows trading additional CPU inference computation for higher mathematical accuracy.",
        },
    }
