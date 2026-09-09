"""
FastAPI Router - Monte Carlo Tree Search (MCTS) & Process Reward Model (PRM) Endpoints (Phase 43)
"""

from typing import Any, Dict, List

from fastapi import APIRouter
from pydantic import BaseModel, Field

from packages.models.reasoning.mcts import ReasoningMCTS
from packages.models.reasoning.prm import ProcessRewardModel
from packages.models.reasoning.self_play import ReasoningSelfPlay

router = APIRouter(prefix="/reasoning/mcts", tags=["MCTS Reasoning & PRMs"])
_prm = ProcessRewardModel()
_mcts = ReasoningMCTS(prm=_prm)
_self_play = ReasoningSelfPlay(prm=_prm)


class MCTSSearchRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Reasoning problem or query")
    n_simulations: int = Field(default=15, ge=1, le=50)
    branch_factor: int = Field(default=3, ge=1, le=5)
    max_depth: int = Field(default=5, ge=2, le=8)


class PRMScoreRequest(BaseModel):
    steps: List[str] = Field(
        ..., min_length=1, description="Sequential intermediate reasoning steps"
    )


class SelfPlayRequest(BaseModel):
    count: int = Field(default=3, ge=1, le=10)


@router.post("/search")
def search_mcts(req: MCTSSearchRequest) -> Dict[str, Any]:
    """
    Executes MCTS tree search over candidate reasoning steps guided by PUCT and PRM.
    Returns the serialized tree graph, optimal trajectory, and final answer.
    """
    mcts_engine = ReasoningMCTS(prm=_prm, max_depth=req.max_depth)
    result = mcts_engine.search(
        prompt=req.prompt,
        n_simulations=req.n_simulations,
        branch_factor=req.branch_factor,
    )
    return {
        "prompt": result.prompt,
        "optimal_path": result.optimal_path,
        "final_answer": result.final_answer,
        "total_nodes": result.total_nodes,
        "total_simulations": result.total_simulations,
        "tree_nodes": result.tree_nodes,
        "tree_edges": result.tree_edges,
        "best_value": result.best_value,
    }


@router.post("/prm/score")
def score_steps_with_prm(req: PRMScoreRequest) -> Dict[str, Any]:
    """
    Evaluates step-level validity using the Process Reward Model (PRM).
    Detects critical reasoning flaws and marks the first point of failure.
    """
    eval_res = _prm.score_trace(req.steps)
    return {
        "is_valid_trace": eval_res.is_valid_trace,
        "first_error_index": eval_res.first_error_index,
        "mean_step_score": eval_res.mean_step_score,
        "min_step_score": eval_res.min_step_score,
        "steps": [
            {
                "step_index": s.step_index,
                "content": s.content,
                "score": s.score,
                "is_valid": s.is_valid,
                "rationale": s.rationale,
                "detected_errors": s.detected_errors,
            }
            for s in eval_res.steps
        ],
    }


@router.post("/self_play/generate")
def generate_self_play(req: SelfPlayRequest) -> List[Dict[str, Any]]:
    """
    Generates synthetic self-play reasoning trajectories and produces
    step-level DPO preference pairs (chosen vs rejected).
    """
    return _self_play.generate_batch(count=req.count)


@router.get("/presets")
def get_mcts_presets() -> List[Dict[str, Any]]:
    """Returns educational reasoning benchmark templates."""
    return [
        {
            "id": "game_of_24",
            "title": "Game of 24 (Numbers: 4, 4, 7, 7)",
            "category": "Arithmetic Tree Search",
            "prompt": "Using numbers [4, 4, 7, 7] and basic arithmetic operations (+, -, *, /), make exactly 24.",
            "description": "Requires non-trivial fraction intermediate (7 - 7/7 = 6, then 4 * 6 = 24) to avoid dead ends.",
        },
        {
            "id": "linear_system",
            "title": "Two-Variable Algebraic System",
            "category": "Algebraic Deduction",
            "prompt": "Solve the system: 2x + 3y = 26 and x - y = 3.",
            "description": "Tests step-by-step variable elimination and substitution verification.",
        },
        {
            "id": "triangle_area",
            "title": "Geometric Area Calculation",
            "category": "Geometry & Proof",
            "prompt": "Find the area of a right triangle with legs 8 and 15.",
            "description": "Verifies correct formula invocation (1/2 * b * h) vs omitting the 1/2 factor.",
        },
    ]
