"""Unit tests for VerifiableSearchEngine and PRM-guided tree search."""

from packages.models.reasoning.prm import ProcessRewardModel
from packages.models.reasoning.verifiable_search import (
    SearchStepNode,
    VerifiableSearchConfig,
    VerifiableSearchEngine,
)


def test_search_step_node_attributes():
    node = SearchStepNode(
        node_id="test_node_1",
        content="Step 1: 15 * 4 = 60.",
        parent_id=None,
        depth=1,
        step_score=0.95,
        cumulative_score=0.95,
        status="accepted",
        rationale="Valid arithmetic equation",
    )
    d = node.to_dict()
    assert d["id"] == "test_node_1"
    assert d["step_score"] == 0.95
    assert d["status"] == "accepted"


def test_verifiable_search_arithmetic_pruning_and_backtracking():
    cfg = VerifiableSearchConfig(
        beam_width=3,
        max_depth=5,
        step_prune_threshold=0.55,
        branching_factor=2,
    )
    engine = VerifiableSearchEngine(config=cfg)

    # Prompt triggers arithmetic compound problem
    prompt = "Calculate (15 * 4) + (24 / 3) - 17"
    res = engine.search(prompt)

    assert res.is_verified is True
    assert "51" in res.final_answer
    assert res.total_steps_explored > 0
    # PRM should have caught the erroneous candidate steps (e.g. 15*4=55, 24/3=6, 60+8=70)
    assert res.pruned_branches_count >= 1
    assert len(res.optimal_trajectory) >= 3


def test_verifiable_search_custom_generator():
    prm = ProcessRewardModel()
    cfg = VerifiableSearchConfig(beam_width=2, max_depth=3, step_prune_threshold=0.55)

    def mock_generator(prompt: str, history: list[str]) -> list[str]:
        depth = len(history)
        if depth == 0:
            return ["Step 1: 2 + 2 = 4.", "Step 1: 2 + 2 = 5."]
        elif depth == 1:
            return ["Step 2: 4 * 10 = 40. Final Answer: 40."]
        return ["Final Answer: 40."]

    engine = VerifiableSearchEngine(prm=prm, step_generator=mock_generator, config=cfg)
    res = engine.search("Calculate (2 + 2) * 10")

    assert res.is_verified is True
    assert "40" in res.final_answer
    assert res.pruned_branches_count >= 1
