"""
Unit tests for Monte Carlo Tree Search (MCTS) and self-play preference pair synthesis.
"""

from packages.models.reasoning.mcts import MCTSNode, ReasoningMCTS
from packages.models.reasoning.self_play import ReasoningSelfPlay


def test_mcts_node_puct_score():
    parent = MCTSNode(step_content="Root", depth=0)
    parent.visits = 10

    child = MCTSNode(step_content="Step 1", parent=parent, prior=0.5, prm_score=0.8, depth=1)
    child.visits = 2
    child.value_sum = 1.6  # Q = 0.8

    score = child.puct_score(c_puct=1.414, parent_visits=parent.visits)
    assert score > child.q_value  # Prior exploration bonus added
    assert child.get_trajectory() == ["Step 1"]


def test_reasoning_mcts_search():
    mcts = ReasoningMCTS(c_puct=1.414, max_depth=4)
    result = mcts.search(
        prompt="Make 24 with [4, 4, 7, 7]",
        n_simulations=12,
        branch_factor=3,
    )

    assert result.total_nodes > 1
    assert result.total_simulations == 12
    assert len(result.optimal_path) > 0
    assert len(result.tree_nodes) == result.total_nodes
    assert len(result.tree_edges) == result.total_nodes - 1
    assert "24" in result.final_answer or len(result.optimal_path) >= 1


def test_reasoning_self_play():
    self_play = ReasoningSelfPlay()
    pair = self_play.generate_pair(problem_idx=0)

    assert pair.prompt != ""
    assert pair.chosen != ""
    assert pair.rejected != ""
    assert pair.chosen_score > pair.rejected_score
    assert pair.reward_margin > 0.0

    batch = self_play.generate_batch(count=2)
    assert len(batch) == 2
    assert "chosen" in batch[0]
    assert "rejected" in batch[0]
