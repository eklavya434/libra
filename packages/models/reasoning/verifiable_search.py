"""
Libra Models - Step-Level Verifiable Search & PRM-Guided Reasoning (Phase 48)

Implements Step-Level Verifiable Search (SV-Search):
Explores reasoning trajectories step-by-step using a Process Reward Model (PRM)
to evaluate intermediate correctness. Immediately prunes branches with detected
arithmetic or logical errors and automatically backtracks to the most promising
alternative prefix on the search frontier.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from packages.models.reasoning.prm import ProcessRewardModel, StepScore


@dataclass
class SearchStepNode:
    """Represents an intermediate reasoning step node in the verifiable search tree."""

    node_id: str
    content: str
    parent_id: str | None = None
    depth: int = 0
    step_score: float = 0.5
    cumulative_score: float = 0.5
    status: str = "active"  # "active", "accepted", "pruned", "terminal"
    rationale: str = ""
    errors: list[str] = field(default_factory=list)
    children_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serializes node attributes for tree visualization and telemetry."""
        return {
            "id": self.node_id,
            "parent_id": self.parent_id,
            "content": self.content,
            "depth": self.depth,
            "step_score": round(self.step_score, 3),
            "cumulative_score": round(self.cumulative_score, 3),
            "status": self.status,
            "rationale": self.rationale,
            "errors": self.errors,
            "num_children": len(self.children_ids),
        }


@dataclass
class VerifiableSearchConfig:
    """Hyperparameters for verifiable step search."""

    beam_width: int = 3
    max_depth: int = 5
    step_prune_threshold: float = 0.55
    branching_factor: int = 2
    backtrack_limit: int = 15


@dataclass
class VerifiableSearchResult:
    """Outcome of a step-level verifiable reasoning search."""

    prompt: str
    optimal_trajectory: list[dict[str, Any]]
    final_answer: str
    is_verified: bool
    total_steps_explored: int
    pruned_branches_count: int
    backtracks_count: int
    compute_savings_pct: float
    tree_nodes: list[dict[str, Any]]
    tree_edges: list[dict[str, str]]


class VerifiableSearchEngine:
    """
    Orchestrates step-by-step reasoning with Process Reward Model verification,
    early branch pruning, and automatic frontier backtracking.
    """

    def __init__(
        self,
        prm: ProcessRewardModel | None = None,
        step_generator: Callable[[str, list[str]], list[str]] | None = None,
        config: VerifiableSearchConfig | None = None,
    ) -> None:
        self.prm = prm or ProcessRewardModel()
        self.step_generator = step_generator or self._default_educational_step_generator
        self.config = config or VerifiableSearchConfig()

    @staticmethod
    def _default_educational_step_generator(prompt: str, history: list[str]) -> list[str]:
        """
        Deterministic educational reasoning step generator for multi-step tasks.
        Provides both valid deductive steps and perturbed invalid candidate branches
        to test PRM error detection and pruning.
        """
        depth = len(history)
        clean_p = prompt.strip().lower()

        # Preset 1: Arithmetic compound problem (e.g. "Calculate (15 * 4) + (24 / 3) - 17")
        if "15" in clean_p and "4" in clean_p:
            if depth == 0:
                return [
                    "Step 1: Compute the first multiplication term: 15 * 4 = 60.",
                    "Step 1: Compute the first term incorrectly: 15 * 4 = 55.",
                ]
            elif depth == 1:
                return [
                    "Step 2: Compute the division term: 24 / 3 = 8.",
                    "Step 2: Division error: 24 / 3 = 6.",
                ]
            elif depth == 2:
                return [
                    "Step 3: Add the two computed terms: 60 + 8 = 68.",
                    "Step 3: Incorrect addition: 60 + 8 = 70.",
                ]
            elif depth == 3:
                return [
                    "Step 4: Subtract the final term: 68 - 17 = 51. Final Answer: 51.",
                    "Step 4: Subtraction error: 68 - 17 = 49. Final Answer: 49.",
                ]

        # Preset 2: Word problem (e.g. "Sarah has 12 apples, buys 8 more, eats 3, divides rest among 2 friends")
        elif "apple" in clean_p or "sarah" in clean_p:
            if depth == 0:
                return [
                    "Step 1: Start with initial apples: Sarah has 12 apples.",
                ]
            elif depth == 1:
                return [
                    "Step 2: Add newly bought apples: 12 + 8 = 20 apples total.",
                    "Step 2: Arithmetic error: 12 + 8 = 18 apples.",
                ]
            elif depth == 2:
                return [
                    "Step 3: Subtract eaten apples: 20 - 3 = 17 apples remaining.",
                    "Step 3: Error: 20 - 3 = 15 apples remaining.",
                ]
            elif depth == 3:
                return [
                    "Step 4: Divide 17 apples among 2 friends: 17 / 2 = 8.5 apples each. Final Answer: 8.5.",
                ]

        # Generic multi-step solver fallback
        if depth == 0:
            return [
                f"Step 1: Analyze the question parameters for '{prompt}'.",
                "Step 1: This contradicts the premise which is impossible.",
            ]
        elif depth == 1:
            return [
                "Step 2: Deconstruct intermediate logic and apply formal rules.",
                "Step 2: Contradictory division by zero encountered.",
            ]
        elif depth == 2:
            return [
                "Step 3: Synthesize deduction. Therefore, the conclusion holds. Final Answer: Valid.",
            ]

        return [f"Final Answer: Completed trajectory for {prompt}."]

    def search(
        self,
        prompt: str,
        custom_generator: Callable[[str, list[str]], list[str]] | None = None,
    ) -> VerifiableSearchResult:
        """
        Executes step-level verifiable search with early pruning and backtracking.
        """
        generator = custom_generator or self.step_generator
        root_id = "root_" + str(uuid.uuid4())[:6]

        root = SearchStepNode(
            node_id=root_id,
            content=f"Prompt: {prompt}",
            parent_id=None,
            depth=0,
            step_score=1.0,
            cumulative_score=1.0,
            status="accepted",
            rationale="Initial Problem Statement",
        )

        all_nodes: dict[str, SearchStepNode] = {root_id: root}
        edges: list[dict[str, str]] = []

        # Active search frontier: nodes that are valid and can be expanded
        # Stored as list of node_ids, sorted by cumulative_score descending
        frontier: list[str] = [root_id]
        completed_nodes: list[SearchStepNode] = []

        total_steps_explored = 0
        pruned_count = 0
        backtracks_count = 0

        while frontier and len(completed_nodes) < self.config.beam_width:
            # Pop the highest-scoring candidate prefix from frontier
            curr_id = frontier.pop(0)
            curr_node = all_nodes[curr_id]

            if curr_node.depth >= self.config.max_depth:
                curr_node.status = "terminal"
                completed_nodes.append(curr_node)
                continue

            # Build trajectory history for step generation
            history: list[str] = []
            trace_ptr: SearchStepNode | None = curr_node
            while trace_ptr and trace_ptr.parent_id:
                history.append(trace_ptr.content)
                trace_ptr = all_nodes.get(trace_ptr.parent_id)
            history.reverse()

            # Generate candidate next steps
            candidates = generator(prompt, history)
            candidates = candidates[: self.config.branching_factor]

            branch_had_valid_child = False

            for cand_text in candidates:
                total_steps_explored += 1
                child_id = f"step_{curr_node.depth + 1}_" + str(uuid.uuid4())[:6]

                # Verify intermediate step with PRM
                score_obj: StepScore = self.prm.score_step(
                    step_content=cand_text,
                    context_history=history,
                )

                # Compute cumulative confidence (geometric mean of steps)
                step_val = max(0.01, score_obj.score)
                prev_cum = curr_node.cumulative_score
                new_cum = (prev_cum * step_val) ** 0.5  # Soft cumulative product

                is_pruned = (
                    score_obj.score < self.config.step_prune_threshold or not score_obj.is_valid
                )

                child_status = "pruned" if is_pruned else "accepted"

                child_node = SearchStepNode(
                    node_id=child_id,
                    content=cand_text,
                    parent_id=curr_id,
                    depth=curr_node.depth + 1,
                    step_score=score_obj.score,
                    cumulative_score=new_cum,
                    status=child_status,
                    rationale=score_obj.rationale,
                    errors=score_obj.detected_errors,
                )

                all_nodes[child_id] = child_node
                curr_node.children_ids.append(child_id)
                edges.append({"from": curr_id, "to": child_id})

                if is_pruned:
                    pruned_count += 1
                else:
                    branch_had_valid_child = True
                    # Check if terminal step
                    if "final answer" in cand_text.lower():
                        child_node.status = "terminal"
                        completed_nodes.append(child_node)
                    else:
                        frontier.append(child_id)

            # If all candidates at this branch were pruned, backtracking occurred!
            if not branch_had_valid_child and candidates:
                backtracks_count += 1

            # Re-sort frontier by cumulative_score descending (best-first beam search)
            frontier.sort(key=lambda nid: all_nodes[nid].cumulative_score, reverse=True)

        # Select best completed path or best frontier leaf
        if completed_nodes:
            best_leaf = max(completed_nodes, key=lambda n: n.cumulative_score)
        else:
            # Fallback to deepest accepted node
            best_leaf = max(
                all_nodes.values(),
                key=lambda n: (n.status == "accepted", n.depth, n.cumulative_score),
            )

        # Reconstruct optimal trajectory
        optimal_trajectory: list[dict[str, Any]] = []
        ptr: SearchStepNode | None = best_leaf
        while ptr and ptr.parent_id:  # Exclude root prompt
            optimal_trajectory.append(ptr.to_dict())
            ptr = all_nodes.get(ptr.parent_id)
        optimal_trajectory.reverse()

        # Extract final answer
        final_answer = "No verified solution found."
        for step in reversed(optimal_trajectory):
            content = step["content"]
            if "final answer:" in content.lower():
                final_answer = content.split("final answer:")[-1].strip()
                break
            elif "answer is" in content.lower():
                final_answer = content.split("answer is")[-1].strip()
                break
            elif step == optimal_trajectory[-1]:
                final_answer = content

        # Compute savings % (steps avoided due to early pruning)
        max_possible_steps = self.config.branching_factor ** (self.config.max_depth + 1) - 1
        compute_savings_pct = round(
            (pruned_count / max(1, total_steps_explored + pruned_count)) * 100.0, 1
        )

        return VerifiableSearchResult(
            prompt=prompt,
            optimal_trajectory=optimal_trajectory,
            final_answer=final_answer,
            is_verified=best_leaf.step_score >= self.config.step_prune_threshold,
            total_steps_explored=total_steps_explored,
            pruned_branches_count=pruned_count,
            backtracks_count=backtracks_count,
            compute_savings_pct=compute_savings_pct,
            tree_nodes=[n.to_dict() for n in all_nodes.values()],
            tree_edges=edges,
        )
