"""
Libra Evaluation - Multi-Needle-In-A-Haystack (M-NIAH) Retrieval Benchmark

Evaluates multi-key associative recall, relational synthesis, and reasoning across
complex long-context distractors at multiple insertion depths simultaneously.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from packages.evaluation.needle_haystack import HAYSTACK_PARAGRAPHS


@dataclass
class MultiNeedleItem:
    """Individual factual needle specification."""

    key: str
    fact: str
    depth_fraction: float


@dataclass
class MultiNeedleResult:
    """Telemetry and score outcome of a multi-needle retrieval trial."""

    context_length: int
    num_needles: int
    needles: List[Dict[str, Any]]
    retrieved_text: str
    all_correct: bool
    partial_score: float
    found_keys: List[str]
    missing_keys: List[str]
    latency_ms: float
    prompt_tokens: int


class MultiNeedleEvaluator:
    """Engine for multi-needle associative recall and synthesis evaluation."""

    DEFAULT_NEEDLE_SET = [
        MultiNeedleItem(
            key="Alpha-77", fact="The primary launch silo code is Alpha-77.", depth_fraction=0.20
        ),
        MultiNeedleItem(
            key="Falcon", fact="The mission commander callsign is Falcon.", depth_fraction=0.50
        ),
        MultiNeedleItem(
            key="Omega-Zero",
            fact="The emergency abort sequence key is Omega-Zero.",
            depth_fraction=0.85,
        ),
    ]
    DEFAULT_QUESTION = "What are the primary launch silo code, the mission commander callsign, and the emergency abort sequence key?"

    def __init__(self, distractors: Optional[List[str]] = None) -> None:
        self.distractors = distractors or HAYSTACK_PARAGRAPHS

    def construct_haystack(
        self,
        target_words: int,
        needles: List[MultiNeedleItem],
    ) -> str:
        """
        Builds a distractor haystack and places multiple needles at their respective
        sorted depth fractions without overwriting or colliding.
        """
        # Sort needles by depth fraction
        sorted_needles = sorted(needles, key=lambda x: x.depth_fraction)

        # Assemble background words
        words: List[str] = []
        idx = 0
        while len(words) < target_words:
            p = self.distractors[idx % len(self.distractors)]
            words.extend(p.split())
            idx += 1
        words = words[:target_words]

        # Insert needles from highest depth to lowest depth to preserve relative indices
        for needle in reversed(sorted_needles):
            depth = max(0.0, min(1.0, float(needle.depth_fraction)))
            insert_pos = int(len(words) * depth)
            needle_words = needle.fact.split()
            words = words[:insert_pos] + needle_words + words[insert_pos:]

        return " ".join(words)

    def format_prompt(self, haystack_text: str, question: str) -> str:
        """Formats the full context prompt with instruction header."""
        return (
            f"Read the following background briefing documents carefully:\n\n"
            f"{haystack_text}\n\n"
            f"Question: {question}\n"
            f"Provide all requested keys precisely.\n"
            f"Answer:"
        )

    def evaluate_response(
        self,
        response: str,
        expected_keys: List[str],
    ) -> tuple[bool, float, List[str], List[str]]:
        """Scores candidate completion against all required needle keys."""
        clean_resp = response.lower()
        found: List[str] = []
        missing: List[str] = []

        for key in expected_keys:
            if key.lower().strip() in clean_resp:
                found.append(key)
            else:
                missing.append(key)

        score = len(found) / len(expected_keys) if expected_keys else 1.0
        all_correct = len(missing) == 0
        return all_correct, round(score, 3), found, missing

    def run_single(
        self,
        generator_fn: Callable[[str], str],
        context_words: int,
        needles: Optional[List[MultiNeedleItem]] = None,
        question: Optional[str] = None,
    ) -> MultiNeedleResult:
        """Runs a single multi-needle retrieval test."""
        active_needles = needles or self.DEFAULT_NEEDLE_SET
        active_question = question or self.DEFAULT_QUESTION

        haystack = self.construct_haystack(context_words, active_needles)
        prompt = self.format_prompt(haystack, active_question)

        t0 = time.perf_counter()
        response = generator_fn(prompt)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        expected_keys = [n.key for n in active_needles]
        all_correct, score, found, missing = self.evaluate_response(response, expected_keys)

        return MultiNeedleResult(
            context_length=context_words,
            num_needles=len(active_needles),
            needles=[
                {"key": n.key, "fact": n.fact, "depth_percent": round(n.depth_fraction * 100, 1)}
                for n in active_needles
            ],
            retrieved_text=response.strip(),
            all_correct=all_correct,
            partial_score=score,
            found_keys=found,
            missing_keys=missing,
            latency_ms=round(latency_ms, 2),
            prompt_tokens=len(prompt.split()),
        )

    def run_grid(
        self,
        generator_fn: Callable[[str], str],
        context_lengths: Optional[List[int]] = None,
        depth_sets: Optional[List[List[float]]] = None,
    ) -> List[MultiNeedleResult]:
        """Runs a multi-needle test matrix over various lengths and depth configurations."""
        lengths = context_lengths or [300, 600, 1200]
        configurations = depth_sets or [
            [0.15, 0.50, 0.85],
            [0.10, 0.30, 0.60, 0.90],
        ]

        results: List[MultiNeedleResult] = []
        for length in lengths:
            for depths in configurations:
                items = [
                    MultiNeedleItem(
                        key=f"Key-{i + 1}",
                        fact=f"Secret key number {i + 1} is SecCode-{100 + i * 37}.",
                        depth_fraction=d,
                    )
                    for i, d in enumerate(depths)
                ]
                q = f"List all secret key codes from key 1 through key {len(depths)}."
                res = self.run_single(generator_fn, length, items, q)
                results.append(res)

        return results
