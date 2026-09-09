"""
Libra Evaluation - Needle-In-A-Haystack (NIAH) Retrieval Benchmark
Evaluates long-context associative recall, memory retrieval, and in-context search.

Algorithm:
1. Synthesizes a distractor background ("Haystack") of selectable token/word length.
2. Injects a target factual "Needle" (e.g. key-value fact or code secret) at a specific relative depth (0.0 to 1.0).
3. Prompts the language model to retrieve the needle.
4. Computes exact/fuzzy retrieval match scores, latency, and evaluates across (Length x Depth) matrix.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

DEFAULT_NEEDLE_FACT = "The secret access code to the vault is 849204."
DEFAULT_NEEDLE_KEY = "849204"
DEFAULT_NEEDLE_PROMPT = "What is the secret access code to the vault?"

HAYSTACK_PARAGRAPHS = [
    "The atmospheric dynamics of gas giants present extraordinary fluid mechanics challenges. High-velocity zonal jets encircle the equator, driven by deep internal thermal convective plumes.",
    "Early microprocessor architectures relied heavily on microcode engines to decode variable-length instructions into micro-operations. Modern designs blend hardwired pipelines with branch predictors.",
    "Cellular mitosis requires precise kinetochore microtubule attachment to sister chromatids during metaphase. Mitotic spindle checkpoints prevent premature anaphase onset.",
    "High-performance transactional databases utilize Write-Ahead Logging (WAL) and Aries recovery algorithms to preserve ACID durability without forcing synchronous random disk writes.",
    "Quantum decoherence occurs when a quantum system interacts irreversibly with its external environment, collapsing superpositions into statistical mixtures of classical states.",
    "Photosynthetic light-harvesting complexes transfer excitation energy to reaction centers with nearly ninety-nine percent quantum efficiency via quantum coherent resonance.",
    "Compiler intermediate representations such as static single assignment (SSA) simplify dead code elimination, constant propagation, and register allocation graph coloring.",
    "The Navier-Stokes equations governing incompressible viscous fluids remain one of the most prominent unsolved Millennium Prize problems in mathematical physics.",
]


@dataclass
class NeedleResult:
    """Telemetry and outcome of a single Needle-In-A-Haystack retrieval trial."""

    context_length: int
    depth_percent: float
    needle: str
    target_key: str
    retrieved_text: str
    is_correct: bool
    score: float
    latency_ms: float
    prompt_tokens: int = 0


class NeedleInHaystackEvaluator:
    """First-principles Needle-In-A-Haystack benchmark engine."""

    def __init__(
        self,
        needle: str = DEFAULT_NEEDLE_FACT,
        target_key: str = DEFAULT_NEEDLE_KEY,
        retrieval_prompt: str = DEFAULT_NEEDLE_PROMPT,
        distractors: list[str] | None = None,
    ) -> None:
        self.needle = needle
        self.target_key = target_key
        self.retrieval_prompt = retrieval_prompt
        self.distractors = distractors or HAYSTACK_PARAGRAPHS

    def construct_haystack(
        self,
        target_words: int,
        depth_fraction: float,
    ) -> str:
        """Constructs a background haystack and places the needle at depth_fraction (0.0 = top, 1.0 = bottom)."""
        depth_fraction = max(0.0, min(1.0, float(depth_fraction)))

        # Build distractor text until reaching target word count
        words: list[str] = []
        idx = 0
        while len(words) < target_words:
            p = self.distractors[idx % len(self.distractors)]
            words.extend(p.split())
            idx += 1

        # Truncate to exact target word count
        words = words[:target_words]

        # Calculate insertion index for needle
        insert_idx = int(len(words) * depth_fraction)

        needle_words = self.needle.split()
        haystack_words = words[:insert_idx] + needle_words + words[insert_idx:]
        return " ".join(haystack_words)

    def format_prompt(self, haystack_text: str) -> str:
        """Formats the full long-context input prompt."""
        return (
            f"Read the following background documents carefully:\n\n"
            f"{haystack_text}\n\n"
            f"Question: {self.retrieval_prompt}\n"
            f"Answer:"
        )

    def evaluate_response(self, response: str) -> tuple[bool, float]:
        """Scores candidate completion against target key."""
        clean_resp = response.lower().strip()
        clean_key = self.target_key.lower().strip()

        if clean_key in clean_resp:
            return True, 1.0

        # Partial token match if key contains multiple terms
        key_tokens = clean_key.split()
        if len(key_tokens) > 1:
            matched = sum(1 for t in key_tokens if t in clean_resp)
            ratio = matched / len(key_tokens)
            return ratio >= 0.5, ratio

        return False, 0.0

    def run_single(
        self,
        generator_fn: Callable[[str], str],
        context_words: int,
        depth_fraction: float,
    ) -> NeedleResult:
        """Runs a single needle retrieval evaluation."""
        haystack = self.construct_haystack(context_words, depth_fraction)
        prompt = self.format_prompt(haystack)

        t0 = time.perf_counter()
        response = generator_fn(prompt)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        is_correct, score = self.evaluate_response(response)

        return NeedleResult(
            context_length=context_words,
            depth_percent=round(depth_fraction * 100.0, 1),
            needle=self.needle,
            target_key=self.target_key,
            retrieved_text=response.strip(),
            is_correct=is_correct,
            score=score,
            latency_ms=round(latency_ms, 2),
            prompt_tokens=len(prompt.split()),
        )

    def run_grid(
        self,
        generator_fn: Callable[[str], str],
        context_lengths: list[int] | None = None,
        depth_fractions: list[float] | None = None,
    ) -> list[NeedleResult]:
        """Runs a 2D evaluation matrix across multiple lengths and depths."""
        lengths = context_lengths or [250, 500, 1000]
        depths = depth_fractions or [0.0, 0.25, 0.5, 0.75, 1.0]

        results: list[NeedleResult] = []
        for length in lengths:
            for depth in depths:
                result = self.run_single(generator_fn, length, depth)
                results.append(result)
        return results
