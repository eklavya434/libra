"""
Libra Evaluation - Automated Model Tournament & Benchmark Runner (Phase 28)
Executes pairwise round-robin competitions across multi-domain prompt benchmarks,
referees completions using the AutomatedJudge, and updates the EloLeaderboard.
"""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, Field

from packages.evaluation.arena import ModelComparisonArena
from packages.evaluation.automated_judge import AutomatedJudge, JudgeVerdict
from packages.evaluation.elo import EloLeaderboard, MatchRecord, get_leaderboard_store


class BenchmarkPrompt(BaseModel):
    """Benchmark prompt item with evaluation domain category."""

    id: str = Field(..., description="Unique prompt ID")
    category: str = Field(
        ..., description="Evaluation domain (e.g. coding, reasoning, factual, instruction)"
    )
    prompt: str = Field(..., description="The benchmark prompt text")
    description: str = Field(default="", description="Description of the test objective")


DEFAULT_BENCHMARK_PROMPTS: list[BenchmarkPrompt] = [
    # 1. Coding
    BenchmarkPrompt(
        id="code_prime",
        category="coding",
        prompt="Write a Python function `is_prime(n)` that returns True if n is prime and False otherwise. Include a brief docstring.",
        description="Tests algorithmic correctness, edge case handling, and Python conventions.",
    ),
    BenchmarkPrompt(
        id="code_reverse_str",
        category="coding",
        prompt="Write a Python function `reverse_words(s: str) -> str` that reverses the order of words in a sentence while preserving single spaces.",
        description="Tests string manipulation and boundary conditions.",
    ),
    BenchmarkPrompt(
        id="code_fibonacci",
        category="coding",
        prompt="Write an efficient Python generator function `fibonacci(n)` that yields the first n Fibonacci numbers.",
        description="Tests generator patterns and computational efficiency.",
    ),
    # 2. Reasoning
    BenchmarkPrompt(
        id="reason_bat_ball",
        category="reasoning",
        prompt="A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Explain step by step.",
        description="Classic cognitive reflection test evaluating multi-step algebraic reasoning.",
    ),
    BenchmarkPrompt(
        id="reason_syllogism",
        category="reasoning",
        prompt="All bloops are razzies, and some razzies are lazies. Are all bloops definitely lazies? Explain your logic in 2 sentences.",
        description="Formal deductive logic and syllogistic reasoning.",
    ),
    BenchmarkPrompt(
        id="reason_balance_scale",
        category="reasoning",
        prompt="You have 8 balls of identical appearance, one is slightly heavier. Using a balance scale, how many weighings are needed to identify it with certainty? Explain briefly.",
        description="Information theory and search space partitioning.",
    ),
    # 3. Factual QA
    BenchmarkPrompt(
        id="fact_ribosome",
        category="factual",
        prompt="What is the primary biological function of the ribosome in living cells? Answer in 2 sentences.",
        description="Scientific accuracy and concise domain knowledge retrieval.",
    ),
    BenchmarkPrompt(
        id="fact_supervised_vs_unsupervised",
        category="factual",
        prompt="Explain the difference between supervised learning and unsupervised learning in 2 sentences.",
        description="Core machine learning concepts explanation.",
    ),
    BenchmarkPrompt(
        id="fact_photosynthesis",
        category="factual",
        prompt="What are the essential inputs and primary outputs of photosynthesis in green plants? Answer in 2 sentences.",
        description="Fundamental biological biochemistry summary.",
    ),
    # 4. Instruction Following
    BenchmarkPrompt(
        id="inst_bullet_points",
        category="instruction",
        prompt="List exactly 3 advantages of renewable energy using bullet points. Do not include any introductory or concluding text.",
        description="Strict formatting and length constraint adherence.",
    ),
    BenchmarkPrompt(
        id="inst_no_word",
        category="instruction",
        prompt="Summarize what gradient descent does in exactly 2 sentences without using the word 'slope'.",
        description="Negative constraint adherence and conceptual precision.",
    ),
    BenchmarkPrompt(
        id="inst_json_schema",
        category="instruction",
        prompt="Provide a valid JSON object containing keys 'name', 'version' (string), and 'tags' (array of 3 strings) describing an AI toolkit. Output only JSON.",
        description="Structured data generation and formatting constraints.",
    ),
]


class TournamentReport(BaseModel):
    """Summary of a completed tournament run."""

    tournament_id: str = Field(..., description="Unique tournament ID")
    total_matches: int = Field(default=0, description="Total matches executed")
    categories_evaluated: list[str] = Field(default_factory=list)
    matches: list[MatchRecord] = Field(default_factory=list)
    leaderboard_rankings: list[dict[str, Any]] = Field(default_factory=list)


class TournamentRunner:
    """Orchestrates automated benchmarks, pairwise judging, and Elo rating updates."""

    def __init__(
        self,
        leaderboard: EloLeaderboard | None = None,
        judge: AutomatedJudge | None = None,
        arena: ModelComparisonArena | None = None,
        prompts: list[BenchmarkPrompt] | None = None,
    ) -> None:
        self.leaderboard = leaderboard or get_leaderboard_store()
        self.judge = judge or AutomatedJudge()
        self.arena = arena or ModelComparisonArena()
        self.prompts = prompts or DEFAULT_BENCHMARK_PROMPTS

    async def run_single_match(
        self,
        model_a: str,
        model_b: str,
        prompt_item: BenchmarkPrompt,
        provider_a: str | None = None,
        provider_b: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 256,
    ) -> tuple[MatchRecord, JudgeVerdict, str, str]:
        """Runs a single head-to-head match on a prompt and records the outcome in Elo leaderboard."""
        # Benchmark both models concurrently
        res_a, res_b = await asyncio.gather(
            self.arena.benchmark_single_model(
                model=model_a,
                prompt=prompt_item.prompt,
                provider_name=provider_a,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            self.arena.benchmark_single_model(
                model=model_b,
                prompt=prompt_item.prompt,
                provider_name=provider_b,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
        )

        completion_a = res_a.output_text if res_a.success else f"⚠️ Error: {res_a.error}"
        completion_b = res_b.output_text if res_b.success else f"⚠️ Error: {res_b.error}"

        # Evaluate completions using position-bias mitigated referee
        verdict = self.judge.evaluate_pair(
            prompt=prompt_item.prompt,
            completion_a=completion_a,
            completion_b=completion_b,
            model_a_id=model_a,
            model_b_id=model_b,
        )

        # Record match in Elo leaderboard
        match_rec = self.leaderboard.record_match(
            model_a=model_a,
            model_b=model_b,
            winner=verdict.winner,
            category=prompt_item.category,
            prompt=prompt_item.prompt,
        )

        return match_rec, verdict, completion_a, completion_b

    async def run_tournament(
        self,
        models: list[dict[str, Any]],
        categories: list[str] | None = None,
        prompts_per_category: int = 1,
        temperature: float = 0.7,
        max_tokens: int = 256,
    ) -> TournamentReport:
        """Executes a round-robin tournament between models across specified benchmark categories."""
        if len(models) < 2:
            raise ValueError("Tournament requires at least 2 competing models.")

        selected_categories = (
            [c.lower() for c in categories]
            if categories
            else ["coding", "reasoning", "factual", "instruction"]
        )

        # Filter benchmark prompts
        target_prompts: list[BenchmarkPrompt] = []
        for cat in selected_categories:
            cat_prompts = [p for p in self.prompts if p.category.lower() == cat]
            target_prompts.extend(cat_prompts[:prompts_per_category])

        if not target_prompts:
            target_prompts = self.prompts[: max(1, prompts_per_category)]

        executed_matches: list[MatchRecord] = []
        tournament_id = f"tourn_{int(asyncio.get_event_loop().time() * 1000)}"

        # Round-robin pairings
        n = len(models)
        for i in range(n):
            for j in range(i + 1, n):
                model_a_info = models[i]
                model_b_info = models[j]

                m_a = (
                    model_a_info.get("model")
                    if isinstance(model_a_info, dict)
                    else str(model_a_info)
                )
                p_a = model_a_info.get("provider") if isinstance(model_a_info, dict) else None

                m_b = (
                    model_b_info.get("model")
                    if isinstance(model_b_info, dict)
                    else str(model_b_info)
                )
                p_b = model_b_info.get("provider") if isinstance(model_b_info, dict) else None

                for prompt_item in target_prompts:
                    match_rec, _, _, _ = await self.run_single_match(
                        model_a=m_a,
                        model_b=m_b,
                        prompt_item=prompt_item,
                        provider_a=p_a,
                        provider_b=p_b,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    executed_matches.append(match_rec)

        leaderboard_data = [
            m.model_dump() for m in self.leaderboard.get_leaderboard(category="overall")
        ]

        return TournamentReport(
            tournament_id=tournament_id,
            total_matches=len(executed_matches),
            categories_evaluated=selected_categories,
            matches=executed_matches,
            leaderboard_rankings=leaderboard_data,
        )
