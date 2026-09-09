"""
Libra Training - Online On-Policy Direct Preference Optimization (Online DPO)

Closes the exploration loop by generating completions on-policy from the current
policy pi_theta, evaluating them with a reward oracle / scoring function,
forming dynamic preference pairs, and applying DPO gradient updates.
Eliminates the off-policy distribution shift of standard offline DPO.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field
from torch import nn

from packages.training.dpo_trainer import DPOConfig, DPOTelemetry, DPOTrainer
from packages.training.preference_dataset import PreferenceDataset, PreferenceSample


class OnlineDPOTelemetry(BaseModel):
    """Telemetry metrics emitted during an Online DPO on-policy training step."""

    loss: float
    accuracy: float
    chosen_implicit_reward: float
    rejected_implicit_reward: float
    reward_margin: float = Field(..., description="Chosen reward minus rejected reward")
    prompt: str
    chosen_text: str
    rejected_text: str
    chosen_oracle_score: float
    rejected_oracle_score: float
    mean_completion_tokens: float


@dataclass
class OnlineDPOConfig:
    """Hyperparameters for Online DPO."""

    beta: float = 0.1
    lr: float = 5e-5
    temperature: float = 0.8
    max_new_tokens: int = 24
    num_candidates: int = 2
    max_length: int = 128


class OnlineDPOTrainer:
    """Orchestrates on-policy candidate generation, scoring, and online DPO updates."""

    def __init__(
        self,
        policy_model: nn.Module,
        reference_model: nn.Module,
        config: OnlineDPOConfig | None = None,
        reward_fn: Callable[[str, str], float] | None = None,
    ) -> None:
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.config = config or OnlineDPOConfig()
        self.reward_fn = reward_fn

        dpo_cfg = DPOConfig(beta=self.config.beta, lr=self.config.lr)
        self.dpo_trainer = DPOTrainer(
            policy_model=self.policy_model,
            reference_model=self.reference_model,
            config=dpo_cfg,
        )

    def generate_candidate(
        self,
        prompt: str,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> str:
        """Generates a candidate completion on-policy from the current policy model."""
        self.policy_model.eval()
        prompt_tokens = list(prompt.encode("utf-8"))
        if not prompt_tokens:
            prompt_tokens = [0]

        device = next(self.policy_model.parameters()).device
        curr_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

        generated: list[int] = []
        with torch.no_grad():
            for _ in range(self.config.max_new_tokens):
                out = self.policy_model(curr_ids)
                logits = out[0] if isinstance(out, tuple) else out
                next_logits = logits[:, -1, :] / max(1e-4, temperature)

                if top_k > 0 and top_k < next_logits.size(-1):
                    v, _ = torch.topk(next_logits, top_k)
                    next_logits[next_logits < v[:, [-1]]] = -float("Inf")

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                if next_token == 0:
                    break
                generated.append(next_token)
                curr_ids = torch.cat(
                    [curr_ids, torch.tensor([[next_token]], dtype=torch.long, device=device)], dim=1
                )
                if curr_ids.size(1) >= self.config.max_length:
                    break

        return bytes(generated).decode("utf-8", errors="replace")

    def step(
        self,
        prompts: list[str],
        scorer_fn: Callable[[str, str], float] | None = None,
    ) -> list[OnlineDPOTelemetry]:
        """
        Executes an on-policy exploration and DPO optimization step.
        """
        eval_scorer = scorer_fn or self.reward_fn
        if eval_scorer is None:
            raise ValueError("A scoring function or reward_fn must be provided for Online DPO.")

        pairs: list[PreferenceSample] = []
        telemetry_items: list[OnlineDPOTelemetry] = []
        candidate_metadata = []

        for prompt in prompts:
            candidates: list[str] = []
            scores: list[float] = []

            for _ in range(max(2, self.config.num_candidates)):
                cand = self.generate_candidate(prompt, temperature=self.config.temperature)
                if not cand:
                    cand = f"token_{len(candidates)}"
                score = eval_scorer(prompt, cand)
                candidates.append(cand)
                scores.append(score)

            ranked = sorted(zip(candidates, scores, strict=False), key=lambda x: x[1], reverse=True)
            chosen_text, chosen_score = ranked[0]
            rejected_text, rejected_score = ranked[-1]

            if chosen_score == rejected_score:
                chosen_score += 0.01

            pairs.append(
                PreferenceSample(prompt=prompt, chosen=chosen_text, rejected=rejected_text)
            )
            candidate_metadata.append(
                (prompt, chosen_text, rejected_text, chosen_score, rejected_score)
            )

        dataset = PreferenceDataset(samples=pairs, max_length=self.config.max_length)
        batch_raw = [dataset[i] for i in range(len(dataset))]
        batch = PreferenceDataset.collate_fn(batch_raw)

        dpo_telemetry: DPOTelemetry = self.dpo_trainer.train_step(batch)

        for prompt, chosen, rejected, c_score, r_score in candidate_metadata:
            mean_len = (len(chosen.encode("utf-8")) + len(rejected.encode("utf-8"))) / 2.0
            telemetry_items.append(
                OnlineDPOTelemetry(
                    loss=dpo_telemetry.loss,
                    accuracy=dpo_telemetry.accuracy,
                    chosen_implicit_reward=dpo_telemetry.chosen_implicit_reward,
                    rejected_implicit_reward=dpo_telemetry.rejected_implicit_reward,
                    reward_margin=dpo_telemetry.reward_margin,
                    prompt=prompt,
                    chosen_text=chosen,
                    rejected_text=rejected,
                    chosen_oracle_score=round(c_score, 4),
                    rejected_oracle_score=round(r_score, 4),
                    mean_completion_tokens=round(mean_len, 1),
                )
            )

        return telemetry_items
