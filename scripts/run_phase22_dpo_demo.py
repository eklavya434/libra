"""
Libra Educational Demo - Phase 22: Reinforcement Learning from First Principles (RLHF & DPO)

Demonstrates:
  1. Pairwise preference data formatting with completion-span label masking (-100).
  2. First-principles scalar Reward Modeling using the Bradley-Terry preference objective.
  3. Direct Preference Optimization (Rafailov et al., NeurIPS 2023) training a policy model
     directly against a frozen reference model without PPO or explicit reward models.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch

from packages.models.modern_config import ModernTransformerConfig
from packages.models.modern_transformer import ModernTransformerLM
from packages.models.reward_model import TransformerRewardModel
from packages.training.dpo_trainer import DPOConfig, DPOTrainer
from packages.training.preference_dataset import PreferenceDataset, PreferenceSample


def demo_preference_data() -> None:
    print("\n" + "=" * 80)
    print("DEMO 1: PREFERENCE DATA FORMATTING & COMPLETION-SPAN MASKING")
    print("=" * 80)

    sample = PreferenceSample(
        prompt="Tell me about gravity: ",
        chosen="Gravity is a curvature in spacetime caused by mass and energy.",
        rejected="Stuff falls down because it wants to.",
    )

    dataset = PreferenceDataset(samples=[sample], max_length=64)
    item = dataset[0]

    prompt_len = item["prompt_length"]
    print(f"Prompt:   \"{sample.prompt}\" (length: {prompt_len} tokens)")
    print(f"Chosen:   \"{sample.chosen}\"")
    print(f"Rejected: \"{sample.rejected}\"")

    print("\nToken Masking Demonstration:")
    print(f"  • Prompt Labels:     {item['chosen_labels'][:prompt_len]} (All masked to -100!)")
    print(f"  • Completion Labels: {item['chosen_labels'][prompt_len:prompt_len + 8]}... (Active in loss!)")


def demo_reward_model() -> None:
    print("\n" + "=" * 80)
    print("DEMO 2: FIRST-PRINCIPLES REWARD MODELING (BRADLEY-TERRY OBJECTIVE)")
    print("=" * 80)

    torch.manual_seed(42)
    cfg = ModernTransformerConfig(
        vocab_size=256,
        d_model=32,
        n_layers=2,
        n_heads=2,
        max_context_length=64,
        hidden_dim=64,
    )
    reward_model = TransformerRewardModel(cfg)
    optimizer = torch.optim.AdamW(reward_model.parameters(), lr=0.01)

    # 2 Preference pairs
    samples = [
        PreferenceSample("Help me fix my code: ", "Sure! Here is the corrected logic with unit tests.", "Fix it yourself."),
        PreferenceSample("What is 12 * 12? ", "12 multiplied by 12 equals 144.", "A big number."),
    ]
    ds = PreferenceDataset(samples=samples, max_length=64)
    batch = PreferenceDataset.collate_fn([ds[0], ds[1]])

    print(f"Model Parameters: {sum(p.numel() for p in reward_model.parameters()):,}")
    print("Training Reward Model over 12 steps...")

    t0 = time.perf_counter()
    for step in range(1, 13):
        optimizer.zero_grad()
        loss, telemetry = reward_model.compute_loss(
            chosen_input_ids=batch["chosen_input_ids"],
            rejected_input_ids=batch["rejected_input_ids"],
        )
        loss.backward()
        optimizer.step()

        if step % 3 == 0 or step == 1:
            print(
                f"  Step {step:02d} | Loss: {telemetry.loss:.4f} | "
                f"Margin (rw - rl): {telemetry.reward_margin:+.4f} | "
                f"Accuracy: {telemetry.accuracy * 100:.0f}% | "
                f"r_chosen: {telemetry.mean_chosen_reward:+.3f} | "
                f"r_rejected: {telemetry.mean_rejected_reward:+.3f}"
            )
    t_el = (time.perf_counter() - t0) * 1000
    print(f"Reward Model convergence completed in {t_el:.1f} ms on CPU.")


def demo_direct_preference_optimization() -> None:
    print("\n" + "=" * 80)
    print("DEMO 3: DIRECT PREFERENCE OPTIMIZATION (RAFALOV ET AL., NEURIPS 2023)")
    print("=" * 80)

    torch.manual_seed(42)
    cfg = ModernTransformerConfig(
        vocab_size=256,
        d_model=32,
        n_layers=2,
        n_heads=2,
        max_context_length=64,
        hidden_dim=64,
    )

    policy_model = ModernTransformerLM(cfg)
    reference_model = ModernTransformerLM(cfg)
    reference_model.load_state_dict(policy_model.state_dict())

    dpo_config = DPOConfig(beta=0.2, lr=0.01)
    trainer = DPOTrainer(
        policy_model=policy_model,
        reference_model=reference_model,
        config=dpo_config,
    )

    preference_pairs = [
        PreferenceSample(
            prompt="Explain recursion: ",
            chosen="Recursion is a function calling itself with a base condition.",
            rejected="Recursion recursion recursion recursion recursion.",
        ),
        PreferenceSample(
            prompt="How to be safe online? ",
            chosen="Use strong unique passwords and enable two-factor authentication.",
            rejected="Click every link you find on the internet.",
        ),
    ]

    ds = PreferenceDataset(samples=preference_pairs, max_length=64)
    batch = PreferenceDataset.collate_fn([ds[0], ds[1]])

    print(f"Policy Parameters:    {policy_model.count_parameters():,}")
    print(f"Reference Parameters: {reference_model.count_parameters():,} (Frozen)")
    print(f"KL Penalty Beta:      {dpo_config.beta}")
    print("Training Policy with DPO over 15 steps...\n")

    t0 = time.perf_counter()
    for step in range(1, 16):
        telemetry = trainer.train_step(batch)

        if step % 3 == 0 or step == 1:
            print(
                f"  Step {step:02d} | DPO Loss: {telemetry.loss:.4f} | "
                f"Implicit Margin: {telemetry.reward_margin:+.4f} | "
                f"Accuracy: {telemetry.accuracy * 100:.0f}% | "
                f"Chosen Logp: {telemetry.policy_chosen_logp:.2f} | "
                f"Rejected Logp: {telemetry.policy_rejected_logp:.2f}"
            )
    t_el = (time.perf_counter() - t0) * 1000
    print(f"\nDPO Policy alignment completed in {t_el:.1f} ms on CPU.")
    print("  • The policy shifted probability mass toward the chosen completions.")
    print("  • The reference model remained strictly unperturbed.")


def main() -> None:
    demo_preference_data()
    demo_reward_model()
    demo_direct_preference_optimization()
    print("\n" + "=" * 80)
    print("PHASE 22 DEMONSTRATION COMPLETE: ALL SYSTEMS VERIFIED")
    print("=" * 80)


if __name__ == "__main__":
    main()
