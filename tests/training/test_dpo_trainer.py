"""
Tests for Direct Preference Optimization (DPO) Trainer (Phase 22)
"""

import torch

from packages.models.modern_config import ModernTransformerConfig
from packages.models.modern_transformer import ModernTransformerLM
from packages.training.dpo_trainer import DPOConfig, DPOTrainer
from packages.training.preference_dataset import PreferenceDataset, PreferenceSample


def create_models() -> tuple[ModernTransformerLM, ModernTransformerLM]:
    torch.manual_seed(42)
    cfg = ModernTransformerConfig(
        vocab_size=256,
        d_model=32,
        n_layers=2,
        n_heads=2,
        max_context_length=64,
        hidden_dim=64,
    )
    policy = ModernTransformerLM(cfg)
    ref = ModernTransformerLM(cfg)
    ref.load_state_dict(policy.state_dict())
    return policy, ref


def test_reference_model_frozen():
    policy, ref = create_models()
    _ = DPOTrainer(policy_model=policy, reference_model=ref)

    assert not ref.training
    for param in ref.parameters():
        assert not param.requires_grad


def test_batch_logps():
    batch_size = 2
    seq_len = 5
    vocab_size = 10

    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.tensor([[0, 1, 2, 3, 4], [-100, -100, 1, 2, -100]])

    logps = DPOTrainer.get_batch_logps(logits, labels)

    assert isinstance(logps, torch.Tensor)
    assert logps.shape == (batch_size,)
    assert (logps <= 0.0).all()


def test_dpo_train_step_decreases_loss():
    policy, ref = create_models()
    config = DPOConfig(beta=0.5, lr=0.01)
    trainer = DPOTrainer(policy_model=policy, reference_model=ref, config=config)

    samples = [
        PreferenceSample(
            prompt="Hello",
            chosen="world! I am ready.",
            rejected="goodbye bad answer.",
        )
    ]
    ds = PreferenceDataset(samples=samples, max_length=32)
    batch = PreferenceDataset.collate_fn([ds[0]])

    initial_telemetry = trainer.train_step(batch)

    # Perform multiple DPO gradient descent steps
    for _ in range(10):
        final_telemetry = trainer.train_step(batch)

    # DPO optimization should increase reward margin
    assert final_telemetry.reward_margin >= initial_telemetry.reward_margin
    assert final_telemetry.accuracy == 1.0
