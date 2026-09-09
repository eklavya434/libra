"""Unit tests for KTOTrainer and Kahneman-Tversky loss calculation."""

import torch

from packages.models.modern_config import ModernTransformerConfig
from packages.models.modern_transformer import ModernTransformerLM
from packages.training.kto_dataset import KTODataset, KTOSample
from packages.training.kto_trainer import KTOConfig, KTOTrainer


def _create_toy_models():
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


def test_kto_completion_logps():
    batch_size = 2
    seq_len = 8
    vocab_size = 32

    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.tensor(
        [
            [-100, -100, 5, 10, 15, -100, -100, -100],
            [-100, 2, 4, 6, 8, 10, -100, -100],
        ]
    )

    logps = KTOTrainer.get_completion_logps(logits, labels, length_normalized=True)
    assert logps.shape == (batch_size,)
    assert not torch.isnan(logps).any()
    assert (logps < 0).all()  # Log probs must be negative


def test_kto_loss_loss_aversion_asymmetry():
    policy, ref = _create_toy_models()
    config = KTOConfig(beta=0.1, desirable_weight=1.0, undesirable_weight=2.0)
    trainer = KTOTrainer(policy_model=policy, reference_model=ref, config=config)

    rewards = torch.tensor([0.5, -0.5])
    is_desirable = torch.tensor([True, False])

    total_loss, des_loss, und_loss, z_ref = trainer.compute_kto_loss(rewards, is_desirable)

    assert total_loss.item() > 0
    assert config.undesirable_weight > config.desirable_weight


def test_kto_train_step_updates_policy():
    policy, ref = _create_toy_models()
    config = KTOConfig(beta=0.1, desirable_weight=1.0, undesirable_weight=1.5, lr=1e-3)
    trainer = KTOTrainer(policy_model=policy, reference_model=ref, config=config)

    samples = [
        KTOSample(prompt="Helpful prompt", completion=" good answer", is_desirable=True),
        KTOSample(prompt="Harmful prompt", completion=" toxic answer", is_desirable=False),
    ]
    ds = KTODataset(samples, max_length=32)
    batch = KTODataset.collate_fn([ds[0], ds[1]])

    initial_param = next(policy.parameters()).clone()
    telemetry = trainer.train_step(batch)

    assert telemetry.loss >= 0.0
    assert telemetry.loss_aversion_ratio == 1.5
    updated_param = next(policy.parameters())
    assert not torch.equal(initial_param, updated_param)

    for p in ref.parameters():
        assert p.requires_grad is False
