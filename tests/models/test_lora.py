"""
Tests for Low-Rank Adaptation (LoRA) and PEFT
Phase 25: Parameter-Efficient Fine-Tuning (PEFT & LoRA)
"""

from pathlib import Path

import torch
from torch import nn

from packages.models.lora.lora_linear import LoRALinear
from packages.models.lora.lora_model import (
    apply_lora,
    get_lora_parameter_summary,
    load_lora_adapter,
    save_lora_adapter,
)
from packages.models.modern_config import ModernTransformerConfig
from packages.models.modern_transformer import ModernTransformerLM
from packages.training.lora_trainer import LoRATrainer


def test_lora_linear_initial_identity():
    """Verify B=0 initialization ensures exact initial numerical equivalence."""
    torch.manual_seed(42)
    base_linear = nn.Linear(32, 64, bias=True)
    lora_linear = LoRALinear(base_linear, rank=4, alpha=8.0)

    x = torch.randn(2, 5, 32)
    with torch.no_grad():
        out_base = base_linear(x)
        out_lora = lora_linear(x)

    assert torch.equal(out_base, out_lora), (
        "LoRA must match base linear output exactly at initialization"
    )


def test_lora_linear_parameter_freezing():
    """Verify base weight is frozen and only lora_A and lora_B require gradients."""
    base_linear = nn.Linear(16, 32)
    lora_linear = LoRALinear(base_linear, rank=4)

    assert lora_linear.weight.requires_grad is False
    assert lora_linear.lora_A.requires_grad is True
    assert lora_linear.lora_B.requires_grad is True


def test_lora_linear_merge_and_unmerge():
    """Verify folding adapter into base weight produces identical outputs."""
    torch.manual_seed(42)
    base_linear = nn.Linear(16, 32, bias=True)
    lora_layer = LoRALinear(base_linear, rank=4, alpha=8.0)

    # Simulate some training on A and B
    nn.init.normal_(lora_layer.lora_A)
    nn.init.normal_(lora_layer.lora_B)

    x = torch.randn(3, 16)
    out_unmerged = lora_layer(x)

    # Merge
    lora_layer.merge_weights()
    assert lora_layer.merged is True
    out_merged = lora_layer(x)

    diff = (out_unmerged - out_merged).abs().max().item()
    assert diff < 1e-5, f"Merged output discrepancy too large: {diff}"

    # Unmerge
    lora_layer.unmerge_weights()
    assert lora_layer.merged is False
    out_restored = lora_layer(x)
    assert torch.allclose(out_unmerged, out_restored, atol=1e-5)


def test_apply_lora_to_transformer():
    """Verify applying LoRA freezes backbone and replaces target layers."""
    cfg = ModernTransformerConfig(vocab_size=50, d_model=32, n_heads=4, n_layers=2)
    model = ModernTransformerLM(cfg)

    # Target attention Q and V projections
    apply_lora(model, rank=4, alpha=8.0, target_modules=["q_proj", "v_proj"])

    summary = get_lora_parameter_summary(model)
    assert summary["trainable_parameters"] > 0
    assert (
        summary["trainable_parameters"] < summary["total_parameters"] * 0.15
    )  # < 15% in tiny test model
    assert summary["frozen_parameters"] > 0

    # Ensure all token embeddings and norm layers are frozen
    assert model.tok_emb.weight.requires_grad is False
    assert model.output_head.weight.requires_grad is False


def test_lora_trainer_step():
    """Verify training loop updates only low-rank matrices and reduces loss."""
    torch.manual_seed(42)
    cfg = ModernTransformerConfig(vocab_size=50, d_model=32, n_heads=4, n_layers=2)
    model = ModernTransformerLM(cfg)
    apply_lora(model, rank=4, alpha=8.0)

    trainer = LoRATrainer(model, learning_rate=1e-2)

    x = torch.randint(0, 50, (2, 8))
    y = torch.randint(0, 50, (2, 8))

    res1 = trainer.train_step(x, y)
    res2 = trainer.train_step(x, y)

    assert "loss" in res1 and "loss" in res2
    assert res1["loss"] > 0.0 and res2["loss"] > 0.0

    # Verify optimizer step updated adapter
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)


def test_lora_save_and_load(tmp_path: Path):
    """Verify saving and reloading lightweight adapter weights."""
    torch.manual_seed(42)
    cfg = ModernTransformerConfig(vocab_size=50, d_model=32, n_heads=4, n_layers=2)
    model = ModernTransformerLM(cfg)
    apply_lora(model, rank=4, alpha=8.0)

    # Perturb adapter weights
    for m in model.modules():
        if isinstance(m, LoRALinear):
            nn.init.normal_(m.lora_A)
            nn.init.normal_(m.lora_B)

    sample = torch.randint(0, 50, (1, 6))
    with torch.no_grad():
        orig_out, _ = model(sample)

    save_file = tmp_path / "lora_adapter.pt"
    save_lora_adapter(model, save_file, metadata={"test": "peft"})
    assert save_file.exists()

    # Fresh model initialized with identical base weights
    torch.manual_seed(42)
    new_model = ModernTransformerLM(cfg)
    apply_lora(new_model, rank=4, alpha=8.0)

    load_lora_adapter(new_model, save_file)
    with torch.no_grad():
        new_out, _ = new_model(sample)

    assert torch.allclose(orig_out, new_out, atol=1e-5)
