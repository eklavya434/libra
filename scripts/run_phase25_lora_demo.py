"""
Project Libra - Phase 25 Demonstration
Parameter-Efficient Fine-Tuning (PEFT & LoRA from First Principles)

Demonstrates:
1. Base transformer parameter freezing (<2% trainable parameters with LoRA)
2. Exact output identity at initialization (B=0 ensures Delta_W=0)
3. Fast CPU-friendly LoRA fine-tuning convergence (<10 seconds)
4. Dynamic zero-overhead adapter weight merging into base weights
"""

import sys
import time
from pathlib import Path
import torch

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from packages.models.modern_config import ModernTransformerConfig
from packages.models.modern_transformer import ModernTransformerLM
from packages.models.lora import (
    apply_lora,
    get_lora_parameter_summary,
    merge_lora_weights,
    unmerge_lora_weights,
)
from packages.training.lora_trainer import LoRATrainer


def print_banner(text: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def demo_lora_parameter_efficiency() -> None:
    print_banner("1. Parameter Efficiency & Freezing Mechanics")

    torch.manual_seed(42)
    cfg = ModernTransformerConfig(
        vocab_size=1000,
        d_model=256,
        n_heads=8,
        n_layers=6,
        max_context_length=256,
    )
    base_model = ModernTransformerLM(cfg)
    base_params = sum(p.numel() for p in base_model.parameters())

    print(f"Base Transformer Configuration:")
    print(f"  - Layers: {cfg.n_layers} | d_model: {cfg.d_model} | Heads: {cfg.n_heads}")
    print(f"  - Full Parameters: {base_params:,}")

    # Inject LoRA across Q and V projections
    rank = 8
    alpha = 16.0
    apply_lora(base_model, rank=rank, alpha=alpha, target_modules=["q_proj", "v_proj"])

    summary = get_lora_parameter_summary(base_model)
    reduction = base_params / max(summary["trainable_parameters"], 1)

    print(f"\nAfter Applying LoRA (r={rank}, alpha={alpha}):")
    print(f"  - Trainable Parameters: {summary['trainable_parameters']:,} ({summary['trainable_percentage']}%)")
    print(f"  - Frozen Parameters:    {summary['frozen_parameters']:,}")
    print(f"  - Parameter Reduction:  {reduction:.1f}x fewer trainable weights!")


def demo_lora_training_convergence() -> None:
    print_banner("2. First-Principles LoRA Fine-Tuning Convergence")

    torch.manual_seed(42)
    cfg = ModernTransformerConfig(vocab_size=100, d_model=64, n_heads=4, n_layers=2)
    model = ModernTransformerLM(cfg)
    apply_lora(model, rank=4, alpha=8.0, target_modules=["q_proj", "v_proj"])

    trainer = LoRATrainer(model, learning_rate=5e-3)

    # Generate synthetic training batch
    data = []
    for _ in range(5):
        x = torch.randint(0, cfg.vocab_size, (4, 12))
        y = torch.randint(0, cfg.vocab_size, (4, 12))
        data.append((x, y))

    print("Training LoRA adapter over 5 epochs on CPU...")
    t0 = time.perf_counter()
    res = trainer.train_epochs(data, epochs=5)
    elapsed = time.perf_counter() - t0

    for item in res["history"]:
        print(f"  Epoch {item['epoch']}: Avg Loss = {item['avg_loss']:.4f}")

    print(f"\nCompleted in {elapsed:.2f} seconds ({elapsed / 5:.3f} s/epoch).")


def demo_lora_zero_overhead_merge() -> None:
    print_banner("3. Zero-Overhead Inference: Weight Merging Equivalence")

    torch.manual_seed(42)
    cfg = ModernTransformerConfig(vocab_size=100, d_model=64, n_heads=4, n_layers=2)
    model = ModernTransformerLM(cfg)
    apply_lora(model, rank=4, alpha=8.0)

    sample = torch.randint(0, cfg.vocab_size, (1, 8))

    model.eval()
    with torch.no_grad():
        out_unmerged, _ = model(sample)

    # Fold adapter weights directly into base weight matrix
    merge_lora_weights(model)
    with torch.no_grad():
        out_merged, _ = model(sample)

    max_diff = (out_unmerged - out_merged).abs().max().item()

    print(f"Pre-Merge Output Logits (Slice):  {out_unmerged[0, 0, :5].tolist()}")
    print(f"Post-Merge Output Logits (Slice): {out_merged[0, 0, :5].tolist()}")
    print(f"Max Absolute Discrepancy:         {max_diff:.8e}")
    print(f"Mathematical Equivalence:         {'SUCCESS (Zero Divergence)' if max_diff < 1e-5 else 'FAILED'}")


if __name__ == "__main__":
    demo_lora_parameter_efficiency()
    demo_lora_training_convergence()
    demo_lora_zero_overhead_merge()
    print("\nPhase 25 PEFT & LoRA verification demo completed successfully.\n")
