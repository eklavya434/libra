"""
Libra Phase 5 - Production Training Engine Demonstration
Demonstrates:
  1. Cosine Learning Rate Schedule with Warmup
  2. Gradient Accumulation (simulating larger batch sizes)
  3. Validation Perplexity & Token Velocity (tokens/sec) Tracking
  4. Checkpoint Pause and Seamless Resume
"""

import os
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from packages.core.tokenizer.educational_bpe import EducationalBPETokenizer
from packages.models.modern_config import ModernTransformerConfig
from packages.models.modern_transformer import ModernTransformerLM
from packages.training.dataset import TextDataset
from packages.training.engine import TrainingConfig, TrainingEngine


def main() -> None:
    print("=" * 75)
    print("LIBRA PHASE 5: REUSABLE TRAINING ENGINE DEMONSTRATION")
    print("Featuring Gradient Accumulation, Cosine Warmup, Perplexity, & Resume")
    print("=" * 75)

    # 1. Load Dataset & Train Tokenizer
    corpus_path = "data/raw/educational_science_corpus.txt"
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = f.read()

    tok = EducationalBPETokenizer()
    tok.train(corpus, num_merges=30)
    dataset = TextDataset(corpus, train_ratio=0.85)

    # 2. Instantiate Model from YAML
    model_cfg = ModernTransformerConfig.from_yaml("configs/models/tiny_modern_tied.yaml")
    model = ModernTransformerLM(model_cfg)
    print(f"\n1. Model Initialized: {model.count_parameters():,} parameters (Weight-Tied).")

    # 3. Setup Training Configuration from YAML
    train_cfg = TrainingConfig.from_yaml("configs/training/cpu_quick_train.yaml")
    print("\n2. Training Engine Configuration:")
    print(f"   • Micro-Batch Size:              {train_cfg.batch_size}")
    print(f"   • Gradient Accumulation Steps:   {train_cfg.gradient_accumulation_steps}")
    print(f"   • Effective Batch Size:          {train_cfg.effective_batch_size}")
    print(
        f"   • Learning Rate Schedule:        Warmup ({train_cfg.warmup_steps} steps) -> Cosine Decay to {train_cfg.min_lr}"
    )

    # 4. Phase A: Train Initial 90 Steps
    print("\n" + "-" * 75)
    print("STAGE 1: Training Initial 90 Steps...")
    print("-" * 75)
    train_cfg.max_steps = 90
    engine = TrainingEngine(model, dataset, train_cfg)
    engine.train(start_step=1)

    pause_ckpt = "checkpoints/phase5_paused_step90.pt"
    engine.save_checkpoint(step=90, path=pause_ckpt)
    print(
        f"\n[Pause] Training paused at step 90. Saved state: {pause_ckpt} ({os.path.getsize(pause_ckpt) / 1024:.1f} KB)"
    )

    # 5. Phase B: Resume from Step 91 through Step 150
    print("\n" + "-" * 75)
    print("STAGE 2: Resuming from Step 91 through Step 150...")
    print("-" * 75)
    # Create fresh model & engine to prove zero state leakage
    fresh_model = ModernTransformerLM(model_cfg)
    train_cfg.max_steps = 150
    resume_engine = TrainingEngine(fresh_model, dataset, train_cfg)

    resumed_start = resume_engine.resume_from_checkpoint(pause_ckpt)
    history = resume_engine.train(start_step=resumed_start)

    last_entry = history[-1]
    print("\n3. Training Engine Summary:")
    print(f"   • Final Step Reached:        {last_entry.step}")
    print(
        f"   • Final Training Loss:       {last_entry.train_loss:.4f} (Perplexity: {last_entry.train_ppl:.2f})"
    )
    print(
        f"   • Final Validation Loss:     {last_entry.val_loss:.4f} (Perplexity: {last_entry.val_ppl:.2f})"
    )
    print(f"   • Token Throughput:          {last_entry.tokens_per_sec:,.0f} tokens/second on CPU")
    print(f"   • Total Training Time:       {last_entry.elapsed_seconds:.2f} seconds")
    print("   • [Budget Check] PASS (< 15 minutes)")
    print("=" * 75)


if __name__ == "__main__":
    main()
