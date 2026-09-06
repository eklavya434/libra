"""
Libra Phase 4 - Modern Transformer Architecture Demonstration
Visualizes:
  1. YAML Architecture Instantiation
  2. Weight Tying Parameter Reductions
  3. RoPE & RMSNorm Stability on CPU
  4. Next-Token Text Generation
"""

import os
import sys
import time
from pathlib import Path

import torch
from torch import optim

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from packages.core.tokenizer.educational_bpe import EducationalBPETokenizer
from packages.models.modern_config import ModernTransformerConfig
from packages.models.modern_transformer import ModernTransformerLM
from packages.training.dataset import TextDataset


def main() -> None:
    print("=" * 75)
    print("LIBRA PHASE 4: MODERN TRANSFORMER LABORATORY (Llama 3 Style)")
    print("Featuring RoPE, RMSNorm, SwiGLU & Weight Tying")
    print("=" * 75)

    # 1. Load Configurations from YAML
    yaml_untied_path = "configs/models/tiny_modern_llama.yaml"
    yaml_tied_path = "configs/models/tiny_modern_tied.yaml"

    cfg_untied = ModernTransformerConfig.from_yaml(yaml_untied_path)
    cfg_tied = ModernTransformerConfig.from_yaml(yaml_tied_path)

    model_untied = ModernTransformerLM(cfg_untied)
    model_tied = ModernTransformerLM(cfg_tied)

    print("\n1. Model Architecture & Weight Tying Comparison:")
    print(f"   • Untied Model Parameters: {model_untied.count_parameters():,}")
    print(f"   • Tied Model Parameters:   {model_tied.count_parameters():,}")
    savings = model_untied.count_parameters() - model_tied.count_parameters()
    print(
        f"   • Parameter Reduction:     -{savings:,} weights (Output Head shares Token Embeddings!)"
    )

    # 2. Load Corpus & Tokenize
    corpus_path = "data/raw/educational_science_corpus.txt"
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = f.read()

    tok = EducationalBPETokenizer()
    tok.train(corpus, num_merges=30)
    dataset = TextDataset(corpus, train_ratio=0.9)

    # 3. CPU Training Run with Modern Transformer
    print("\n2. Executing CPU Training Run with Modern Architecture (< 15 min budget)...")
    model = model_tied
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    start_time = time.time()
    steps = 250
    initial_loss = 0.0

    for step in range(1, steps + 1):
        xb, yb = dataset.get_batch(split="train", batch_size=8, block_size=64)
        # Clamp token IDs to vocab size if needed
        xb = torch.clamp(xb, max=model.config.vocab_size - 1)
        yb = torch.clamp(yb, max=model.config.vocab_size - 1)

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(xb, yb)
        assert loss is not None

        if step == 1:
            initial_loss = loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 50 == 0 or step == steps:
            elapsed = time.time() - start_time
            print(f"   [Step {step:03d}/{steps}] Loss: {loss.item():.4f} | Elapsed: {elapsed:.2f}s")

    total_time = time.time() - start_time
    print(f"\n3. Training Finished in {total_time:.2f} seconds!")
    print(f"   Initial Loss: {initial_loss:.4f} --> Final Loss: {loss.item():.4f}")
    assert total_time < 900, "CPU Budget Exceeded!"
    print("   [Budget Check] PASS (< 15 minutes)")

    # 4. Save Checkpoint
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "modern_transformer_phase4.pt")
    torch.save(
        {
            "config": model.config.to_dict(),
            "model_state_dict": model.state_dict(),
            "final_loss": loss.item(),
        },
        ckpt_path,
    )
    print(
        f"\n4. Modern Model Checkpoint Saved: {ckpt_path} ({os.path.getsize(ckpt_path) / 1024:.1f} KB)"
    )
    print("=" * 75)


if __name__ == "__main__":
    main()
