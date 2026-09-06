"""
Libra Phase 1 - Executable Training & Text Generation Demonstration
Demonstrates the full lifecycle:
Raw Text -> Tokenization -> Untrained Generation -> CPU Training -> Checkpoint -> Trained Generation.
"""

import os
import sys
import time
from pathlib import Path

import torch

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from packages.models.config import TinyTransformerConfig
from packages.models.generation import decode_tokens, encode_string, generate
from packages.models.transformer import TinyTransformerLM
from packages.training.dataset import TextDataset
from packages.training.trainer import train_tiny_llm


def main() -> None:
    print("=" * 70)
    print("LIBRA PHASE 1: EDUCATIONAL TINY LLM TRAINING DEMO")
    print("Target Hardware: CPU (Intel Core i5-12450H)")
    print("=" * 70)

    corpus_path = "data/raw/tiny_corpus.txt"
    if not os.path.exists(corpus_path):
        print(f"Error: Corpus not found at {corpus_path}")
        sys.exit(1)

    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus_text = f.read()

    print(f"1. Loaded Corpus: {len(corpus_text)} characters.")

    # Model Hyperparameters
    config = TinyTransformerConfig(
        vocab_size=256,  # Byte-level vocabulary
        max_context_length=128,  # Sequence context window
        d_model=128,  # Hidden embedding size
        n_heads=4,  # 4 Attention heads (head_dim = 32)
        n_layers=2,  # 2 Transformer blocks
        mlp_ratio=4,  # FeedForward inner dim = 512
        dropout=0.1,
        device="cpu",
    )

    model = TinyTransformerLM(config)
    print(f"2. Initialized Model: {model.count_parameters():,} trainable parameters.")

    # Demonstration 1: Text generation BEFORE training (Random Weights)
    prompt = "Libra is "
    prompt_tokens = torch.tensor([encode_string(prompt)], dtype=torch.long)

    print("\n--- SAMPLE GENERATION BEFORE TRAINING (Random Weights) ---")
    untrained_out = generate(model, prompt_tokens, max_new_tokens=40, temperature=0.8)
    print(repr(decode_tokens(untrained_out[0].tolist())))
    print("----------------------------------------------------------\n")

    # Dataset & Training
    dataset = TextDataset(corpus_text, train_ratio=0.9)

    print("3. Executing CPU Training Run (Budget: < 15 minutes)...")
    start_time = time.time()
    metrics = train_tiny_llm(
        model=model,
        dataset=dataset,
        max_steps=500,
        batch_size=8,
        learning_rate=1e-3,
        eval_interval=100,
        checkpoint_path="checkpoints/tiny_llm_phase1.pt",
    )
    total_time = time.time() - start_time

    print(f"\n4. Training Finished in {total_time:.2f} seconds!")
    print(f"   Initial Loss: {metrics.initial_loss:.4f}  -->  Final Loss: {metrics.final_loss:.4f}")
    assert total_time < 900, "CPU Training budget exceeded!"
    print("   [Budget Check] PASS (< 15 minutes)")

    # Demonstration 2: Text generation AFTER training
    print("\n--- SAMPLE GENERATION AFTER TRAINING ---")
    trained_out = generate(model, prompt_tokens, max_new_tokens=80, temperature=0.5)
    print(repr(decode_tokens(trained_out[0].tolist())))
    print("----------------------------------------\n")

    checkpoint_size_mb = os.path.getsize(metrics.checkpoint_path) / (1024 * 1024)
    print(f"5. Checkpoint Saved: {metrics.checkpoint_path} ({checkpoint_size_mb:.2f} MB)")
    print("=" * 70)


if __name__ == "__main__":
    main()
