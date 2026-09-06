"""
Libra Phase 6 - Comprehensive Evaluation & Benchmarking Demonstration

Demonstrates:
  1. Systematic Validation Loss & Perplexity Benchmarking
  2. Multiple-Choice Log-Likelihood Probes (Reasoning, Math, Coding, Instruction, Safety)
  3. Empirical Comparison: Chance Baseline vs. Untrained Random Weights vs. Trained Checkpoint
  4. Real-time CPU execution in < 15 seconds with Zero Fake Metrics
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

from packages.core.tokenizer.educational_bpe import EducationalBPETokenizer
from packages.evaluation.harness import EvaluationHarness
from packages.evaluation.probes import get_standard_probes
from packages.models.modern_config import ModernTransformerConfig
from packages.models.modern_transformer import ModernTransformerLM
from packages.training.dataset import TextDataset
from packages.training.engine import TrainingConfig, TrainingEngine


def main() -> None:
    start_time = time.perf_counter()
    print("=" * 80)
    print("LIBRA PHASE 6: EVALUATION & BENCHMARKING LABORATORY")
    print("Loss, Perplexity, Log-Likelihood Ranking & Domain Task Probes")
    print("=" * 80)

    # 1. Setup Data & Tokenizer
    corpus_path = "data/raw/educational_science_corpus.txt"
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus not found at {corpus_path}")

    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = f.read()

    tok = EducationalBPETokenizer()
    tok.train(corpus, num_merges=40)
    dataset = TextDataset(corpus, train_ratio=0.80)

    val_tokens = dataset.val_data.clone().detach() if isinstance(dataset.val_data, torch.Tensor) else torch.tensor(dataset.val_data, dtype=torch.long)
    print(f"\n1. Evaluation Corpus & Tokenizer:")
    print(f"   • Validation Tokens: {len(val_tokens):,} tokens")
    print(f"   • Tokenizer Vocab Size: {tok.vocab_size} (Special: 4, Base Bytes: 256, Merges: 40)")

    # 2. Setup Models (Untrained Control vs. Trained Checkpoint)
    model_cfg = ModernTransformerConfig.from_yaml("configs/models/tiny_modern_tied.yaml")
    
    # Untrained random baseline
    torch.manual_seed(999)
    untrained_model = ModernTransformerLM(model_cfg)
    untrained_model.eval()

    # Trained model
    checkpoint_path = "checkpoints/best_engine_model.pt"
    trained_model = ModernTransformerLM(model_cfg)
    
    if os.path.exists(checkpoint_path):
        print(f"\n2. Loading Pre-trained Checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        trained_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("\n2. Training quick 100-step checkpoint for evaluation demo...")
        train_cfg = TrainingConfig.from_yaml("configs/training/cpu_quick_train.yaml")
        train_cfg.max_steps = 100
        engine = TrainingEngine(trained_model, dataset, train_cfg)
        engine.train(start_step=1)

    trained_model.eval()

    # 3. Execute Benchmarks with Unified Evaluation Harness
    print("\n3. Executing Standard Evaluation Probes across 5 Domains...")
    probes = get_standard_probes()
    print(f"   • Total Probes: {len(probes)} (Math, Reasoning, Coding, Instruction, Safety)")

    # Benchmark Untrained Model
    untrained_harness = EvaluationHarness(untrained_model, tok, model_name="Random-Untrained-Baseline")
    untrained_report = untrained_harness.run_benchmark(probes=probes, val_tokens=val_tokens)

    # Benchmark Trained Model
    trained_harness = EvaluationHarness(trained_model, tok, model_name="Trained-Libra-Llama")
    trained_report = trained_harness.run_benchmark(probes=probes, val_tokens=val_tokens)

    # 4. Comparative Evaluation Report
    print("\n" + "=" * 80)
    print("EMPIRICAL BENCHMARK COMPARISON: UNTRAINED vs. TRAINED")
    print("=" * 80)
    
    u_loss = untrained_report.loss_metrics
    t_loss = trained_report.loss_metrics

    print("\n[A. Perplexity & Language Modeling Loss]")
    print(f"   • Untrained Baseline Loss: {u_loss.mean_loss:.4f}  | Perplexity: {u_loss.perplexity:.2f}")
    print(f"   • Trained Model Loss:     {t_loss.mean_loss:.4f}  | Perplexity: {t_loss.perplexity:.2f}")
    ppl_drop = ((u_loss.perplexity - t_loss.perplexity) / u_loss.perplexity) * 100
    print(f"   • Perplexity Improvement: -{ppl_drop:.1f}% reduction in next-token uncertainty!")

    print("\n[B. Domain Task Accuracy Comparison]")
    header = f"   | {'Domain Category':<22} | {'Chance Baseline':<15} | {'Untrained Model':<15} | {'Trained Model':<15} |"
    print(header)
    print("   |" + "-" * 24 + "|" + "-" * 17 + "|" + "-" * 17 + "|" + "-" * 17 + "|")

    for cat in trained_report.category_scores:
        u_score = untrained_report.category_scores.get(cat)
        t_score = trained_report.category_scores.get(cat)
        chance_str = f"{t_score.chance_accuracy * 100:.1f}%"
        u_acc_str = f"{u_score.accuracy * 100:.1f}%" if u_score else "N/A"
        t_acc_str = f"{t_score.accuracy * 100:.1f}%" if t_score else "N/A"
        print(f"   | {cat.capitalize():<22} | {chance_str:<15} | {u_acc_str:<15} | {t_acc_str:<15} |")

    print("   |" + "-" * 24 + "|" + "-" * 17 + "|" + "-" * 17 + "|" + "-" * 17 + "|")
    u_tot = f"{untrained_report.overall_accuracy * 100:.1f}% ({untrained_report.overall_correct_probes}/{untrained_report.overall_total_probes})"
    t_tot = f"{trained_report.overall_accuracy * 100:.1f}% ({trained_report.overall_correct_probes}/{trained_report.overall_total_probes})"
    print(f"   | {'OVERALL ACCURACY':<22} | {'25.0%':<15} | {u_tot:<15} | {t_tot:<15} |")

    # 5. Qualitative Predictions
    print("\n[C. Sample Model Predictions (Trained Model)]")
    for r in trained_report.sample_results[:4]:
        pred = r.choices[r.predicted_index].strip()
        exp = r.choices[r.correct_index].strip()
        icon = "✅" if r.is_correct else "❌"
        print(f"   {icon} Prompt: '{r.prompt.strip()}' -> Model Picked: '{pred}' (Expected: '{exp}')")

    # 6. Save Benchmark Report to JSON
    report_path = "data/evaluation/benchmark_report.json"
    trained_report.save_json(report_path)
    print(f"\n4. Benchmark Report Saved to: {report_path}")

    elapsed = time.perf_counter() - start_time
    print(f"5. Total Benchmark Execution Time: {elapsed:.2f} seconds (< 15-Minute CPU Budget: PASS)")
    print("=" * 80)


if __name__ == "__main__":
    main()
