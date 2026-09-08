"""
Interactive Verification Demo — Phase 26: Streaming Token Telemetry & Token-Level Metrics
Demonstrates real-time token-level confidence, Shannon surprisal, entropy, top-k candidate
distributions, and sequence perplexity from mathematical first principles.
"""

import os
import sys
import time

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Reconfigure stdout to handle UTF-8 on Windows cp1252 consoles
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

import torch

from packages.models.modern_config import ModernTransformerConfig
from packages.models.modern_transformer import ModernTransformerLM
from packages.models.telemetry import (
    analyze_sequence_telemetry,
    stream_generate_with_telemetry,
)

# ANSI Color Codes for terminal visualization
COLOR_RESET = "\033[0m"
COLOR_GREEN = "\033[92m"  # Low surprisal / High confidence (< 1.0 bit)
COLOR_YELLOW = "\033[93m"  # Moderate surprisal (1.0 - 3.0 bits)
COLOR_MAGENTA = "\033[95m"  # High surprisal (> 3.0 bits)
COLOR_CYAN = "\033[96m"
COLOR_BOLD = "\033[1m"
COLOR_DIM = "\033[2m"


def get_token_color(surprisal_bits: float) -> str:
    if surprisal_bits < 1.0:
        return COLOR_GREEN
    elif surprisal_bits <= 3.0:
        return COLOR_YELLOW
    else:
        return COLOR_MAGENTA


def print_banner():
    print(f"\n{COLOR_BOLD}{COLOR_CYAN}{'=' * 75}{COLOR_RESET}")
    print(
        f"{COLOR_BOLD}{COLOR_CYAN}  ♎ PROJECT LIBRA — PHASE 26: STREAMING TOKEN TELEMETRY & METRICS{COLOR_RESET}"
    )
    print(f"{COLOR_BOLD}{COLOR_CYAN}{'=' * 75}{COLOR_RESET}")
    print(
        f"{COLOR_DIM}Consumer CPU Target: Intel Core i5-12450H | Free Offline Inference ($0 / ₹0){COLOR_RESET}\n"
    )


def run_demo():
    print_banner()

    # 1. First-Principles Formula Walkthrough
    print(f"{COLOR_BOLD}1. First-Principles Mathematical Definitions:{COLOR_RESET}")
    print(
        f"   • {COLOR_BOLD}Conditional Probability{COLOR_RESET}:   p(w_t | w_<t) = softmax(z_t)[w_t]"
    )
    print(
        f"   • {COLOR_BOLD}Shannon Surprisal{COLOR_RESET}:         I(w_t) = -log2(p(w_t | w_<t)) [bits]"
    )
    print(
        f"   • {COLOR_BOLD}Distribution Entropy{COLOR_RESET}:      H(P_t) = -sum_v p_t(v) log2(p_t(v)) [bits]"
    )
    print(
        f"   • {COLOR_BOLD}Sequence Perplexity{COLOR_RESET}:       PPL = 2^(mean_surprisal_bits)\n"
    )

    # 2. Model Initialization
    print(f"{COLOR_BOLD}2. Initializing Educational Modern Transformer Architecture:{COLOR_RESET}")
    torch.manual_seed(42)
    config = ModernTransformerConfig(
        vocab_size=128,
        d_model=64,
        n_heads=4,
        n_layers=2,
        max_context_length=128,
    )
    model = ModernTransformerLM(config)
    model.eval()
    print(f"   Model parameters: {model.count_parameters():,} trainable weights")
    print(
        f"   Vocab size: {config.vocab_size} | Embedding dim: {config.d_model} | Layers: {config.n_layers}\n"
    )

    # 3. Live Streaming Generation with Real-Time Telemetry
    prompt_str = "Libra AI"
    prompt_ids = list(prompt_str.encode("utf-8"))
    idx = torch.tensor([prompt_ids], dtype=torch.long)
    max_new_tokens = 12

    print(
        f"{COLOR_BOLD}3. Live Streaming Autoregressive Decoding with Surprisal Heatmap:{COLOR_RESET}"
    )
    print(f'   Prompt: "{prompt_str}"')
    print(
        f"   Surprisal Color Key: {COLOR_GREEN}■ <1.0 bit (Confident){COLOR_RESET}  {COLOR_YELLOW}■ 1.0-3.0 bits (Moderate){COLOR_RESET}  {COLOR_MAGENTA}■ >3.0 bits (Surprising){COLOR_RESET}\n"
    )
    print("   Generated stream: ", end="", flush=True)

    gen = stream_generate_with_telemetry(
        model=model,
        idx=idx,
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        candidate_top_k=3,
    )

    collected_telemetry = []
    for tok in gen:
        collected_telemetry.append(tok)
        color = get_token_color(tok.surprisal_bits)
        # Print colored token
        print(f"{color}{tok.token_text}{COLOR_RESET}", end="", flush=True)
        time.sleep(0.04)  # Visual pacing
    print("\n")

    # 4. Detailed Step-by-Step Breakdown Table
    print(f"{COLOR_BOLD}4. Token-by-Token Telemetry Breakdown Table:{COLOR_RESET}")
    header = f"   {'Step':<5} | {'Token':<8} | {'Prob (%)':<10} | {'Surprisal':<12} | {'Entropy':<10} | {'Latency':<9} | {'Top Candidate Alternatives'}"
    print(f"{COLOR_DIM}{header}{COLOR_RESET}")
    print(f"   {'-' * 74}")

    for tok in collected_telemetry:
        color = get_token_color(tok.surprisal_bits)
        top_str = ", ".join(f'"{c.token_text}" ({(c.prob * 100):.1f}%)' for c in tok.top_k)
        step_str = f"#{tok.index + 1}"
        prob_str = f"{(tok.prob * 100):.2f}%"
        surp_str = f"{tok.surprisal_bits:.3f} bits"
        ent_str = f"{tok.entropy_bits:.3f} bits"
        lat_str = f"{tok.latency_ms:.1f} ms"
        print(
            f"   {step_str:<5} | {color}{tok.token_text:<8}{COLOR_RESET} | {prob_str:<10} | {surp_str:<12} | {ent_str:<10} | {lat_str:<9} | {top_str}"
        )

    print()

    # 5. Aggregate Sequence Metrics
    from packages.models.telemetry import aggregate_sequence_telemetry

    total_time_ms = sum(t.latency_ms for t in collected_telemetry)
    summary = aggregate_sequence_telemetry(collected_telemetry, total_time_ms)

    print(f"{COLOR_BOLD}5. Aggregate Sequence-Level Metrics:{COLOR_RESET}")
    print(f"   • Total Generated Tokens:    {summary.total_tokens}")
    print(f"   • Cumulative Step Latency:   {summary.total_duration_ms:.2f} ms")
    print(f"   • Generation Throughput:     {summary.tokens_per_second:.1f} tokens/second")
    print(f"   • Mean Sequence Surprisal:   {summary.mean_surprisal_bits:.4f} bits/token")
    print(
        f"   • Sequence Perplexity:       {COLOR_BOLD}{COLOR_CYAN}{summary.perplexity:.4f}{COLOR_RESET} (2^{summary.mean_surprisal_bits:.4f})"
    )
    print(f"   • Mean Distribution Entropy: {summary.mean_entropy_bits:.4f} bits")
    if summary.max_surprisal_token:
        print(
            f'   • Most Surprising Token:     "{summary.max_surprisal_token.token_text}" ({summary.max_surprisal_token.surprisal_bits:.3f} bits, p = {(summary.max_surprisal_token.prob * 100):.2f}%)'
        )
    if summary.min_surprisal_token:
        print(
            f'   • Most Confident Token:      "{summary.min_surprisal_token.token_text}" ({summary.min_surprisal_token.surprisal_bits:.3f} bits, p = {(summary.min_surprisal_token.prob * 100):.2f}%)'
        )

    # 6. Teacher-Forcing Analysis Demo
    print(
        f"\n{COLOR_BOLD}6. Teacher-Forcing Sequence Surprisal Evaluation (Without Sampling):{COLOR_RESET}"
    )
    test_phrase = "Deep learning laboratory"
    test_tokens = torch.tensor([list(test_phrase.encode("utf-8"))], dtype=torch.long)
    tf_summary = analyze_sequence_telemetry(model, test_tokens, candidate_top_k=2)

    print(f'   Evaluated text: "{test_phrase}" ({tf_summary.total_tokens} transitions)')
    print(
        f"   Evaluation Perplexity: {tf_summary.perplexity:.4f} | Mean Surprisal: {tf_summary.mean_surprisal_bits:.4f} bits/tok"
    )

    print(
        f"\n{COLOR_BOLD}{COLOR_GREEN}✓ Phase 26 Streaming Token Telemetry verification completed successfully.{COLOR_RESET}\n"
    )


if __name__ == "__main__":
    run_demo()
