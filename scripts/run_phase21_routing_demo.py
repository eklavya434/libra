"""
Libra Educational Demo - Phase 21: Dynamic Model Routing & Speculative Decoding

Demonstrates:
  1. Multi-factor prompt complexity analysis & intent classification.
  2. Dynamic tier routing across Fast/Local, Balanced, Frontier, and Multi-Agent tiers.
  3. First-principles draft-and-verify speculative decoding (Leviathan et al., 2023)
     demonstrating exact mathematical equivalence and target pass reductions.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch

from packages.models.modern_config import ModernTransformerConfig
from packages.models.modern_transformer import ModernTransformerLM
from packages.models.speculative import (
    SpeculativeDecoder,
    standard_autoregressive_generate,
)
from packages.routing.classifier import (
    get_query_classifier,
)
from packages.routing.dynamic_router import (
    RoutingPolicy,
    get_dynamic_router,
)


def demo_query_classification() -> None:
    print("\n" + "=" * 80)
    print("DEMO 1: MULTI-FACTOR QUERY COMPLEXITY & INTENT CLASSIFICATION")
    print("=" * 80)

    classifier = get_query_classifier()

    queries = [
        "Hey! How's it going today?",
        "What is the mathematical definition of Rotary Position Embeddings (RoPE)?",
        "Write an async thread pool in Python with error handling and pytest assertions:\ndef run_pool(): pass",
        "Prove that the derivative of softmax with cross-entropy loss simplifies to (p_i - y_i).",
        "Architect an end-to-end full-stack SaaS platform with multi-agent orchestration and database sharding.",
    ]

    for q in queries:
        res = classifier.classify(q)
        print(f'\nPrompt: "{res.prompt_snippet}"')
        print(f"  • Intent:          {res.intent.value.upper()} (Domain: {res.domain})")
        print(f"  • Recommended Tier: {res.recommended_tier.value.upper()}")
        print(
            f"  • Complexity:      Overall={res.complexity.overall_score:.2f} "
            f"(len={res.complexity.length_score:.2f}, code={res.complexity.code_score:.2f}, "
            f"reason={res.complexity.reasoning_score:.2f}, const={res.complexity.constraint_score:.2f})"
        )
        print(f"  • Rationale:       {res.rationale}")


async def demo_dynamic_routing() -> None:
    print("\n" + "=" * 80)
    print("DEMO 2: DYNAMIC TIER ROUTING & FALLBACK CASCADE EXECUTION")
    print("=" * 80)

    router = get_dynamic_router()

    prompts = [
        ("Good morning!", RoutingPolicy.AUTO),
        ("Summarize the benefits of KV caching.", RoutingPolicy.AUTO),
        ("Write a high-performance vector search indexing algorithm in Rust.", RoutingPolicy.AUTO),
        ("Emergency greeting with forced quality", RoutingPolicy.QUALITY),
    ]

    for prompt, policy in prompts:
        decision = router.plan_routing(prompt, policy=policy)
        print(f"\n[Prompt]: {prompt}")
        print(f"  • Policy:            {policy.value}")
        print(f"  • Selected Tier:     {decision.selected_tier.value}")
        print(f"  • Primary Engine:    {decision.selected_provider} / {decision.selected_model}")
        print(f"  • Estimated Latency: {decision.estimated_latency_tier}")
        print(f"  • Estimated Cost:    {decision.estimated_cost_tier}")
        print(
            f"  • Fallback Chain:    {[fb['provider'] + '/' + fb['model'] for fb in decision.fallback_chain]}"
        )

        # Execute
        messages = [{"role": "user", "content": prompt}]
        exec_res = await router.route_and_execute(messages, policy=policy)
        print(
            f"  • Execution Model:   {exec_res['model_used']} (Fallback Triggered: {exec_res['fallback_triggered']})"
        )


def demo_speculative_decoding() -> None:
    print("\n" + "=" * 80)
    print("DEMO 3: FIRST-PRINCIPLES SPECULATIVE DECODING (LEVIATHAN ET AL., 2023)")
    print("=" * 80)

    torch.manual_seed(42)

    # 1. Instantiate lightweight draft and target models
    draft_cfg = ModernTransformerConfig(
        vocab_size=128,
        d_model=32,
        n_layers=1,
        n_heads=2,
        max_context_length=64,
        hidden_dim=64,
    )
    target_cfg = ModernTransformerConfig(
        vocab_size=128,
        d_model=32,
        n_layers=4,
        n_heads=2,
        max_context_length=64,
        hidden_dim=64,
    )

    draft_model = ModernTransformerLM(draft_cfg)
    target_model = ModernTransformerLM(target_cfg)

    # Simulate aligned draft model (e.g. distilled or early-exit draft)
    draft_model.tok_emb.load_state_dict(target_model.tok_emb.state_dict())
    draft_model.blocks[0].load_state_dict(target_model.blocks[0].state_dict())
    draft_model.norm_f.load_state_dict(target_model.norm_f.state_dict())
    draft_model.output_head.load_state_dict(target_model.output_head.state_dict())

    prompt_tokens = [72, 101, 108, 108, 111]  # ASCII for 'Hello'
    max_tokens = 20

    print(f"Target Model: {target_model.count_parameters():,} parameters (4 layers)")
    print(f"Draft Model:  {draft_model.count_parameters():,} parameters (1 layer)")
    print(f"Prompt:       {[chr(t) if 32 <= t <= 126 else t for t in prompt_tokens]}")
    print(f"Tokens Target: {max_tokens} new tokens\n")

    # A. Standard Autoregressive Generation
    t0 = time.perf_counter()
    std_tokens, std_passes = standard_autoregressive_generate(
        target_model, prompt_tokens, max_new_tokens=max_tokens, temperature=0.0
    )
    t_std = time.perf_counter() - t0

    print("[Standard Autoregressive]")
    print(f"  • Generated Tokens:     {std_tokens}")
    print(f"  • Target Passes:        {std_passes} passes (1 pass per token)")
    print(f"  • Time:                 {t_std * 1000:.2f} ms")

    # B. Speculative Decoding (Lookahead K=3)
    decoder = SpeculativeDecoder(
        target_model=target_model,
        draft_model=draft_model,
        lookahead_k=3,
        temperature=0.0,
    )
    spec_result = decoder.generate(prompt_tokens, max_new_tokens=max_tokens)

    print("\n[Speculative Decoding (Lookahead K=3)]")
    print(f"  • Generated Tokens:     {spec_result.output_tokens}")
    print(
        f"  • Target Passes:        {spec_result.target_forward_passes} passes (Saved {spec_result.passes_saved} passes!)"
    )
    print(f"  • Proposed Drafts:      {spec_result.draft_tokens_proposed}")
    print(f"  • Accepted Drafts:      {spec_result.draft_tokens_accepted}")
    print(f"  • Acceptance Rate (α):  {spec_result.acceptance_rate * 100:.1f}%")
    print(
        f"  • Theoretical Speedup:  {spec_result.theoretical_speedup:.2f}x reduction in target passes"
    )
    print(f"  • Time:                 {spec_result.elapsed_time_sec * 1000:.2f} ms")

    # Verify mathematical identity
    identical = spec_result.output_tokens == std_tokens
    print(
        f"\n[Mathematical Equivalence Invariant]: {'✅ EXACT 100% MATCH' if identical else '❌ MISMATCH'}"
    )
    assert identical, (
        "Speculative decoding under greedy mode MUST match target autoregressive decoding exactly!"
    )


async def main() -> None:
    demo_query_classification()
    await demo_dynamic_routing()
    demo_speculative_decoding()
    print("\n" + "=" * 80)
    print("PHASE 21 DEMONSTRATION COMPLETE: ALL SYSTEMS VERIFIED")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
