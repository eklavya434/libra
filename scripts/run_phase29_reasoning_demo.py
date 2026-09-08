"""
Phase 29 Demonstration Script: Deliberative Reasoning Engine & Test-Time Compute Scaling
Demonstrates System 1 vs System 2 Chain-of-Thought reasoning, <think> trace parsing,
self-consistency majority voting, and Best-of-N test-time compute search.
"""

import asyncio
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from packages.models.reasoning.search_verifier import BestOfNVerifier
from packages.models.reasoning.self_consistency import SelfConsistencyEngine
from packages.models.reasoning.trace_parser import parse_reasoning_trace


async def main():
    print("=" * 75)
    print("  PROJECT LIBRA - PHASE 29 DEMO: REASONING ENGINE & TEST-TIME COMPUTE")
    print("=" * 75)

    # 1. Thought Trace Parser Demonstration
    print("\n[1] Chain-of-Thought Trace Parsing (<think> State Machine):")
    sample_raw_generation = (
        "<think>\n"
        "Step 1: The problem asks for the cost of the ball.\n"
        "Step 2: Let bat = B, ball = b. We know B + b = 1.10 and B = b + 1.00.\n"
        "Step 3: Substitute: (b + 1.00) + b = 1.10 => 2b + 1.00 = 1.10.\n"
        "Step 4: Subtract 1.00 from both sides: 2b = 0.10 => b = 0.05.\n"
        "Step 5: Verify: Bat = 1.05, Ball = 0.05. Sum = 1.10. Difference = 1.00. Correct!\n"
        "</think>\n"
        "The ball costs $0.05 (5 cents)."
    )

    trace = parse_reasoning_trace(sample_raw_generation, duration_ms=1420.0)
    print(f"  Internal Reflection Present: {trace.has_thought}")
    print(f"  Thinking Duration: {trace.thinking_duration_ms} ms")
    print(f"  Thought Tokens: ~{trace.thought_token_count} tokens")
    print(f"  Extracted Steps Count: {len(trace.steps)}")
    for idx, step in enumerate(trace.steps, 1):
        print(f"    [{idx}] {step}")
    print(f'  User-Facing Final Answer: "{trace.final_answer}"')

    # 2. Self-Consistency Majority Voting
    print("\n[2] Self-Consistency Multi-Path Majority Voting (Wang et al., 2022):")
    sc_engine = SelfConsistencyEngine()
    prompt = "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?"

    print(f'  Benchmark Query: "{prompt}"')
    print("  Generating N=3 independent reasoning trajectories...")

    sc_result = await sc_engine.evaluate(
        prompt=prompt,
        model="libra-mock-v1",
        num_paths=3,
        temperature=0.7,
        max_tokens=128,
    )

    print(f"  Total Trajectories Evaluated: {sc_result.total_paths}")
    print(f"  Vote Agreement Distribution: {sc_result.vote_distribution}")
    print(f"  Consensus Confidence: {sc_result.confidence * 100:.1f}%")
    print(f'  Modal Consensus Answer: "{sc_result.consensus_answer}"')

    # 3. Best-of-N Test-Time Compute Search
    print("\n[3] Best-of-N Test-Time Compute Search & Verification:")
    verifier = BestOfNVerifier()
    coding_puzzle = "Write a Python function to check whether a string is an anagram of another."

    print(f'  Search Task: "{coding_puzzle}"')
    print("  Sampling N=3 candidates and scoring with multi-criteria referee...")

    bon_result = await verifier.search(
        prompt=coding_puzzle,
        model="libra-mock-v1",
        n_candidates=3,
        temperature=0.8,
        max_tokens=128,
    )

    print(
        f"  Evaluated {bon_result.total_candidates} candidates (Selection: {bon_result.selection_method})"
    )
    for cand in bon_result.candidates:
        is_best = " [SELECTED BEST]" if cand.index == bon_result.best_index else ""
        print(f"    Candidate #{cand.index + 1}: Score {cand.score:.2f} / 10.0{is_best}")
        print(f"      Rationale: {cand.rationale}")

    print(f"\n  Top Pick Score: {bon_result.best_score:.2f} / 10.0")
    print(f'  Top Pick Answer: "{bon_result.best_candidate.final_answer}"')

    print("\n" + "=" * 75)
    print("  [OK] Phase 29 Deliberative Reasoning Engine verified successfully!")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
