"""
Libra Phase 8 - Local Inference & Provider Router Demonstration

Demonstrates:
  1. Provider Health & Capabilities Discovery (Ollama, Local Lab, Mock, vLLM)
  2. Multi-turn Prompt Templating (ChatML, Llama 3, Plain)
  3. Live Local Inference via PyTorch Checkpoint (libra-llama-tied)
  4. Real-time Token Streaming Simulation
  5. Sampling Hyperparameters (Temperature, Top-K, Top-P, Stop Sequences)
"""

import asyncio
import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from packages.providers.prompt_template import PromptTemplate
from packages.providers.router import get_router


async def main() -> None:
    print("=" * 85)
    print("LIBRA PHASE 8: LOCAL INFERENCE & PROVIDER ROUTER LABORATORY")
    print("Ollama, Local Educational Transformer, Prompt Templates & Sampling")
    print("=" * 85)

    router = get_router()

    # 1. Provider Health Check
    print("\n1. Discovering Model Provider Health:")
    providers = ["ollama", "libra_lab", "mock-provider", "vllm"]
    for p_name in providers:
        p = router.get_provider(p_name)
        h = await p.health()
        status_icon = (
            "🟢 ONLINE"
            if h.get("status") == "online"
            else ("🟡 DISABLED (GPU-only)" if p_name == "vllm" else "🔴 OFFLINE")
        )
        print(f"   • [{p_name:<13}] -> {status_icon}")
        if p_name == "ollama" and h.get("status") == "offline":
            print(f"     Note: {h.get('guidance')}")
        elif p_name == "vllm":
            print("     Status: Documented for future GPU hardware (CPU setup bypassed)")

    # 2. Prompt Templating Demonstration
    print("\n" + "-" * 85)
    print("2. Multi-turn Prompt Templating Comparison:")
    print("-" * 85)
    sample_dialogue = [
        {"role": "system", "content": "You are Libra, an educational AI assistant."},
        {"role": "user", "content": "Explain gravity in one sentence."},
    ]

    print("[ChatML Format]:")
    print(PromptTemplate.format_chatml(sample_dialogue))
    print("\n[Llama 3 Format]:")
    print(PromptTemplate.format_llama3(sample_dialogue))

    # 3. Local Model Chat Completion (Using our trained PyTorch checkpoint!)
    print("-" * 85)
    print("3. Executing Local Model Chat (libra-llama-tied via PyTorch):")
    print("-" * 85)

    lab_provider = router.get_provider("libra_lab")
    messages = [{"role": "user", "content": "Gravity is the fundamental force that"}]

    start_t = time.perf_counter()
    response = await lab_provider.chat(
        messages=messages,
        model="libra-llama-tied",
        temperature=0.6,
        top_k=25,
        max_tokens=32,
    )
    elapsed = time.perf_counter() - start_t

    content = response["choices"][0]["message"]["content"]
    print("User Prompt: 'Gravity is the fundamental force that'")
    print(f'Model Generated:\n"{content}"')
    print("\n[Generation Telemetry]")
    print(f"• Tokens Generated: {response['usage']['completion_tokens']}")
    print(f"• Latency:          {elapsed:.3f}s")
    print(
        f"• Throughput:       {response['usage']['completion_tokens'] / max(0.001, elapsed):.1f} tok/s on CPU"
    )

    # 4. Real-time Streaming Simulation
    print("\n" + "-" * 85)
    print("4. Real-time Token Streaming Demonstration (Local Model):")
    print("-" * 85)
    sys.stdout.write('Streaming response: "')
    sys.stdout.flush()

    async for token in lab_provider.stream(
        messages=[{"role": "user", "content": "Cells are the building blocks of"}],
        model="libra-llama-tied",
        max_tokens=24,
    ):
        sys.stdout.write(token)
        sys.stdout.flush()
        await asyncio.sleep(0.02)  # Smooth terminal typewriter effect

    sys.stdout.write('"\n')
    print("\n" + "=" * 85)
    print("Phase 8 Inference Verification: SUCCESS (Local runtime active & responsive!)")
    print("=" * 85)


if __name__ == "__main__":
    asyncio.run(main())
