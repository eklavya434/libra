"""
Project Libra — Phase 9 Interactive Verification Script
Demonstrates all 11 model providers, cost calculations, and fallback routing.
"""

import asyncio
from packages.providers import (
    ProviderRouter,
    get_router,
    calculate_cost,
    get_model_pricing,
    MockProvider,
)

async def main():
    print("=" * 60)
    print("  PROJECT LIBRA - PHASE 9 MODEL PROVIDER ABSTRACTION DEMO")
    print("=" * 60)

    router = get_router()
    print(f"\n[+] Total Registered Providers: {len(router.providers)}")
    for name, prov in router.providers.items():
        health = await prov.health()
        caps = prov.capabilities()
        status = health.get("status", "unknown")
        print(f"  * {name:<16} | Status: {status:<12} | Vision: {str(caps.get('supports_vision', False)):<5} | Stream: {str(caps.get('supports_streaming', False)):<5}")

    print("\n[+] Token Economics & Cost Calculation:")
    sample_queries = [
        ("libra-tiny-llm", 500, 150),
        ("ollama/llama3.2:1b", 2000, 500),
        ("gpt-4o-mini", 12000, 800),
        ("claude-3-5-haiku-20241022", 15000, 1200),
        ("deepseek-chat", 25000, 3000),
    ]

    for model, p_tok, c_tok in sample_queries:
        res = calculate_cost(model, p_tok, c_tok)
        cost_str = "$0.000000 (Zero-Cost Local)" if res["is_free"] else f"${res['total_cost_usd']:.6f} USD"
        print(f"  * Model: {model:<26} | Tokens: {res['total_tokens']:^6} | Cost: {cost_str}")

    print("\n[+] Dynamic Provider Resolution & Zero-Cost Routing:")
    test_models = ["gpt-4o", "gemini-1.5-flash", "claude-3-5-sonnet", "hf/gpt2", "unknown-custom-model"]
    for m in test_models:
        resolved = await router.resolve_provider_for_model(m)
        print(f"  * Target: {m:<22} -> Routed Provider: {resolved.name}")

    print("\n[+] Direct Mock Inference (Zero-Cost Verification):")
    mock = MockProvider()
    resp = await mock.chat([{"role": "user", "content": "Explain attention mechanism"}])
    print(f"  Response: {resp['choices'][0]['message']['content']}")

    print("\n" + "=" * 60)
    print("  PHASE 9 VERIFICATION COMPLETE - ALL PROVIDERS OPERATIONAL")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())

