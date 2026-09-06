"""
Phase 17 Interactive Demonstration: Structured Outputs & Grammar Decoders

Demonstrates:
1. Compiling Pydantic models into strict JSON Schema constraints.
2. Character-level Pushdown Automaton transitions for partial JSON strings.
3. Logit masking demonstration (-inf on invalid candidate tokens).
4. Self-healing repair loop correcting malformed schema outputs.
"""

from __future__ import annotations

import json
import os
import sys
from typing import List, Optional
from pydantic import BaseModel, Field
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from packages.core.grammar import (
    ConstrainedLogitsProcessor,
    IncrementalJSONStateMachine,
    SchemaCompiler,
)
from packages.providers.base import BaseProvider, ModelMetadata
from packages.providers.router import ProviderRouter
from packages.providers.structured import StructuredOutputGenerator


class BenchmarkResult(BaseModel):
    model_name: str = Field(..., description="Name of the model tested")
    accuracy: float = Field(..., description="Accuracy between 0.0 and 1.0")
    latency_ms: float = Field(..., description="Latency in milliseconds")
    tags: List[str] = Field(default_factory=list, description="Categorization tags")


def print_banner(title: str) -> None:
    print("\n" + "=" * 75)
    print(f"  {title}")
    print("=" * 75)


def demo_schema_compilation():
    print_banner("1. PYDANTIC SCHEMA COMPILATION & RESPONSE_FORMAT")
    constraint = SchemaCompiler.compile(BenchmarkResult)
    print(f"[+] Compiled Schema Name: {constraint.name}")
    print(f"[+] Required Properties:  {constraint.required}")
    print("\n[+] OpenAI/Ollama Compatible Response Format:")
    print(json.dumps(constraint.to_openai_response_format(), indent=2))


def demo_incremental_state_machine():
    print_banner("2. INCREMENTAL PUSHDOWN AUTOMATON (PDA) TRACE")
    pda = IncrementalJSONStateMachine()
    test_fragments = ['{', '"model_name"', ':', '"TinyTransformer"', ',', '"accuracy"', ':', '0.94', '}']

    current = ""
    for frag in test_fragments:
        for ch in frag:
            pda.feed_char(ch)
            current += ch
        allowed = sorted(list(pda.get_allowed_characters()))[:8]
        print(f"  Prefix: {current:<35} | Complete: {str(pda.is_complete()):<5} | Sample Allowed Next: {allowed}")


def demo_logit_masking():
    print_banner("3. LOGIT PROCESSOR MASKING DEMONSTRATION")
    vocab = ["<eos>", " ", "{", "}", '"', "result", ":", "100", "invalid_token"]
    decode_fn = lambda ids: "".join(vocab[i] for i in ids)

    processor = ConstrainedLogitsProcessor(
        decode_fn=decode_fn,
        vocab_size=len(vocab),
        eos_token_id=0,
    )

    # Prefix is '{"result"'
    prefix_tokens = [vocab.index("{"), vocab.index('"'), vocab.index("result"), vocab.index('"')]
    input_ids = torch.tensor(prefix_tokens, dtype=torch.long)
    raw_scores = torch.zeros(len(vocab))

    masked_scores = processor(input_ids, raw_scores)

    print("  Generated Prefix: '{\"result\"'")
    print("  Candidate Tokens Logits:")
    for i, tok in enumerate(vocab):
        status = "ALLOWED" if masked_scores[i] == 0.0 else "MASKED (-inf)"
        print(f"    Token [{i}] '{tok:<13}': {masked_scores[i]:>6.1f}  ({status})")


class SimulatedFlakyProvider(BaseProvider):
    def __init__(self) -> None:
        self.call_count = 0

    @property
    def name(self) -> str:
        return "flaky-sim"

    async def list_models(self) -> list[ModelMetadata]:
        return []

    async def chat(self, messages, model=None, **kwargs):
        self.call_count += 1
        if self.call_count == 1:
            print("    [Attempt 1] Model emitted incomplete JSON (missing 'latency_ms')...")
            return {
                "choices": [{"message": {"content": '{"model_name": "Libra-1", "accuracy": 0.92}'}}]
            }
        else:
            print("    [Attempt 2] Model received validation feedback and generated corrected JSON...")
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"model_name": "Libra-1", "accuracy": 0.92, '
                                '"latency_ms": 14.5, "tags": ["cpu", "fast"]}'
                            )
                        }
                    }
                ]
            }

    async def stream(self, messages, model=None, **kwargs):
        yield ""

    async def health(self):
        return {"status": "healthy"}

    async def embeddings(self, texts, model=None):
        return [[0.0] * 16]

    def capabilities(self):
        return {"supports_text": True}


def demo_self_healing_repair():
    print_banner("4. SELF-HEALING REFLECTION REPAIR LOOP")
    router = ProviderRouter()
    router._providers["flaky-sim"] = SimulatedFlakyProvider()

    generator = StructuredOutputGenerator(router=router)
    print("  Prompt: 'Evaluate Libra-1 performance metrics'")
    res = generator.generate_sync(
        prompt="Evaluate Libra-1 performance metrics",
        schema=BenchmarkResult,
        model_id="flaky-sim",
        provider_name="flaky-sim",
        max_retries=2,
    )

    print(f"\n  Result Success: {res.success}")
    print(f"  Total Attempts: {res.attempts}")
    print(f"  Parsed Data:    {res.data}")


def main():
    print("=" * 75)
    print("      LIBRA PHASE 17: STRUCTURED OUTPUTS & GRAMMAR DECODING DEMO")
    print("=" * 75)
    demo_schema_compilation()
    demo_incremental_state_machine()
    demo_logit_masking()
    demo_self_healing_repair()
    print_banner("PHASE 17 DEMO COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
