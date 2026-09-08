"""
Libra API v1 - Grammar-Constrained Decoding Endpoints
Phase 27: Regex & Context-Free Grammar (CFG) Token Masking
"""

from __future__ import annotations

from typing import Any

import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from packages.core.grammar.cfg_parser import CFGGrammar
from packages.core.grammar.grammar_processor import GrammarLogitsProcessor
from packages.core.grammar.regex_automaton import RegexAutomaton
from packages.models.grammar_generation import generate_with_grammar
from packages.models.modern_config import ModernTransformerConfig
from packages.models.modern_transformer import ModernTransformerLM
from packages.providers.local_transformer import LocalTransformerProvider

router = APIRouter(prefix="/grammar", tags=["Grammar & Constrained Decoding"])

_local_provider: LocalTransformerProvider | None = None


def get_model_and_tokenizer():
    """Retrieves or initializes local transformer and tokenizer."""
    global _local_provider
    if _local_provider is None:
        _local_provider = LocalTransformerProvider()
    try:
        model, tokenizer = _local_provider._ensure_loaded()
        return model, tokenizer
    except (RuntimeError, FileNotFoundError, OSError):
        config = ModernTransformerConfig(
            vocab_size=256,
            d_model=64,
            n_heads=4,
            n_layers=2,
            max_context_length=512,
        )
        model = ModernTransformerLM(config)
        model.eval()
        return model, None


class GrammarGenerateRequest(BaseModel):
    prompt: str = Field(default="", description="Conditioning prompt text")
    grammar_type: str = Field(
        default="regex",
        description="Type of grammar constraint: 'regex' or 'cfg'",
    )
    grammar_spec: str = Field(
        ...,
        description="Regex pattern (e.g. '\\d{1,3}\\.\\d{1,3}') or EBNF rules (e.g. 'root -> expr\\nexpr -> NUMBER')",
    )
    max_tokens: int = Field(default=24, ge=1, le=256, description="Max generated tokens")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling randomness")


class GrammarValidateRequest(BaseModel):
    grammar_type: str = Field(default="regex", description="'regex' or 'cfg'")
    grammar_spec: str = Field(..., description="Regex pattern or CFG EBNF rules")
    candidate_text: str = Field(..., description="String text to validate against the grammar")


class GrammarNextTokensRequest(BaseModel):
    grammar_type: str = Field(default="regex", description="'regex' or 'cfg'")
    grammar_spec: str = Field(..., description="Regex pattern or CFG EBNF rules")
    prefix_text: str = Field(default="", description="Current generated prefix string")


@router.post("/generate", summary="Generate text strictly constrained by Regex or CFG")
def generate_constrained(request: GrammarGenerateRequest) -> dict[str, Any]:
    """Autoregressively decodes tokens, masking invalid token logits to -inf at every step,

    guaranteeing 100% syntactic compliance with the specified Regex or CFG.
    """
    model, tokenizer = get_model_and_tokenizer()

    # Compile grammar
    try:
        if request.grammar_type.lower() == "regex":
            grammar_obj = RegexAutomaton(request.grammar_spec)
        elif request.grammar_type.lower() == "cfg":
            grammar_obj = CFGGrammar(request.grammar_spec)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported grammar_type '{request.grammar_type}'. Use 'regex' or 'cfg'.",
            )
    except (ValueError, TypeError, IndexError, KeyError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=f"Grammar compilation error: {e!s}")

    decode_fn = (
        tokenizer.decode
        if tokenizer is not None and hasattr(tokenizer, "decode")
        else lambda ids: bytes([i for i in ids if 0 <= i < 256]).decode("utf-8", errors="replace")
    )
    vocab_size = (
        tokenizer.vocab_size
        if tokenizer is not None and hasattr(tokenizer, "vocab_size")
        else getattr(model.config, "vocab_size", 256)
    )

    processor = GrammarLogitsProcessor(
        grammar=grammar_obj,
        decode_fn=decode_fn,
        vocab_size=vocab_size,
        eos_token_id=0,
    )

    if tokenizer is not None and request.prompt:
        prompt_ids = tokenizer.encode(request.prompt)
    elif request.prompt:
        prompt_ids = list(request.prompt.encode("utf-8"))
    else:
        prompt_ids = [0]

    device = next(model.parameters()).device
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    _, text, telemetry = generate_with_grammar(
        model=model,
        idx=idx,
        processor=processor,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        eos_token_id=0,
    )

    # Check acceptance
    if isinstance(grammar_obj, RegexAutomaton):
        is_accepted = grammar_obj.is_accepted(text)
    elif isinstance(grammar_obj, CFGGrammar):
        is_accepted = grammar_obj.is_accepted(text.strip().split())
    else:
        is_accepted = False

    return {
        "prompt": request.prompt,
        "generated_text": text,
        "is_accepted": is_accepted,
        "grammar_type": request.grammar_type,
        "telemetry": telemetry,
    }


@router.post("/validate", summary="Validate string against a grammar or regex")
def validate_against_grammar(request: GrammarValidateRequest) -> dict[str, Any]:
    """Determines whether candidate_text is a valid prefix and fully accepted string."""
    try:
        if request.grammar_type.lower() == "regex":
            grammar = RegexAutomaton(request.grammar_spec)
            is_prefix = grammar.is_valid_prefix(request.candidate_text)
            is_accepted = grammar.is_accepted(request.candidate_text)
        elif request.grammar_type.lower() == "cfg":
            grammar = CFGGrammar(request.grammar_spec)
            words = request.candidate_text.strip().split()
            is_prefix = grammar.is_valid_prefix(words)
            is_accepted = grammar.is_accepted(words)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported grammar_type '{request.grammar_type}'.",
            )
    except (ValueError, TypeError, IndexError, KeyError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=f"Grammar validation error: {e!s}")

    return {
        "candidate_text": request.candidate_text,
        "is_valid_prefix": is_prefix,
        "is_accepted": is_accepted,
    }


@router.post("/next_tokens", summary="Inspect allowed next vocabulary tokens")
def inspect_next_tokens(request: GrammarNextTokensRequest) -> dict[str, Any]:
    """Inspects which tokens in the vocabulary are permitted as the next continuation."""
    model, tokenizer = get_model_and_tokenizer()

    decode_fn = (
        tokenizer.decode
        if tokenizer is not None and hasattr(tokenizer, "decode")
        else lambda ids: bytes([i for i in ids if 0 <= i < 256]).decode("utf-8", errors="replace")
    )
    vocab_size = getattr(model.config, "vocab_size", 256)

    try:
        if request.grammar_type.lower() == "regex":
            grammar = RegexAutomaton(request.grammar_spec)
        elif request.grammar_type.lower() == "cfg":
            grammar = CFGGrammar(request.grammar_spec)
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported grammar type. Use 'regex' or 'cfg'.",
            )
    except (ValueError, TypeError, IndexError, KeyError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    processor = GrammarLogitsProcessor(
        grammar=grammar,
        decode_fn=decode_fn,
        vocab_size=vocab_size,
        eos_token_id=0,
    )

    # Fake logit vector
    dummy_scores = torch.zeros(vocab_size)
    prefix_tokens = (
        tokenizer.encode(request.prefix_text)
        if tokenizer is not None and request.prefix_text
        else list(request.prefix_text.encode("utf-8"))
    )

    dummy_input = (
        torch.tensor([prefix_tokens], dtype=torch.long)
        if prefix_tokens
        else torch.tensor([[]], dtype=torch.long)
    )
    masked = processor(dummy_input, dummy_scores)

    allowed_ids = [i for i in range(vocab_size) if masked[i] > -float("inf")]
    allowed_tokens = [processor.token_strings[i] for i in allowed_ids]

    return {
        "prefix_text": request.prefix_text,
        "allowed_token_count": len(allowed_ids),
        "total_vocab_size": vocab_size,
        "allowed_tokens": allowed_tokens[:20],  # Sample first 20 for preview
        "allowed_ids": allowed_ids[:20],
    }
