"""
Libra Models - Streaming Token Telemetry and Token-Level Metrics
Computes fine-grained token-level statistics from first principles:
- Token conditional probability: p(w_t | w_<t)
- Shannon Surprisal (Information content): I(w_t) = -log2 p(w_t | w_<t) in bits
- Next-token distribution entropy: H(P_t) = -sum_v p_t(v) log2 p_t(v) in bits
- Top-k candidate alternatives with individual probabilities and log-probabilities
- Per-token step generation latency (ms) and throughput (tokens/sec)
- Sequence perplexity: PPL = 2^(mean_surprisal)
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import AsyncGenerator, Generator
from typing import Any

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field


class CandidateToken(BaseModel):
    """Candidate alternative token in next-token distribution."""

    token_id: int = Field(..., description="Vocabulary token index")
    token_text: str = Field(..., description="Decoded text representation")
    prob: float = Field(..., description="Conditional probability p(v | w_<t)")
    logprob: float = Field(..., description="Natural log-probability ln p(v | w_<t)")


class TokenTelemetry(BaseModel):
    """Fine-grained telemetry data for a single generated token step."""

    index: int = Field(..., description="0-indexed position within generated sequence")
    token_id: int = Field(..., description="Chosen vocabulary token index")
    token_text: str = Field(..., description="Decoded token string representation")
    prob: float = Field(..., description="Probability of chosen token p(w_t | w_<t)")
    logprob: float = Field(..., description="Natural log probability ln p(w_t | w_<t)")
    surprisal_bits: float = Field(
        ...,
        description="Shannon self-information / surprisal: -log2 p(w_t) in bits",
    )
    entropy_bits: float = Field(
        ...,
        description="Shannon entropy of the next-token distribution H(P_t) in bits",
    )
    latency_ms: float = Field(
        ...,
        description="Time taken to compute this token forward step in milliseconds",
    )
    top_k: list[CandidateToken] = Field(
        default_factory=list,
        description="Top-k alternative candidates considered at this decode step",
    )


class SequenceTelemetry(BaseModel):
    """Aggregated telemetry metrics for a complete sequence."""

    tokens: list[TokenTelemetry] = Field(
        default_factory=list,
        description="Chronological token telemetry entries",
    )
    text: str = Field("", description="Complete generated or analyzed text")
    total_tokens: int = Field(0, description="Total number of generated/evaluated tokens")
    total_duration_ms: float = Field(0.0, description="Total generation duration in milliseconds")
    tokens_per_second: float = Field(0.0, description="Generation throughput in tokens/second")
    mean_surprisal_bits: float = Field(
        0.0,
        description="Mean surprisal across sequence (bits/token)",
    )
    perplexity: float = Field(
        1.0,
        description="Sequence perplexity: 2^(mean_surprisal_bits)",
    )
    mean_entropy_bits: float = Field(
        0.0,
        description="Mean distribution entropy across all steps in bits",
    )
    max_surprisal_token: TokenTelemetry | None = Field(
        None,
        description="The single most surprising / unexpected token in the sequence",
    )
    min_surprisal_token: TokenTelemetry | None = Field(
        None,
        description="The single most confident / expected token in the sequence",
    )


def decode_token_id(token_id: int, tokenizer: Any = None) -> str:
    """Decodes a single token index into its UTF-8 string representation."""
    if tokenizer is not None:
        try:
            if hasattr(tokenizer, "decode"):
                decoded = tokenizer.decode([token_id])
                if isinstance(decoded, str):
                    return decoded
        except (AttributeError, ValueError, TypeError):
            pass

    # Byte/ASCII fallback
    if 0 <= token_id < 256:
        try:
            return bytes([token_id]).decode("utf-8", errors="replace")
        except (UnicodeDecodeError, ValueError):
            pass

    return f"<{token_id}>"


def compute_surprisal_bits(prob: float, eps: float = 1e-12) -> float:
    """Calculates Shannon self-information / surprisal in bits: I(w) = -log2(p(w)).

    Properties:
    - Certain event (p = 1.0) -> surprisal = 0.0 bits.
    - Improbable event (p -> 0) -> surprisal -> infinity.
    - Always non-negative.
    """
    safe_prob = max(float(prob), eps)
    return -math.log2(safe_prob)


def compute_entropy_bits(probs: torch.Tensor, eps: float = 1e-12) -> float:
    """Calculates Shannon entropy in bits for a discrete probability vector:

    H(P) = -sum_{v} p(v) * log2(p(v))

    Properties:
    - Minimum (deterministic one-hot): H = 0.0 bits.
    - Maximum (uniform over V items): H = log2(V) bits.
    """
    mask = probs > eps
    if not torch.any(mask):
        return 0.0
    p = probs[mask]
    entropy = -torch.sum(p * torch.log2(p)).item()
    return max(0.0, float(entropy))


def compute_step_telemetry(
    logits: torch.Tensor,
    chosen_token_id: int,
    index: int,
    latency_ms: float,
    tokenizer: Any = None,
    candidate_top_k: int = 5,
    temperature: float = 1.0,
) -> TokenTelemetry:
    """Computes complete step telemetry for a single token decode step."""
    if logits.dim() == 2:
        logits = logits.squeeze(0)

    scaled_logits = (
        logits if temperature <= 0.0 or math.isclose(temperature, 1.0) else logits / temperature
    )
    probs = F.softmax(scaled_logits, dim=-1)

    chosen_prob = float(probs[chosen_token_id].item())
    safe_prob = max(chosen_prob, 1e-12)
    chosen_logprob = math.log(safe_prob)
    surprisal = compute_surprisal_bits(chosen_prob)
    entropy = compute_entropy_bits(probs)

    k = min(candidate_top_k, probs.size(-1))
    top_probs, top_indices = torch.topk(probs, k=k)

    top_candidates: list[CandidateToken] = []
    for p_val, idx_val in zip(top_probs.tolist(), top_indices.tolist()):
        c_prob = float(p_val)
        c_logp = math.log(max(c_prob, 1e-12))
        c_text = decode_token_id(idx_val, tokenizer=tokenizer)
        top_candidates.append(
            CandidateToken(
                token_id=idx_val,
                token_text=c_text,
                prob=round(c_prob, 5),
                logprob=round(c_logp, 5),
            )
        )

    token_text = decode_token_id(chosen_token_id, tokenizer=tokenizer)

    return TokenTelemetry(
        index=index,
        token_id=chosen_token_id,
        token_text=token_text,
        prob=round(chosen_prob, 5),
        logprob=round(chosen_logprob, 5),
        surprisal_bits=round(surprisal, 4),
        entropy_bits=round(entropy, 4),
        latency_ms=round(latency_ms, 2),
        top_k=top_candidates,
    )


def aggregate_sequence_telemetry(
    tokens: list[TokenTelemetry],
    total_duration_ms: float,
    text: str = "",
) -> SequenceTelemetry:
    """Aggregates a list of TokenTelemetry steps into SequenceTelemetry."""
    total_tokens = len(tokens)
    if total_tokens == 0:
        return SequenceTelemetry(
            tokens=[],
            text=text,
            total_tokens=0,
            total_duration_ms=round(total_duration_ms, 2),
            tokens_per_second=0.0,
            mean_surprisal_bits=0.0,
            perplexity=1.0,
            mean_entropy_bits=0.0,
            max_surprisal_token=None,
            min_surprisal_token=None,
        )

    if not text:
        text = "".join(t.token_text for t in tokens)

    duration_sec = max(1e-6, total_duration_ms / 1000.0)
    tok_per_sec = total_tokens / duration_sec

    mean_surprisal = sum(t.surprisal_bits for t in tokens) / total_tokens
    perplexity = 2.0**mean_surprisal

    mean_entropy = sum(t.entropy_bits for t in tokens) / total_tokens
    max_surprisal = max(tokens, key=lambda t: t.surprisal_bits)
    min_surprisal = min(tokens, key=lambda t: t.surprisal_bits)

    return SequenceTelemetry(
        tokens=tokens,
        text=text,
        total_tokens=total_tokens,
        total_duration_ms=round(total_duration_ms, 2),
        tokens_per_second=round(tok_per_sec, 2),
        mean_surprisal_bits=round(mean_surprisal, 4),
        perplexity=round(perplexity, 4),
        mean_entropy_bits=round(mean_entropy, 4),
        max_surprisal_token=max_surprisal,
        min_surprisal_token=min_surprisal,
    )


def stream_generate_with_telemetry(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    candidate_top_k: int = 5,
    tokenizer: Any = None,
) -> Generator[TokenTelemetry, None, SequenceTelemetry]:
    """Autoregressively generates tokens while yielding TokenTelemetry on each decode step."""
    model.eval()
    device = next(model.parameters()).device
    idx = idx.to(device)

    collected_tokens: list[TokenTelemetry] = []
    seq_start_time = time.perf_counter()

    max_ctx = getattr(model.config, "max_context_length", 512)

    for step_idx in range(max_new_tokens):
        step_start = time.perf_counter()
        idx_cond = idx if idx.size(1) <= max_ctx else idx[:, -max_ctx:]

        with torch.no_grad():
            logits, _ = model(idx_cond)

        step_logits = logits[0, -1, :]

        if temperature <= 0.0:
            next_token_id = int(torch.argmax(step_logits, dim=-1).item())
        else:
            scaled = step_logits / temperature
            if top_k is not None:
                v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
                scaled = scaled.clone()
                scaled[scaled < v[-1]] = float("-inf")
            sample_probs = F.softmax(scaled, dim=-1)
            next_token_id = int(torch.multinomial(sample_probs, num_samples=1).item())

        step_elapsed_ms = (time.perf_counter() - step_start) * 1000.0

        telemetry = compute_step_telemetry(
            logits=step_logits,
            chosen_token_id=next_token_id,
            index=step_idx,
            latency_ms=step_elapsed_ms,
            tokenizer=tokenizer,
            candidate_top_k=candidate_top_k,
            temperature=temperature,
        )

        collected_tokens.append(telemetry)
        yield telemetry

        next_tok_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
        idx = torch.cat((idx, next_tok_tensor), dim=1)

    total_elapsed_ms = (time.perf_counter() - seq_start_time) * 1000.0
    return aggregate_sequence_telemetry(collected_tokens, total_elapsed_ms)


async def astream_generate_with_telemetry(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    candidate_top_k: int = 5,
    tokenizer: Any = None,
) -> AsyncGenerator[TokenTelemetry, None]:
    """Async generator wrapper for token telemetry streaming in FastAPI SSE endpoints."""
    loop = asyncio.get_running_loop()
    model.eval()
    device = next(model.parameters()).device
    idx = idx.to(device)

    max_ctx = getattr(model.config, "max_context_length", 512)

    for step_idx in range(max_new_tokens):
        step_start = time.perf_counter()
        idx_cond = idx if idx.size(1) <= max_ctx else idx[:, -max_ctx:]

        def _forward(curr_ctx=idx_cond):
            with torch.no_grad():
                logits, _ = model(curr_ctx)
            return logits[0, -1, :]

        step_logits = await loop.run_in_executor(None, _forward)

        if temperature <= 0.0:
            next_token_id = int(torch.argmax(step_logits, dim=-1).item())
        else:
            scaled = step_logits / temperature
            if top_k is not None:
                v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
                scaled = scaled.clone()
                scaled[scaled < v[-1]] = float("-inf")
            sample_probs = F.softmax(scaled, dim=-1)
            next_token_id = int(torch.multinomial(sample_probs, num_samples=1).item())

        step_elapsed_ms = (time.perf_counter() - step_start) * 1000.0

        telemetry = compute_step_telemetry(
            logits=step_logits,
            chosen_token_id=next_token_id,
            index=step_idx,
            latency_ms=step_elapsed_ms,
            tokenizer=tokenizer,
            candidate_top_k=candidate_top_k,
            temperature=temperature,
        )

        yield telemetry

        next_tok_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
        idx = torch.cat((idx, next_tok_tensor), dim=1)
        await asyncio.sleep(0)


def analyze_sequence_telemetry(
    model: torch.nn.Module,
    input_tokens: torch.Tensor,
    tokenizer: Any = None,
    candidate_top_k: int = 5,
) -> SequenceTelemetry:
    """Evaluates an existing sequence under teacher-forcing to calculate surprisal and entropy."""
    model.eval()
    device = next(model.parameters()).device
    input_tokens = input_tokens.to(device)

    _, T = input_tokens.shape
    if T < 2:
        return SequenceTelemetry(
            tokens=[],
            text="",
            total_tokens=0,
            total_duration_ms=0.0,
            tokens_per_second=0.0,
            mean_surprisal_bits=0.0,
            perplexity=1.0,
            mean_entropy_bits=0.0,
        )

    start_time = time.perf_counter()

    with torch.no_grad():
        logits, _ = model(input_tokens)

    tokens: list[TokenTelemetry] = []
    for t in range(1, T):
        step_logits = logits[0, t - 1, :]
        chosen_id = int(input_tokens[0, t].item())

        telemetry = compute_step_telemetry(
            logits=step_logits,
            chosen_token_id=chosen_id,
            index=t - 1,
            latency_ms=0.0,
            tokenizer=tokenizer,
            candidate_top_k=candidate_top_k,
            temperature=1.0,
        )
        tokens.append(telemetry)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return aggregate_sequence_telemetry(tokens, elapsed_ms)
