"""
Libra Evaluation - Model Comparison Arena Engine (Phase 10)

Runs side-by-side model completions concurrently, measuring TTFT, total latency,
token generation throughput (tok/s), and financial token expenditure.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

from packages.evaluation.metrics import ArenaModelMetric
from packages.providers.cost import calculate_cost
from packages.providers.router import ProviderRouter, get_router


class ModelComparisonArena:
    """Orchestrates concurrent model benchmarking and side-by-side evaluation."""

    def __init__(self, router: Optional[ProviderRouter] = None) -> None:
        self.router = router or get_router()

    async def benchmark_single_model(
        self,
        model: str,
        prompt: str,
        provider_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 256,
        system_prompt: Optional[str] = None,
    ) -> ArenaModelMetric:
        """Executes a single model request, measuring TTFT and generation throughput."""
        try:
            prov = await self.router.resolve_provider_for_model(
                model_id=model,
                requested_provider=provider_name,
            )
        except Exception as e:
            return ArenaModelMetric(
                model=model,
                provider=provider_name or "unknown",
                success=False,
                output_text="",
                ttft_ms=None,
                total_latency_ms=0.0,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                tokens_per_second=0.0,
                cost_usd=0.0,
                is_free=True,
                error=f"Provider resolution failed: {str(e)}",
            )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start_time = time.perf_counter()
        first_token_time: Optional[float] = None
        collected_tokens: list[str] = []

        try:
            # Try streaming first to accurately measure TTFT
            async for chunk in prov.stream(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                if first_token_time is None and chunk:
                    first_token_time = time.perf_counter()
                collected_tokens.append(chunk)

            end_time = time.perf_counter()
            total_latency_ms = (end_time - start_time) * 1000.0
            ttft_ms = (first_token_time - start_time) * 1000.0 if first_token_time else total_latency_ms

            output_text = "".join(collected_tokens)
            # Estimate token counts: approx 1 token per 4 characters if not provided
            prompt_tokens = max(1, len(prompt) // 4)
            completion_tokens = max(1, len(output_text) // 4)
            total_tokens = prompt_tokens + completion_tokens

            elapsed_sec = max(0.001, (end_time - start_time))
            tok_per_sec = completion_tokens / elapsed_sec

            cost_info = calculate_cost(model, prompt_tokens, completion_tokens)

            return ArenaModelMetric(
                model=model,
                provider=prov.name,
                success=True,
                output_text=output_text,
                ttft_ms=ttft_ms,
                total_latency_ms=total_latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                tokens_per_second=tok_per_sec,
                cost_usd=cost_info["total_cost_usd"],
                is_free=cost_info["is_free"],
                error=None,
            )
        except Exception as stream_err:
            # Fallback to non-streaming chat() if streaming failed or unsupported
            try:
                chat_start = time.perf_counter()
                resp = await prov.chat(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                chat_end = time.perf_counter()
                total_latency_ms = (chat_end - chat_start) * 1000.0

                output_text = ""
                choices = resp.get("choices", [])
                if choices:
                    output_text = choices[0].get("message", {}).get("content", "")

                usage = resp.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens") or max(1, len(prompt) // 4)
                completion_tokens = usage.get("completion_tokens") or max(1, len(output_text) // 4)
                total_tokens = prompt_tokens + completion_tokens

                elapsed_sec = max(0.001, (chat_end - chat_start))
                tok_per_sec = completion_tokens / elapsed_sec

                cost_info = resp.get("cost") or calculate_cost(model, prompt_tokens, completion_tokens)

                return ArenaModelMetric(
                    model=model,
                    provider=prov.name,
                    success=True,
                    output_text=output_text,
                    ttft_ms=total_latency_ms,  # TTFT equal to total latency for non-streaming
                    total_latency_ms=total_latency_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    tokens_per_second=tok_per_sec,
                    cost_usd=cost_info.get("total_cost_usd", 0.0),
                    is_free=cost_info.get("is_free", True),
                    error=None,
                )
            except Exception as final_err:
                return ArenaModelMetric(
                    model=model,
                    provider=prov.name,
                    success=False,
                    output_text="",
                    ttft_ms=None,
                    total_latency_ms=(time.perf_counter() - start_time) * 1000.0,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    tokens_per_second=0.0,
                    cost_usd=0.0,
                    is_free=True,
                    error=str(final_err),
                )


    async def compare(
        self,
        prompt: str,
        models: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 256,
        system_prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        """Compares multiple models concurrently, isolating faults and computing summary rankings."""
        tasks = []
        for m in models:
            model_id = m.get("model") if isinstance(m, dict) else str(m)
            provider_id = m.get("provider") if isinstance(m, dict) else None
            tasks.append(
                self.benchmark_single_model(
                    model=model_id,
                    prompt=prompt,
                    provider_name=provider_id,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_prompt=system_prompt,
                )
            )

        results: list[ArenaModelMetric] = await asyncio.gather(*tasks)
        metric_dicts = [r.to_dict() for r in results]

        # Determine winner metrics among successful runs
        successful = [r for r in results if r.success]
        fastest_ttft = min(successful, key=lambda x: x.ttft_ms or float("inf")).model if successful and any(x.ttft_ms is not None for x in successful) else None
        highest_throughput = max(successful, key=lambda x: x.tokens_per_second).model if successful else None
        lowest_cost = min(successful, key=lambda x: x.cost_usd).model if successful else None

        return {
            "prompt": prompt,
            "total_models": len(models),
            "successful_models": len(successful),
            "results": metric_dicts,
            "rankings": {
                "fastest_ttft": fastest_ttft,
                "highest_throughput": highest_throughput,
                "lowest_cost": lowest_cost,
            },
        }

