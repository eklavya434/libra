"""
Libra Providers Package - Structured Output Generation Engine

Coordinates schema-constrained decoding, JSON response_formats, and
iterative self-healing repair loops across local and cloud providers.
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any, Generic, Optional, Type, TypeVar

from pydantic import BaseModel, Field

from packages.core.grammar.schema_compiler import SchemaCompiler, SchemaConstraint
from packages.providers.base import BaseProvider
from packages.providers.router import ProviderRouter, get_router

T = TypeVar("T")


class StructuredResult(BaseModel, Generic[T]):
    """Structured generation result container with metadata and parsed schema."""

    success: bool = Field(..., description="Whether generation produced a valid schema instance")
    data: Optional[Any] = Field(None, description="Parsed Pydantic instance or JSON dict")
    raw_text: str = Field("", description="Raw response text emitted by the model")
    attempts: int = Field(
        1, description="Number of attempts taken (including self-healing retries)"
    )
    error: Optional[str] = Field(None, description="Validation error message if generation failed")
    latency_ms: float = Field(0.0, description="Total duration in milliseconds")


def extract_json_from_text(text: str) -> str:
    """Extracts JSON object substring if model wrapped it in markdown code fences or commentary."""
    text_clean = text.strip()
    # Check for ```json ... ``` codeblocks
    fence_match = re.search(
        r"```(?:json)?\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```", text_clean, re.DOTALL
    )
    if fence_match:
        return fence_match.group(1).strip()

    # Check for outermost { ... }
    first_brace = text_clean.find("{")
    last_brace = text_clean.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        return text_clean[first_brace : last_brace + 1].strip()

    first_bracket = text_clean.find("[")
    last_bracket = text_clean.rfind("]")
    if first_bracket != -1 and last_bracket > first_bracket:
        return text_clean[first_bracket : last_bracket + 1].strip()

    return text_clean


class StructuredOutputGenerator:
    """
    Coordinates structured schema-constrained output generation with
    automatic validation and self-healing repair loops.
    """

    def __init__(self, router: Optional[ProviderRouter] = None) -> None:
        self.router = router or get_router()

    async def generate(
        self,
        prompt: str,
        schema: Type[BaseModel] | dict[str, Any],
        model_id: str = "mock-model",
        provider_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_retries: int = 2,
        temperature: float = 0.0,
    ) -> StructuredResult[Any]:
        """
        Generates structured data strictly conforming to schema.
        Automatically repairs malformed outputs via feedback reflection.
        """
        t0 = time.perf_counter()
        constraint: SchemaConstraint = SchemaCompiler.compile(schema)
        provider: BaseProvider = await self.router.resolve_provider_for_model(
            model_id=model_id,
            requested_provider=provider_name,
        )

        system_instruction = constraint.generate_system_instruction()
        combined_system = (
            f"{system_prompt}\n\n{system_instruction}" if system_prompt else system_instruction
        )
        response_format = constraint.to_openai_response_format()

        # Multi-turn repair history
        messages: list[dict[str, str]] = [
            {"role": "system", "content": combined_system},
            {"role": "user", "content": prompt},
        ]

        last_raw = ""
        last_error = ""

        for attempt in range(1, max_retries + 2):
            try:
                # Dispatch completion to provider (supporting both chat and complete interfaces)
                if hasattr(provider, "chat"):
                    res = await provider.chat(
                        messages=messages,
                        model=model_id,
                        temperature=temperature,
                        response_format=response_format,
                    )
                    if isinstance(res, dict) and "choices" in res:
                        raw_content = res["choices"][0]["message"]["content"]
                    elif hasattr(res, "content"):
                        raw_content = res.content
                    else:
                        raw_content = str(res)
                elif hasattr(provider, "complete"):
                    res = await provider.complete(
                        messages=messages,
                        model_id=model_id,
                        temperature=temperature,
                        response_format=response_format,
                    )
                    raw_content = res.content if hasattr(res, "content") else str(res)
                else:
                    raise AttributeError(
                        f"Provider {provider.name} does not implement chat or complete"
                    )

                last_raw = raw_content

                extracted_json = extract_json_from_text(raw_content)
                is_valid, err_msg, parsed_obj = constraint.validate_text(extracted_json)

                if is_valid:
                    duration_ms = (time.perf_counter() - t0) * 1000
                    return StructuredResult(
                        success=True,
                        data=parsed_obj,
                        raw_text=raw_content,
                        attempts=attempt,
                        error=None,
                        latency_ms=round(duration_ms, 2),
                    )

                last_error = err_msg or "Unknown validation failure"

            except Exception as ex:
                last_error = f"Provider error: {ex}"

            # Self-healing feedback step if retries remain
            if attempt <= max_retries:
                messages.append({"role": "assistant", "content": last_raw})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your previous response produced a schema validation error:\n{last_error}\n\n"
                            "Please correct the error and reply with ONLY the corrected, valid JSON object."
                        ),
                    }
                )

        duration_ms = (time.perf_counter() - t0) * 1000
        return StructuredResult(
            success=False,
            data=None,
            raw_text=last_raw,
            attempts=max_retries + 1,
            error=last_error,
            latency_ms=round(duration_ms, 2),
        )

    def generate_sync(
        self,
        prompt: str,
        schema: Type[BaseModel] | dict[str, Any],
        model_id: str = "mock-model",
        provider_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_retries: int = 2,
        temperature: float = 0.0,
    ) -> StructuredResult[Any]:
        """Synchronous convenience wrapper for generate."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.generate(
                    prompt=prompt,
                    schema=schema,
                    model_id=model_id,
                    provider_name=provider_name,
                    system_prompt=system_prompt,
                    max_retries=max_retries,
                    temperature=temperature,
                )
            )

        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(
                asyncio.run,
                self.generate(
                    prompt=prompt,
                    schema=schema,
                    model_id=model_id,
                    provider_name=provider_name,
                    system_prompt=system_prompt,
                    max_retries=max_retries,
                    temperature=temperature,
                ),
            ).result()
