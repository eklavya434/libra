"""
Libra API v1 - Structured Output & Grammar Endpoints

Provides endpoints for generating guaranteed schema-compliant JSON data
and validating raw generation outputs against JSON Schemas.
"""

from __future__ import annotations

from typing import Any, Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field

from packages.core.grammar import SchemaCompiler
from packages.providers import StructuredOutputGenerator, StructuredResult

router = APIRouter()


class StructuredGenerateRequest(BaseModel):
    """Request payload to generate structured output conforming to a JSON schema."""

    prompt: str = Field(..., description="User prompt describing the requested information")
    schema_dict: dict[str, Any] = Field(..., description="Target JSON schema specification")
    model_id: str = Field("mock-model", description="Model identifier to use for generation")
    provider_name: Optional[str] = Field(None, description="Optional provider override")
    system_prompt: Optional[str] = Field(None, description="Optional custom system prompt")
    temperature: float = Field(0.0, ge=0.0, le=2.0, description="Sampling temperature")
    max_retries: int = Field(2, ge=0, le=5, description="Maximum self-healing repair attempts")


class ValidateStructuredRequest(BaseModel):
    """Request to validate an existing JSON string against a schema."""

    text: str = Field(..., description="Raw string or JSON to validate")
    schema_dict: dict[str, Any] = Field(..., description="Target JSON schema")


class ValidateStructuredResponse(BaseModel):
    """Validation response report."""

    is_valid: bool
    error: Optional[str] = None
    parsed_data: Optional[Any] = None


@router.post("/generate", response_model=StructuredResult[Any], summary="Generate schema-constrained output")
async def generate_structured(request: StructuredGenerateRequest) -> StructuredResult[Any]:
    """Generate structured response guaranteed to parse against schema_dict with self-healing fallback."""
    generator = StructuredOutputGenerator()
    result = await generator.generate(
        prompt=request.prompt,
        schema=request.schema_dict,
        model_id=request.model_id,
        provider_name=request.provider_name,
        system_prompt=request.system_prompt,
        max_retries=request.max_retries,
        temperature=request.temperature,
    )
    return result


@router.post("/validate", response_model=ValidateStructuredResponse, summary="Validate text against JSON schema")
async def validate_structured(request: ValidateStructuredRequest) -> ValidateStructuredResponse:
    """Check whether a text string satisfies a target JSON schema."""
    constraint = SchemaCompiler.compile(request.schema_dict)
    is_valid, err, parsed = constraint.validate_text(request.text)
    return ValidateStructuredResponse(
        is_valid=is_valid,
        error=err,
        parsed_data=parsed,
    )
