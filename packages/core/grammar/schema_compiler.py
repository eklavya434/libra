"""
Libra Core Grammar Package - JSON Schema Compiler

Compiles Pydantic models or JSON Schema specifications into constrained decoding
rules, provider response_formats, and system prompt instructions.
"""

from __future__ import annotations

import json
from typing import Any, Optional, Type

from pydantic import BaseModel, ValidationError


class SchemaConstraint:
    """Compiled schema constraint containing properties, required keys, and validator."""

    def __init__(
        self,
        schema: dict[str, Any],
        pydantic_model: Optional[Type[BaseModel]] = None,
        name: str = "StructuredResponse",
    ) -> None:
        self.schema = schema
        self.pydantic_model = pydantic_model
        self.name = name
        self.properties = schema.get("properties", {})
        self.required = schema.get("required", [])

    def validate_text(self, text: str) -> tuple[bool, Optional[str], Optional[Any]]:
        """
        Validates JSON text against schema.
        Returns (is_valid, error_message, parsed_object_or_model).
        """
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as err:
            return False, f"Invalid JSON syntax: {err}", None

        if self.pydantic_model is not None:
            try:
                validated = self.pydantic_model.model_validate(parsed)
                return True, None, validated
            except ValidationError as v_err:
                return False, f"Schema validation error: {v_err}", parsed

        # Basic JSON Schema dictionary validation
        if isinstance(parsed, dict):
            for req in self.required:
                if req not in parsed:
                    return False, f"Missing required property: '{req}'", parsed
            return True, None, parsed

        return False, "Expected JSON object at root", parsed

    def to_openai_response_format(self) -> dict[str, Any]:
        """Generates OpenAI/Ollama compatible json_schema response_format."""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": self.name,
                "strict": True,
                "schema": self.schema,
            },
        }

    def generate_system_instruction(self) -> str:
        """Generates an explicit system prompt directing the LLM to output valid JSON matching schema."""
        schema_json = json.dumps(self.schema, indent=2)
        return (
            "You MUST respond ONLY with a single, valid JSON object strictly conforming to the following JSON Schema.\n"
            "Do NOT wrap in markdown fences or include explanatory text before or after the JSON.\n"
            f"<json_schema>\n{schema_json}\n</json_schema>"
        )


class SchemaCompiler:
    """Utility compiler for translating Pydantic classes and raw schemas into SchemaConstraint."""

    @classmethod
    def compile(
        cls,
        target: Type[BaseModel] | dict[str, Any],
        name: Optional[str] = None,
    ) -> SchemaConstraint:
        """Compiles a Pydantic model class or raw JSON schema dict."""
        if isinstance(target, type) and issubclass(target, BaseModel):
            schema_dict = target.model_json_schema()
            model_name = name or target.__name__
            return SchemaConstraint(
                schema=schema_dict,
                pydantic_model=target,
                name=model_name,
            )
        elif isinstance(target, dict):
            return SchemaConstraint(
                schema=target,
                pydantic_model=None,
                name=name or target.get("title", "StructuredResponse"),
            )
        else:
            raise TypeError(
                f"Target must be a Pydantic BaseModel subclass or dict, got: {type(target)}"
            )
