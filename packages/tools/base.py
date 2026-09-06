"""
Libra Tools Package - Base Tool Abstractions & Schemas

Defines the abstract BaseTool class, ToolResult schema, and OpenAI-compatible
function schema generator.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, Optional, Type
from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Encapsulates the execution output of a tool call."""

    success: bool = Field(..., description="Whether the tool executed without error")
    output: Any = Field(None, description="The returned data or result string")
    error: Optional[str] = Field(None, description="Error message if execution failed")
    execution_time_ms: float = Field(0.0, description="Duration of tool execution in milliseconds")

    def to_chat_message(self, tool_call_id: Optional[str] = None) -> dict[str, Any]:
        """Formats the result as a standard OpenAI tool response message."""
        content_str = str(self.output) if self.success else f"Error: {self.error}"
        msg: dict[str, Any] = {
            "role": "tool",
            "content": content_str,
        }
        if tool_call_id:
            msg["tool_call_id"] = tool_call_id
        return msg


class BaseTool(ABC):
    """
    Abstract Base Class for all Libra Tools.
    Provides standard metadata, validation, schema export, and execution handlers.
    """

    name: str = ""
    description: str = ""
    args_schema: Optional[Type[BaseModel]] = None

    def to_openai_schema(self) -> dict[str, Any]:
        """Generates OpenAI-compatible function definition schema."""
        parameters: dict[str, Any]
        if self.args_schema is not None:
            schema = self.args_schema.model_json_schema()
            # Clean pydantic specific internal fields
            parameters = {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", []),
            }
        else:
            parameters = {
                "type": "object",
                "properties": {},
            }

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }

    @abstractmethod
    def _run(self, **kwargs: Any) -> Any:
        """Core synchronous execution logic to be implemented by child classes."""
        pass

    async def _arun(self, **kwargs: Any) -> Any:
        """Core asynchronous execution logic (defaults to running _run in threadpool)."""
        return await asyncio.to_thread(self._run, **kwargs)

    def execute(self, **kwargs: Any) -> ToolResult:
        """Executes tool synchronously with timing and error isolation."""
        start = time.perf_counter()
        try:
            # Validate against Pydantic schema if present
            if self.args_schema:
                validated = self.args_schema(**kwargs)
                kwargs = validated.model_dump()

            res = self._run(**kwargs)
            duration_ms = (time.perf_counter() - start) * 1000
            return ToolResult(
                success=True,
                output=res,
                execution_time_ms=round(duration_ms, 2),
            )
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            return ToolResult(
                success=False,
                error=str(e),
                execution_time_ms=round(duration_ms, 2),
            )

    async def execute_async(self, **kwargs: Any) -> ToolResult:
        """Executes tool asynchronously with timing and error isolation."""
        start = time.perf_counter()
        try:
            if self.args_schema:
                validated = self.args_schema(**kwargs)
                kwargs = validated.model_dump()

            res = await self._arun(**kwargs)
            duration_ms = (time.perf_counter() - start) * 1000
            return ToolResult(
                success=True,
                output=res,
                execution_time_ms=round(duration_ms, 2),
            )
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            return ToolResult(
                success=False,
                error=str(e),
                execution_time_ms=round(duration_ms, 2),
            )
