"""
Libra Tools Package - Central Tool Registry

Maintains the catalog of registered tools, exports OpenAI function schemas,
and provides synchronous and asynchronous tool execution dispatchers.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from packages.tools.base import BaseTool, ToolResult
from packages.tools.builtin import (
    CalculatorTool,
    KnowledgeBaseTool,
    PythonInterpreterTool,
    WebSearchTool,
)

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Central registry and dispatcher for all tool definitions."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Registers a tool instance under its tool.name."""
        if not tool.name:
            raise ValueError("Tool name cannot be empty")
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: '{tool.name}'")

    def unregister(self, name: str) -> bool:
        """Removes a tool from registry by name."""
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def get(self, name: str) -> Optional[BaseTool]:
        """Retrieves a registered tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[BaseTool]:
        """Returns all registered tool instances."""
        return list(self._tools.values())

    def get_tool_names(self) -> list[str]:
        """Returns list of registered tool names."""
        return list(self._tools.keys())

    def get_schemas(self) -> list[dict[str, Any]]:
        """Returns OpenAI-compatible tool schemas for all registered tools."""
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def execute_tool(self, name: str, arguments: dict[str, Any] | str) -> ToolResult:
        """
        Executes a registered tool synchronously.
        Accepts arguments as either a Python dictionary or a JSON-encoded string.
        """
        tool = self.get(name)
        if not tool:
            return ToolResult(
                success=False,
                output=None,
                error=f"Tool '{name}' not found. Available tools: {list(self._tools.keys())}",
            )

        parsed_args: dict[str, Any]
        if isinstance(arguments, str):
            try:
                parsed_args = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as err:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Invalid JSON arguments: {err}",
                )
        elif isinstance(arguments, dict):
            parsed_args = arguments
        else:
            return ToolResult(
                success=False,
                output=None,
                error=f"Invalid arguments type: {type(arguments)}. Expected dict or JSON string.",
            )

        return tool.execute(**parsed_args)

    async def execute_tool_async(self, name: str, arguments: dict[str, Any] | str) -> ToolResult:
        """
        Executes a registered tool asynchronously.
        """
        tool = self.get(name)
        if not tool:
            return ToolResult(
                success=False,
                output=None,
                error=f"Tool '{name}' not found. Available tools: {list(self._tools.keys())}",
            )

        parsed_args: dict[str, Any]
        if isinstance(arguments, str):
            try:
                parsed_args = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as err:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Invalid JSON arguments: {err}",
                )
        elif isinstance(arguments, dict):
            parsed_args = arguments
        else:
            return ToolResult(
                success=False,
                output=None,
                error=f"Invalid arguments type: {type(arguments)}. Expected dict or JSON string.",
            )

        return await tool.execute_async(**parsed_args)


_GLOBAL_TOOL_REGISTRY: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Returns the singleton ToolRegistry initialized with default built-in tools."""
    global _GLOBAL_TOOL_REGISTRY
    if _GLOBAL_TOOL_REGISTRY is None:
        registry = ToolRegistry()
        # Register core built-in educational tools
        registry.register(CalculatorTool())
        registry.register(PythonInterpreterTool())
        registry.register(WebSearchTool())
        registry.register(KnowledgeBaseTool())
        _GLOBAL_TOOL_REGISTRY = registry
    return _GLOBAL_TOOL_REGISTRY


def reset_tool_registry() -> None:
    """Resets the singleton ToolRegistry (useful for testing)."""
    global _GLOBAL_TOOL_REGISTRY
    _GLOBAL_TOOL_REGISTRY = None
