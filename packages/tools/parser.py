"""
Libra Tools Package - Tool Call Output Parser

Extracts structured tool calls from raw model text outputs.
Supports multiple model calling styles:
1. XML tag style: <tool_call>{"name": "calculator", "arguments": {"expression": "12*12"}}</tool_call>
2. Markdown codeblock style: ```json {"name": "calculator", ...} ```
3. Direct JSON structure: {"tool": "...", "parameters": {...}} or {"name": "...", "arguments": {...}}
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from pydantic import BaseModel, Field


class ParsedToolCall(BaseModel):
    """Represents a validated tool call extracted from model text."""

    name: str = Field(..., description="Name of the tool to execute")
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="Parsed dictionary of arguments"
    )
    call_id: Optional[str] = Field(None, description="Optional unique tool call identifier")
    raw_text: str = Field("", description="The raw segment from which this call was extracted")


class ToolCallParser:
    """Robust parser for locating and decoding tool calls in LLM generation streams."""

    XML_PATTERN = re.compile(
        r"<(?:tool_call|tool)>(.*?)</(?:tool_call|tool)>",
        re.DOTALL | re.IGNORECASE,
    )
    CODEBLOCK_PATTERN = re.compile(
        r"```(?:json)?\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```",
        re.DOTALL | re.IGNORECASE,
    )

    @classmethod
    def parse(cls, text: str) -> list[ParsedToolCall]:
        """
        Parses text for tool call patterns. Returns a list of ParsedToolCall instances.
        """
        if not text or not text.strip():
            return []

        calls: list[ParsedToolCall] = []

        # 1. Look for XML <tool_call>...</tool_call> tags
        for match in cls.XML_PATTERN.finditer(text):
            raw_content = match.group(1).strip()
            parsed = cls._try_parse_json_or_dict(raw_content, match.group(0))
            if parsed:
                calls.extend(parsed)

        if calls:
            return calls

        # 2. Look for JSON code fences: ```json { ... } ```
        for match in cls.CODEBLOCK_PATTERN.finditer(text):
            raw_content = match.group(1).strip()
            parsed = cls._try_parse_json_or_dict(raw_content, match.group(0))
            if parsed:
                calls.extend(parsed)

        if calls:
            return calls

        # 3. Direct JSON string if whole text is valid JSON or contains JSON object
        raw_trimmed = text.strip()
        if (raw_trimmed.startswith("{") and raw_trimmed.endswith("}")) or (
            raw_trimmed.startswith("[") and raw_trimmed.endswith("]")
        ):
            parsed = cls._try_parse_json_or_dict(raw_trimmed, raw_trimmed)
            if parsed:
                calls.extend(parsed)

        return calls

    @classmethod
    def _try_parse_json_or_dict(cls, json_str: str, raw_segment: str) -> list[ParsedToolCall]:
        """Attempts to parse a JSON segment into one or more ParsedToolCall objects."""
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # Attempt minor repair if single quotes were used
            try:
                data = json.loads(json_str.replace("'", '"'))
            except Exception:
                return []

        results: list[ParsedToolCall] = []

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    call = cls._dict_to_tool_call(item, raw_segment)
                    if call:
                        results.append(call)
        elif isinstance(data, dict):
            call = cls._dict_to_tool_call(data, raw_segment)
            if call:
                results.append(call)

        return results

    @classmethod
    def _dict_to_tool_call(cls, d: dict[str, Any], raw_segment: str) -> Optional[ParsedToolCall]:
        """Normalizes various LLM tool invocation dictionary keys."""
        # Detect tool name key: 'name', 'tool', 'tool_name', 'function'
        name = d.get("name") or d.get("tool") or d.get("tool_name")
        if not name and isinstance(d.get("function"), dict):
            name = d["function"].get("name")
            args = d["function"].get("arguments", {})
        else:
            # Detect arguments key: 'arguments', 'parameters', 'args', 'params', 'input'
            args = (
                d.get("arguments")
                or d.get("parameters")
                or d.get("args")
                or d.get("params")
                or d.get("input")
                or {}
            )

        if not name or not isinstance(name, str):
            return None

        # Parse string arguments if model serialized inner JSON
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"raw_input": args}
        elif not isinstance(args, dict):
            args = {"value": args}

        call_id = d.get("id") or d.get("call_id") or d.get("tool_call_id")

        return ParsedToolCall(
            name=name.strip(),
            arguments=args,
            call_id=str(call_id) if call_id else None,
            raw_text=raw_segment,
        )

    @classmethod
    def strip_tool_calls(cls, text: str) -> str:
        """Removes tool call markup from model output, leaving conversational text."""
        cleaned = cls.XML_PATTERN.sub("", text)
        cleaned = cls.CODEBLOCK_PATTERN.sub("", cleaned)
        return cleaned.strip()
