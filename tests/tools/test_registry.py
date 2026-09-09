"""
Tests for ToolRegistry (packages/tools/registry.py)

Validates tool registration, OpenAI schema generation, argument decoding,
and async/sync dispatching.
"""

import pytest
from pydantic import BaseModel, Field

from packages.tools.base import BaseTool
from packages.tools.registry import ToolRegistry, get_tool_registry, reset_tool_registry


class EchoArgs(BaseModel):
    message: str = Field(..., description="Message to echo back")
    repeat: int = Field(1, description="Number of times to repeat message")


class DummyEchoTool(BaseTool):
    name = "echo_tool"
    description = "Echoes a message multiple times"
    args_schema = EchoArgs

    def _run(self, message: str, repeat: int = 1) -> str:
        return " ".join([message] * repeat)


@pytest.fixture(autouse=True)
def clean_registry():
    reset_tool_registry()
    yield
    reset_tool_registry()


def test_registry_register_and_get():
    reg = ToolRegistry()
    tool = DummyEchoTool()
    reg.register(tool)

    assert reg.get("echo_tool") is tool
    assert "echo_tool" in reg.get_tool_names()
    assert len(reg.list_tools()) == 1

    unreg = reg.unregister("echo_tool")
    assert unreg is True
    assert reg.get("echo_tool") is None


def test_registry_schemas_generation():
    reg = ToolRegistry()
    reg.register(DummyEchoTool())

    schemas = reg.get_schemas()
    assert len(schemas) == 1
    s = schemas[0]
    assert s["type"] == "function"
    assert s["function"]["name"] == "echo_tool"
    assert "message" in s["function"]["parameters"]["properties"]
    assert "repeat" in s["function"]["parameters"]["properties"]


def test_registry_execution_with_dict():
    reg = ToolRegistry()
    reg.register(DummyEchoTool())

    res = reg.execute_tool("echo_tool", {"message": "hello", "repeat": 3})
    assert res.success is True
    assert res.output == "hello hello hello"
    assert res.execution_time_ms >= 0


def test_registry_execution_with_json_string():
    reg = ToolRegistry()
    reg.register(DummyEchoTool())

    res = reg.execute_tool("echo_tool", '{"message": "libra", "repeat": 2}')
    assert res.success is True
    assert res.output == "libra libra"


@pytest.mark.asyncio
async def test_registry_execution_async():
    reg = ToolRegistry()
    reg.register(DummyEchoTool())

    res = await reg.execute_tool_async("echo_tool", {"message": "async_ok", "repeat": 1})
    assert res.success is True
    assert res.output == "async_ok"


def test_registry_unknown_tool_and_invalid_args():
    reg = ToolRegistry()
    res = reg.execute_tool("non_existent", {})
    assert res.success is False
    assert "not found" in res.error

    reg.register(DummyEchoTool())
    res_bad_json = reg.execute_tool("echo_tool", "not a json string {")
    assert res_bad_json.success is False
    assert "Invalid JSON arguments" in res_bad_json.error


def test_global_tool_registry_prepopulated():
    global_reg = get_tool_registry()
    names = global_reg.get_tool_names()
    assert "calculator" in names
    assert "python_interpreter" in names
    assert "web_search" in names
    assert "knowledge_base_search" in names
