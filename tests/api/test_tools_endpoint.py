"""
Tests for Tools REST API Endpoints (apps/backend/api/v1/endpoints/tools.py)
"""

import pytest
from fastapi.testclient import TestClient

from apps.backend.main import app
from packages.tools.registry import reset_tool_registry


@pytest.fixture(autouse=True)
def clean_registry():
    reset_tool_registry()
    yield
    reset_tool_registry()


@pytest.fixture
def client():
    return TestClient(app)


def test_list_tools(client):
    res = client.get("/api/v1/tools")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    names = [t["name"] for t in data]
    assert "calculator" in names
    assert "python_interpreter" in names
    assert "web_search" in names
    assert "knowledge_base_search" in names


def test_get_schemas(client):
    res = client.get("/api/v1/tools/schemas")
    assert res.status_code == 200
    schemas = res.json()
    assert isinstance(schemas, list)
    assert len(schemas) >= 4
    for schema in schemas:
        assert schema["type"] == "function"
        assert "name" in schema["function"]
        assert "description" in schema["function"]


def test_execute_calculator_tool(client):
    res = client.post(
        "/api/v1/tools/execute",
        json={"name": "calculator", "arguments": {"expression": "25 * 4"}},
    )
    assert res.status_code == 200
    result = res.json()
    assert result["success"] is True
    assert result["output"]["result"] == 100
    assert result["execution_time_ms"] >= 0


def test_execute_python_sandbox_tool(client):
    code = "import math\nval = math.sqrt(64)\nprint(f'Computed {val}')\nval"
    res = client.post(
        "/api/v1/tools/execute",
        json={"name": "python_interpreter", "arguments": {"code": code}},
    )
    assert res.status_code == 200
    result = res.json()
    assert result["success"] is True
    assert result["output"]["result"] == 8.0
    assert "Computed 8.0" in result["output"]["stdout"]


def test_execute_python_sandbox_blocks_os(client):
    code = "import os\nos.system('dir')"
    res = client.post(
        "/api/v1/tools/execute",
        json={"name": "python_interpreter", "arguments": {"code": code}},
    )
    assert res.status_code == 200
    result = res.json()
    assert result["success"] is False
    assert "Security policy violation" in result["error"]


def test_execute_unknown_tool(client):
    res = client.post(
        "/api/v1/tools/execute",
        json={"name": "mystery_tool", "arguments": {}},
    )
    assert res.status_code == 404
    assert "not found" in res.json()["detail"]


def test_parse_tool_endpoint(client):
    text = '<tool_call>{"name": "calculator", "arguments": {"expression": "3 * 7"}}</tool_call>'
    res = client.post(
        "/api/v1/tools/parse",
        json={"text": text},
    )
    assert res.status_code == 200
    calls = res.json()
    assert len(calls) == 1
    assert calls[0]["name"] == "calculator"
    assert calls[0]["arguments"]["expression"] == "3 * 7"
