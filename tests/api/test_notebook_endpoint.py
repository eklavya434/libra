"""
Integration tests for Notebook FastAPI endpoints.

Note: arbitrary code execution is DISABLED by default on shared deployments
(LIBRA_PUBLIC_CODE_EXEC=false). The execution tests flip the flag on to
exercise the kernel behavior, and separately assert the default gate.
"""

import pytest
from fastapi.testclient import TestClient

import apps.backend.api.v1.endpoints.notebook as notebook_endpoint
from apps.backend.main import app

client = TestClient(app)


@pytest.fixture
def enable_code_exec(monkeypatch):
    monkeypatch.setattr(notebook_endpoint.settings, "libra_public_code_exec_enabled", True)
    yield


def test_api_create_and_list_sessions():
    # Create session
    resp = client.post("/api/v1/notebook/sessions", json={"session_id": "test_api_session"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "test_api_session"
    assert data["execution_count"] == 0

    # List sessions
    list_resp = client.get("/api/v1/notebook/sessions")
    assert list_resp.status_code == 200
    sids = [s["session_id"] for s in list_resp.json()]
    assert "test_api_session" in sids


def test_api_execution_gated_off_by_default():
    resp = client.post(
        "/api/v1/notebook/sessions/default_gate/execute",
        json={"code": "print(1)"},
    )
    assert resp.status_code == 503
    assert "disabled" in resp.json()["detail"]


def test_api_execute_cell_and_persist_state(enable_code_exec):
    session_id = "test_exec_api"

    # Cell 1
    resp1 = client.post(
        f"/api/v1/notebook/sessions/{session_id}/execute",
        json={"code": "val_a = 50\nval_b = 30"},
    )
    assert resp1.status_code == 200
    d1 = resp1.json()
    assert d1["status"] == "ok"
    assert d1["execution_count"] == 1

    # Cell 2: depends on Cell 1
    resp2 = client.post(
        f"/api/v1/notebook/sessions/{session_id}/execute",
        json={"code": "val_c = val_a + val_b\nval_c"},
    )
    assert resp2.status_code == 200
    d2 = resp2.json()
    assert d2["status"] == "ok"
    assert d2["result"] == "80"
    assert d2["execution_count"] == 2

    # Check variable inspection
    vars_resp = client.get(f"/api/v1/notebook/sessions/{session_id}/variables")
    assert vars_resp.status_code == 200
    v_map = {v["name"]: v["value_repr"] for v in vars_resp.json()}
    assert v_map["val_c"] == "80"

    # Reset
    reset_resp = client.post(f"/api/v1/notebook/sessions/{session_id}/reset")
    assert reset_resp.status_code == 200
    assert reset_resp.json()["execution_count"] == 0


def test_api_presets():
    resp = client.get("/api/v1/notebook/presets")
    assert resp.status_code == 200
    presets = resp.json()
    assert len(presets) >= 3
    assert any("financial" in p["id"] for p in presets)


def test_api_security_error_response(enable_code_exec):
    resp = client.post(
        "/api/v1/notebook/sessions/sec_test/execute",
        json={"code": "import subprocess\nsubprocess.call(['whoami'])"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "error"
    assert "Security policy violation" in data["error_message"]
