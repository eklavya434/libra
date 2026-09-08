"""
Tests for API v1 Grammar & Constrained Decoding Endpoints
Verifies /grammar/generate, /grammar/validate, and /grammar/next_tokens.
"""

from fastapi.testclient import TestClient

from apps.backend.main import app

client = TestClient(app)


def test_grammar_validate_regex_endpoint():
    """Verifies that /grammar/validate correctly evaluates regex strings."""
    payload = {
        "grammar_type": "regex",
        "grammar_spec": r"\d{3}-\d{2}",
        "candidate_text": "123-45",
    }
    response = client.post("/api/v1/grammar/validate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_valid_prefix"] is True
    assert data["is_accepted"] is True

    # Invalid prefix
    payload_invalid = {
        "grammar_type": "regex",
        "grammar_spec": r"\d{3}-\d{2}",
        "candidate_text": "abc",
    }
    res_inv = client.post("/api/v1/grammar/validate", json=payload_invalid)
    assert res_inv.status_code == 200
    assert res_inv.json()["is_valid_prefix"] is False


def test_grammar_validate_cfg_endpoint():
    """Verifies that /grammar/validate evaluates CFG expressions."""
    rules = """
    root -> expr
    expr -> expr "+" term | term
    term -> NUMBER
    """
    payload = {
        "grammar_type": "cfg",
        "grammar_spec": rules,
        "candidate_text": "NUMBER + NUMBER",
    }
    response = client.post("/api/v1/grammar/validate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_valid_prefix"] is True
    assert data["is_accepted"] is True


def test_grammar_generate_endpoint():
    """Verifies that /grammar/generate produces text conforming to the grammar."""
    payload = {
        "prompt": "5",
        "grammar_type": "regex",
        "grammar_spec": r"[0-9]+",
        "max_tokens": 5,
        "temperature": 0.5,
    }
    response = client.post("/api/v1/grammar/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "generated_text" in data
    assert data["grammar_type"] == "regex"
    assert "telemetry" in data
    # Text should be strictly digits
    if data["generated_text"]:
        assert data["generated_text"].isdigit()


def test_grammar_next_tokens_endpoint():
    """Verifies that /grammar/next_tokens inspects permitted next tokens."""
    payload = {
        "grammar_type": "regex",
        "grammar_spec": r"[a-z]+",
        "prefix_text": "",
    }
    response = client.post("/api/v1/grammar/next_tokens", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "allowed_token_count" in data
    assert data["allowed_token_count"] > 0
    assert "allowed_tokens" in data
