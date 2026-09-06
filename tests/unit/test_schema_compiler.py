"""
Tests for SchemaCompiler and SchemaConstraint (packages/core/grammar/schema_compiler.py)
"""

from typing import Optional
from pydantic import BaseModel, Field
from packages.core.grammar.schema_compiler import SchemaCompiler


class UserProfile(BaseModel):
    user_id: int = Field(..., description="Unique user identifier")
    username: str = Field(..., description="Display name")
    is_active: bool = Field(True, description="Account status")
    bio: Optional[str] = Field(None, description="Short biography")


def test_compile_pydantic_model():
    constraint = SchemaCompiler.compile(UserProfile)
    assert constraint.name == "UserProfile"
    assert "user_id" in constraint.properties
    assert "username" in constraint.properties
    assert "user_id" in constraint.required
    assert "username" in constraint.required

    # OpenAI format
    resp_fmt = constraint.to_openai_response_format()
    assert resp_fmt["type"] == "json_schema"
    assert resp_fmt["json_schema"]["strict"] is True

    # Validate valid JSON text
    valid_text = '{"user_id": 101, "username": "alice", "is_active": true}'
    is_valid, err, model = constraint.validate_text(valid_text)
    assert is_valid is True
    assert err is None
    assert isinstance(model, UserProfile)
    assert model.user_id == 101
    assert model.username == "alice"


def test_compile_raw_schema_dict():
    raw_schema = {
        "title": "ConfigSchema",
        "type": "object",
        "properties": {
            "rate": {"type": "number"},
            "enabled": {"type": "boolean"},
        },
        "required": ["rate"],
    }
    constraint = SchemaCompiler.compile(raw_schema)
    assert constraint.name == "ConfigSchema"

    # Missing required field
    is_valid, err, parsed = constraint.validate_text('{"enabled": true}')
    assert is_valid is False
    assert "Missing required property: 'rate'" in err

    # Valid JSON
    is_valid, err, parsed = constraint.validate_text('{"rate": 0.05, "enabled": true}')
    assert is_valid is True
    assert parsed["rate"] == 0.05


def test_generate_system_instruction():
    constraint = SchemaCompiler.compile(UserProfile)
    instruction = constraint.generate_system_instruction()
    assert "<json_schema>" in instruction
    assert "user_id" in instruction
