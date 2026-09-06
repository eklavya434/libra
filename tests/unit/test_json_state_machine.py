"""
Tests for IncrementalJSONStateMachine (packages/core/grammar/json_state_machine.py)

Validates pushdown automaton transitions, prefix validity, character allowlists,
and completion detection for JSON generation.
"""

import pytest
from packages.core.grammar.json_state_machine import IncrementalJSONStateMachine


@pytest.fixture
def pda():
    return IncrementalJSONStateMachine()


def test_pda_valid_objects(pda):
    valid_json = '{"name": "Libra", "count": 42, "active": true, "items": [1, 2, 3]}'
    for ch in valid_json:
        assert pda.feed_char(ch) is True
    assert pda.is_complete() is True


def test_pda_empty_structures(pda):
    # Empty object
    for ch in "{}":
        assert pda.feed_char(ch) is True
    assert pda.is_complete() is True

    # Empty array
    pda.reset()
    for ch in "[]":
        assert pda.feed_char(ch) is True
    assert pda.is_complete() is True


def test_pda_prefix_validity(pda):
    assert pda.is_valid_prefix('{"name":') is True
    assert pda.is_valid_prefix('{"name": "test", "num": 12') is True
    assert pda.is_valid_prefix('{"items": [true, false,') is True
    assert pda.is_valid_prefix('{"score": -42.5e2}') is True

    # Invalid prefixes
    assert pda.is_valid_prefix('{"name": bad_value}') is False
    assert pda.is_valid_prefix('{123: "key"}') is False
    assert pda.is_valid_prefix('{"name": "test"}}') is False


def test_pda_allowed_characters(pda):
    pda.reset()
    # At start, can open object, array, or primitives
    allowed = pda.get_allowed_characters()
    assert '{' in allowed
    assert '[' in allowed
    assert '"' in allowed

    # Feed '{'
    pda.feed_char('{')
    allowed = pda.get_allowed_characters()
    assert '"' in allowed
    assert '}' in allowed
    assert ':' not in allowed

    # Feed '"key"'
    for ch in '"key"':
        pda.feed_char(ch)
    allowed = pda.get_allowed_characters()
    assert ':' in allowed
    assert '{' not in allowed

    # Feed ':'
    pda.feed_char(':')
    allowed = pda.get_allowed_characters()
    assert '"' in allowed
    assert '{' in allowed
    assert '[' in allowed
    assert 't' in allowed  # true
    assert 'f' in allowed  # false


def test_pda_string_escapes(pda):
    # Test escaped quotes inside strings
    escaped_json = r'{"msg": "Hello \"world\" and \n newline"}'
    for ch in escaped_json:
        assert pda.feed_char(ch) is True
    assert pda.is_complete() is True
