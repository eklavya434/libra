"""
Libra Core Grammar Package - Incremental JSON State Machine

Implements a pushdown automaton (PDA) that tracks partial JSON strings
character-by-character to determine valid next characters, prefix validity,
and completion status for constrained autoregressive decoding.
"""

from __future__ import annotations

import json
from enum import Enum, auto
from typing import Optional, Set


class JSONState(Enum):
    START = auto()                 # Before root value
    IN_OBJECT_START = auto()       # Right after '{'
    EXPECT_KEY = auto()            # Expecting string key
    IN_KEY_STRING = auto()         # Inside key "..."
    AFTER_KEY = auto()             # Key ended, expecting ':'
    EXPECT_VALUE = auto()          # After ':' or array start/comma, expecting value
    IN_VALUE_STRING = auto()       # Inside string value "..."
    IN_NUMBER = auto()             # Inside numeric literal
    IN_LITERAL = auto()            # Inside true/false/null
    AFTER_VALUE = auto()           # After value, expecting ',' or '}' / ']'
    EXPECT_COMMA_OR_END = auto()   # Inside object/array after value
    DONE = auto()                  # Root value closed, only whitespace allowed
    INVALID = auto()               # Unrecoverable syntax error


DIGITS = set("0123456789")
NUMBER_START_CHARS = DIGITS | {"-"}
NUMBER_CHARS = DIGITS | {".", "e", "E", "+", "-"}
WHITESPACE = {" ", "\t", "\n", "\r"}


class IncrementalJSONStateMachine:
    """
    Pushdown Automaton tracking partial JSON generation.
    Maintains a stack of nested contexts ('OBJECT', 'ARRAY') and the current state.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.stack: list[str] = []         # Context stack: 'OBJECT' or 'ARRAY'
        self.state: JSONState = JSONState.START
        self.in_escape: bool = False       # True if preceding char was '\'
        self.literal_buffer: str = ""      # Buffer for true/false/null or numbers
        self.expected_literal: str = ""    # Target string: 'true', 'false', 'null'

    def is_valid_prefix(self, text: str) -> bool:
        """Returns True if text is a valid prefix that can potentially complete to valid JSON."""
        temp = IncrementalJSONStateMachine()
        for ch in text:
            if not temp.feed_char(ch):
                return False
        return temp.state != JSONState.INVALID

    def feed_char(self, ch: str) -> bool:
        """
        Transitions the state machine with a single character.
        Returns False if the character violates the JSON grammar.
        """
        # Handle string escape sequences
        if self.state in (JSONState.IN_KEY_STRING, JSONState.IN_VALUE_STRING):
            if self.in_escape:
                # Valid escape character
                if ch in {'"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u'}:
                    self.in_escape = False
                    return True
                return False
            elif ch == '\\':
                self.in_escape = True
                return True
            elif ch == '"':
                # End of string
                if self.state == JSONState.IN_KEY_STRING:
                    self.state = JSONState.AFTER_KEY
                else:
                    self._complete_value()
                return True
            else:
                # Regular string character (reject control chars)
                if ord(ch) < 0x20:
                    return False
                return True

        # Handle number literal continuation
        if self.state == JSONState.IN_NUMBER:
            if ch in NUMBER_CHARS:
                self.literal_buffer += ch
                return True
            else:
                # Number terminated; validate the number buffer
                if not self._is_valid_number(self.literal_buffer):
                    return False
                self._complete_value()
                # Re-evaluate ch from the new state
                return self.feed_char(ch)

        # Handle literal (true, false, null) continuation
        if self.state == JSONState.IN_LITERAL:
            self.literal_buffer += ch
            if not self.expected_literal.startswith(self.literal_buffer):
                return False
            if self.literal_buffer == self.expected_literal:
                self._complete_value()
            return True

        # Whitespace handling outside literals/strings
        if ch in WHITESPACE:
            return True

        # State dispatch
        if self.state == JSONState.START:
            return self._start_value(ch)

        if self.state == JSONState.IN_OBJECT_START:
            if ch == '}':
                # Empty object
                self.stack.pop()
                self._complete_value()
                return True
            elif ch == '"':
                self.state = JSONState.IN_KEY_STRING
                return True
            return False

        if self.state == JSONState.EXPECT_KEY:
            if ch == '"':
                self.state = JSONState.IN_KEY_STRING
                return True
            return False

        if self.state == JSONState.AFTER_KEY:
            if ch == ':':
                self.state = JSONState.EXPECT_VALUE
                return True
            return False

        if self.state == JSONState.EXPECT_VALUE:
            return self._start_value(ch)

        if self.state == JSONState.EXPECT_COMMA_OR_END:
            if not self.stack:
                return False
            context = self.stack[-1]
            if context == "OBJECT":
                if ch == '}':
                    self.stack.pop()
                    self._complete_value()
                    return True
                elif ch == ',':
                    self.state = JSONState.EXPECT_KEY
                    return True
            elif context == "ARRAY":
                if ch == ']':
                    self.stack.pop()
                    self._complete_value()
                    return True
                elif ch == ',':
                    self.state = JSONState.EXPECT_VALUE
                    return True
            return False

        if self.state == JSONState.DONE:
            # Only whitespace permitted after root completion
            return ch in WHITESPACE

        return False

    def _start_value(self, ch: str) -> bool:
        """Starts parsing a JSON value (string, object, array, number, boolean, null)."""
        if ch == '{':
            self.stack.append("OBJECT")
            self.state = JSONState.IN_OBJECT_START
            return True
        elif ch == '[':
            self.stack.append("ARRAY")
            self.state = JSONState.EXPECT_VALUE
            return True
        elif ch == '"':
            self.state = JSONState.IN_VALUE_STRING
            return True
        elif ch in NUMBER_START_CHARS:
            self.state = JSONState.IN_NUMBER
            self.literal_buffer = ch
            return True
        elif ch == 't':
            self.state = JSONState.IN_LITERAL
            self.expected_literal = "true"
            self.literal_buffer = "t"
            return True
        elif ch == 'f':
            self.state = JSONState.IN_LITERAL
            self.expected_literal = "false"
            self.literal_buffer = "f"
            return True
        elif ch == 'n':
            self.state = JSONState.IN_LITERAL
            self.expected_literal = "null"
            self.literal_buffer = "n"
            return True
        elif ch == ']' and self.stack and self.stack[-1] == "ARRAY":
            # Empty array []
            self.stack.pop()
            self._complete_value()
            return True
        return False

    def _complete_value(self) -> None:
        """Transitions state when a value (or object/array) has completed."""
        self.literal_buffer = ""
        self.expected_literal = ""
        if not self.stack:
            self.state = JSONState.DONE
        else:
            self.state = JSONState.EXPECT_COMMA_OR_END

    def _is_valid_number(self, s: str) -> bool:
        """Validates that a string buffer represents a valid JSON number."""
        try:
            float(s)
            # Prevent leading zeros like '01' unless '0' or '0.x'
            if len(s) > 1 and s.startswith("0") and s[1] not in (".", "e", "E"):
                return False
            if len(s) > 2 and s.startswith("-0") and s[2] not in (".", "e", "E"):
                return False
            return True
        except ValueError:
            return False

    def is_complete(self) -> bool:
        """Returns True if the current parsed stream represents a complete, closed JSON value."""
        if self.state == JSONState.IN_NUMBER:
            return not self.stack and self._is_valid_number(self.literal_buffer)
        return self.state == JSONState.DONE and not self.stack

    def get_allowed_characters(self) -> Set[str]:
        """Returns the set of characters allowed at the current state."""
        allowed: Set[str] = set()

        if self.state in (JSONState.IN_KEY_STRING, JSONState.IN_VALUE_STRING):
            if self.in_escape:
                return {'"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u'}
            # Allow common printable ASCII characters
            allowed.update([chr(i) for i in range(32, 127)])
            return allowed

        if self.state == JSONState.IN_NUMBER:
            allowed.update(NUMBER_CHARS)
            # Also allow characters that can legally terminate a number
            if self._is_valid_number(self.literal_buffer):
                allowed.update(WHITESPACE)
                if self.stack:
                    if self.stack[-1] == "OBJECT":
                        allowed.update({",", "}"})
                    elif self.stack[-1] == "ARRAY":
                        allowed.update({",", "]"})
            return allowed

        if self.state == JSONState.IN_LITERAL:
            idx = len(self.literal_buffer)
            if idx < len(self.expected_literal):
                return {self.expected_literal[idx]}
            return set()

        # Whitespace is allowed between tokens
        allowed.update(WHITESPACE)

        if self.state in (JSONState.START, JSONState.EXPECT_VALUE):
            allowed.update({'{', '[', '"', 't', 'f', 'n'})
            allowed.update(NUMBER_START_CHARS)
            if self.state == JSONState.EXPECT_VALUE and self.stack and self.stack[-1] == "ARRAY":
                allowed.add(']')
            return allowed

        if self.state == JSONState.IN_OBJECT_START:
            allowed.update({'"', '}'})
            return allowed

        if self.state == JSONState.EXPECT_KEY:
            allowed.add('"')
            return allowed

        if self.state == JSONState.AFTER_KEY:
            allowed.add(':')
            return allowed

        if self.state == JSONState.EXPECT_COMMA_OR_END:
            if self.stack:
                if self.stack[-1] == "OBJECT":
                    allowed.update({',', '}'})
                elif self.stack[-1] == "ARRAY":
                    allowed.update({',', ']'})
            return allowed

        return allowed
