"""
Tests for Streaming Markdown & Code Fence Extraction Logic
"""

import re
import pytest


def parse_think_block(content: str) -> tuple[str | None, str, bool]:
    """Simulates StreamingMarkdown think block extraction."""
    if not content.startswith("<think>"):
        return None, content, False

    think_end = content.indexOf("</think>") if hasattr(content, "indexOf") else content.find("</think>")
    if think_end != -1:
        think = content[7:think_end].strip()
        body = content[think_end + 8 :].strip()
        return think, body, False
    else:
        think = content[7:].strip()
        return think, "", True


def extract_code_segments(text: str) -> list[dict]:
    """Simulates StreamingMarkdown segment splitting."""
    segments = []
    lines = text.split("\n")
    in_code = False
    current_lang = ""
    current_code = []
    current_md = []

    for line in lines:
        if line.strip().startswith("```"):
            if not in_code:
                if current_md:
                    segments.append({"type": "markdown", "content": "\n".join(current_md)})
                    current_md = []
                in_code = True
                current_lang = line.strip()[3:].strip() or "text"
                current_code = []
            else:
                segments.append({
                    "type": "code",
                    "language": current_lang,
                    "code": "\n".join(current_code),
                    "is_complete": True,
                })
                in_code = False
                current_lang = ""
                current_code = []
        else:
            if in_code:
                current_code.append(line)
            else:
                current_md.append(line)

    if in_code:
        segments.append({
            "type": "code",
            "language": current_lang,
            "code": "\n".join(current_code),
            "is_complete": False,
        })
    elif current_md:
        segments.append({"type": "markdown", "content": "\n".join(current_md)})

    return segments


def test_closed_code_fence_extraction():
    text = "Here is Python code:\n```python\nx = 42\nprint(x)\n```\nAll done."
    segments = extract_code_segments(text)
    assert len(segments) == 3
    assert segments[0]["type"] == "markdown"
    assert segments[1]["type"] == "code"
    assert segments[1]["language"] == "python"
    assert segments[1]["is_complete"] is True
    assert "print(x)" in segments[1]["code"]
    assert segments[2]["type"] == "markdown"


def test_in_flight_open_code_fence():
    # Streaming arrives mid-code block without closing fence
    text = "Generating response:\n```typescript\nconst token = 'active';\n"
    segments = extract_code_segments(text)
    assert len(segments) == 2
    assert segments[1]["type"] == "code"
    assert segments[1]["language"] == "typescript"
    assert segments[1]["is_complete"] is False
    assert "const token = 'active';" in segments[1]["code"]


def test_think_block_extraction():
    complete_text = "<think>\nEvaluate user query against attention math.\n</think>\nAttention is Scaled Dot-Product."
    think, body, in_flight = parse_think_block(complete_text)
    assert think == "Evaluate user query against attention math."
    assert body == "Attention is Scaled Dot-Product."
    assert in_flight is False

    partial_text = "<think>\nAnalyzing prompt..."
    think_part, body_part, in_flight_part = parse_think_block(partial_text)
    assert think_part == "Analyzing prompt..."
    assert body_part == ""
    assert in_flight_part is True
