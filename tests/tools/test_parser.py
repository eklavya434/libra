"""
Tests for ToolCallParser (packages/tools/parser.py)

Validates extraction of tool calls across XML tags, markdown blocks,
and raw JSON, as well as conversational text stripping.
"""

from packages.tools.parser import ToolCallParser


def test_parse_xml_tool_call():
    text = """
To find the square root of 144, I will use the calculator:
<tool_call>
{
    "name": "calculator",
    "arguments": {
        "expression": "sqrt(144)"
    }
}
</tool_call>
Let me compute this now.
"""
    calls = ToolCallParser.parse(text)
    assert len(calls) == 1
    call = calls[0]
    assert call.name == "calculator"
    assert call.arguments["expression"] == "sqrt(144)"


def test_parse_xml_tool_shorthand():
    text = """<tool>{"tool": "python_interpreter", "parameters": {"code": "print(42)"}}</tool>"""
    calls = ToolCallParser.parse(text)
    assert len(calls) == 1
    assert calls[0].name == "python_interpreter"
    assert calls[0].arguments["code"] == "print(42)"


def test_parse_codeblock_tool_call():
    text = """
I will search for the latest documentation:
```json
{
    "name": "web_search",
    "arguments": {
        "query": "PyTorch 2.0 release notes",
        "max_results": 3
    }
}
```
"""
    calls = ToolCallParser.parse(text)
    assert len(calls) == 1
    assert calls[0].name == "web_search"
    assert calls[0].arguments["query"] == "PyTorch 2.0 release notes"
    assert calls[0].arguments["max_results"] == 3


def test_parse_raw_json():
    text = '{"name": "calculator", "arguments": {"expression": "25 * 4"}}'
    calls = ToolCallParser.parse(text)
    assert len(calls) == 1
    assert calls[0].name == "calculator"
    assert calls[0].arguments["expression"] == "25 * 4"


def test_parse_multiple_tool_calls_in_list():
    text = """
```json
[
    {"name": "calculator", "arguments": {"expression": "100 / 4"}},
    {"name": "calculator", "arguments": {"expression": "50 * 2"}}
]
```
"""
    calls = ToolCallParser.parse(text)
    assert len(calls) == 2
    assert calls[0].arguments["expression"] == "100 / 4"
    assert calls[1].arguments["expression"] == "50 * 2"


def test_parse_malformed_and_empty():
    assert ToolCallParser.parse("") == []
    assert ToolCallParser.parse("Just normal response with no tools.") == []
    assert ToolCallParser.parse("<tool_call>Not a valid json</tool_call>") == []


def test_strip_tool_calls():
    text = (
        "I will calculate this:\n"
        "<tool_call>{\"name\": \"calculator\", \"arguments\": {\"expression\": \"2+2\"}}</tool_call>\n"
        "Here is the final text."
    )
    cleaned = ToolCallParser.strip_tool_calls(text)
    assert "<tool_call>" not in cleaned
    assert "calculator" not in cleaned
    assert "I will calculate this:" in cleaned
    assert "Here is the final text." in cleaned
