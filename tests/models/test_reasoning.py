"""
Unit tests for Reasoning Engine: Thought Trace Parser, Self-Consistency, and Best-of-N (Phase 29).
"""

import pytest

from packages.models.reasoning.search_verifier import BestOfNVerifier
from packages.models.reasoning.self_consistency import (
    SelfConsistencyEngine,
    normalize_answer,
)
from packages.models.reasoning.trace_parser import (
    StreamingTraceParser,
    extract_reasoning_steps,
    parse_reasoning_trace,
)


class TestThoughtTraceParser:
    def test_parse_standard_think_tags(self):
        text = (
            "<think>\n"
            "Step 1: Identify the problem.\n"
            "Step 2: Compute 2 + 2 = 4.\n"
            "</think>\n"
            "The final answer is 4."
        )
        trace = parse_reasoning_trace(text, duration_ms=1250.0)
        assert trace.has_thought is True
        assert "Step 1: Identify the problem." in trace.raw_thought
        assert trace.final_answer == "The final answer is 4."
        assert len(trace.steps) >= 2
        assert trace.thinking_duration_ms == 1250.0

    def test_parse_unclosed_think_tag(self):
        text = "<think>\nThinking in progress but interrupted..."
        trace = parse_reasoning_trace(text)
        assert trace.has_thought is True
        assert "Thinking in progress" in trace.raw_thought
        assert trace.final_answer == ""

    def test_parse_scratchpad_convention(self):
        text = "Thought: Let's calculate the square root of 16.\nAnswer: 4"
        trace = parse_reasoning_trace(text)
        assert trace.has_thought is True
        assert "calculate the square root" in trace.raw_thought
        assert trace.final_answer == "4"

    def test_parse_plain_text(self):
        text = "Paris is the capital of France."
        trace = parse_reasoning_trace(text)
        assert trace.has_thought is False
        assert trace.raw_thought == ""
        assert trace.final_answer == text

    def test_streaming_trace_parser_transitions(self):
        parser = StreamingTraceParser()

        t1, a1 = parser.process_chunk("<think>First")
        assert t1 == "First"
        assert a1 is None
        assert parser.state == "thinking"

        t2, a2 = parser.process_chunk(", we plan.</think>Done!")
        assert t2 == ", we plan."
        assert a2 == "Done!"
        assert parser.state == "answering"

        trace = parser.finalize(duration_ms=500.0)
        assert trace.has_thought is True
        assert trace.raw_thought == "First, we plan."
        assert trace.final_answer == "Done!"

    def test_step_extraction(self):
        thought = (
            "1. First understand the constraints.\n"
            "2. Next compute the intermediate value.\n"
            "Finally, verify edge cases."
        )
        steps = extract_reasoning_steps(thought)
        assert len(steps) >= 3


class TestSelfConsistency:
    def test_normalize_answer(self):
        assert normalize_answer("The answer is 42.") == "42"
        assert normalize_answer("Result: 42") == "42"
        assert normalize_answer("Therefore, 42") == "42"
        assert normalize_answer("  42% ") == "42%"
        assert normalize_answer("Droupadi Murmu.") == "droupadi murmu"

    @pytest.mark.asyncio
    async def test_self_consistency_mock_consensus(self):
        engine = SelfConsistencyEngine()
        result = await engine.evaluate(
            prompt="What is 2 + 2?",
            model="libra-mock-v1",
            num_paths=3,
        )
        assert result.total_paths >= 1
        assert result.confidence > 0.0
        assert result.winning_trajectory is not None


class TestBestOfNVerifier:
    @pytest.mark.asyncio
    async def test_best_of_n_selection(self):
        verifier = BestOfNVerifier()
        result = await verifier.search(
            prompt="Write a function to check palindrome.",
            model="libra-mock-v1",
            n_candidates=3,
        )
        assert result.total_candidates >= 1
        assert result.best_score >= 0.0
        assert len(result.candidates) >= 1
        assert result.best_candidate is not None
