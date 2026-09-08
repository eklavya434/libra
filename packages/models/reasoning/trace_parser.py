"""
Libra Models - Reasoning Thought Trace Parser & State Machine (Phase 29)
Parses streaming and static Chain-of-Thought reasoning traces, separating internal
<think> reflection steps from user-facing answers.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field


class ReasoningTrace(BaseModel):
    """Structured representation of an executed reasoning trajectory."""

    raw_thought: str = Field(default="", description="Internal reasoning tokens / thought trace")
    steps: list[str] = Field(
        default_factory=list, description="Extracted individual reasoning steps"
    )
    final_answer: str = Field(..., description="Cleaned, user-facing final answer")
    thinking_duration_ms: float = Field(
        default=0.0, description="Duration spent in thinking phase in milliseconds"
    )
    thought_token_count: int = Field(
        default=0, description="Estimated token count of thought trace"
    )
    has_thought: bool = Field(
        default=False, description="Whether an internal thought trace was present"
    )


def extract_reasoning_steps(thought_text: str) -> list[str]:
    """Splits raw reasoning text into discrete logical steps based on standard markers."""
    if not thought_text or not thought_text.strip():
        return []

    lines = [line.strip() for line in thought_text.strip().split("\n") if line.strip()]
    steps: list[str] = []
    current_step: list[str] = []

    # Patterns indicating start of a new step
    step_start_pattern = re.compile(
        r"^(step\s*\d+[:.]?|\d+[.)]|firstly|secondly|thirdly|finally|next,|therefore,|now,|let's|wait,|hmm,)",
        re.IGNORECASE,
    )

    for line in lines:
        if step_start_pattern.match(line):
            if current_step:
                steps.append(" ".join(current_step))
                current_step = []
            current_step.append(line)
        else:
            if current_step:
                current_step.append(line)
            else:
                current_step.append(line)

    if current_step:
        steps.append(" ".join(current_step))

    return steps if steps else [thought_text.strip()]


def parse_reasoning_trace(text: str, duration_ms: float = 0.0) -> ReasoningTrace:
    """Parses a full generation string into thought trace and final answer.

    Handles:
      1. Standard XML tags: <think>...</think>
      2. Markdown scratchpad blocks: ```thought ... ```
      3. Explicit Thought/Answer prefixes: Thought: ... Answer: ...
    """
    if not text:
        return ReasoningTrace(
            raw_thought="",
            steps=[],
            final_answer="",
            thinking_duration_ms=duration_ms,
            thought_token_count=0,
            has_thought=False,
        )

    # 1. Check for XML <think>...</think> tags
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    if think_match:
        thought_content = think_match.group(1).strip()
        final_answer = (text[: think_match.start()] + text[think_match.end() :]).strip()
        steps = extract_reasoning_steps(thought_content)
        thought_tokens = max(1, len(thought_content.split()))
        return ReasoningTrace(
            raw_thought=thought_content,
            steps=steps,
            final_answer=final_answer,
            thinking_duration_ms=duration_ms,
            thought_token_count=thought_tokens,
            has_thought=True,
        )

    # 2. Check for open unclosed <think> tag (e.g. truncated generation)
    open_think_match = re.search(r"<think>(.*)", text, re.DOTALL | re.IGNORECASE)
    if open_think_match:
        thought_content = open_think_match.group(1).strip()
        steps = extract_reasoning_steps(thought_content)
        return ReasoningTrace(
            raw_thought=thought_content,
            steps=steps,
            final_answer="",
            thinking_duration_ms=duration_ms,
            thought_token_count=max(1, len(thought_content.split())),
            has_thought=True,
        )

    # 3. Check for Thought: ... Answer: ... convention
    scratch_match = re.search(
        r"^(?:thought|reasoning):\s*(.*?)(?:\n\s*(?:final\s+answer|answer):\s*(.*))?$",
        text.strip(),
        re.DOTALL | re.IGNORECASE,
    )
    if scratch_match:
        thought_content = scratch_match.group(1).strip()
        final_answer = scratch_match.group(2).strip() if scratch_match.group(2) else ""
        steps = extract_reasoning_steps(thought_content)
        return ReasoningTrace(
            raw_thought=thought_content,
            steps=steps,
            final_answer=final_answer or thought_content,
            thinking_duration_ms=duration_ms,
            thought_token_count=max(1, len(thought_content.split())),
            has_thought=True,
        )

    # No explicit thought trace detected; entire text is the final answer
    return ReasoningTrace(
        raw_thought="",
        steps=[],
        final_answer=text.strip(),
        thinking_duration_ms=0.0,
        thought_token_count=0,
        has_thought=False,
    )


class StreamingTraceParser:
    """State machine that processes streaming token deltas in real-time,

    separating thinking tokens from final answer tokens as they arrive.
    """

    def __init__(self) -> None:
        self.state: str = "initial"  # "initial" -> "thinking" -> "answering"
        self.buffer: str = ""
        self.thought_buffer: list[str] = []
        self.answer_buffer: list[str] = []

    def process_chunk(self, chunk: str) -> tuple[str | None, str | None]:
        """Processes an incoming token delta.

        Returns (thought_delta, answer_delta).
        """
        self.buffer += chunk

        thought_delta: str | None = None
        answer_delta: str | None = None

        if self.state == "initial":
            if "<think>" in self.buffer:
                self.state = "thinking"
                # Flush anything before <think> to answer, anything after to thought
                parts = self.buffer.split("<think>", 1)
                if parts[0]:
                    answer_delta = parts[0]
                    self.answer_buffer.append(parts[0])
                self.buffer = parts[1]
                if "</think>" in self.buffer:
                    self.state = "answering"
                    subparts = self.buffer.split("</think>", 1)
                    thought_delta = subparts[0]
                    self.thought_buffer.append(subparts[0])
                    if subparts[1]:
                        answer_delta = (answer_delta or "") + subparts[1]
                        self.answer_buffer.append(subparts[1])
                    self.buffer = ""
                elif self.buffer:
                    thought_delta = self.buffer
                    self.thought_buffer.append(self.buffer)
                    self.buffer = ""
            elif len(self.buffer) > 10:
                # No <think> tag at beginning, treat as direct answer
                self.state = "answering"
                answer_delta = self.buffer
                self.answer_buffer.append(self.buffer)
                self.buffer = ""

        elif self.state == "thinking":
            if "</think>" in self.buffer:
                self.state = "answering"
                parts = self.buffer.split("</think>", 1)
                thought_delta = parts[0]
                self.thought_buffer.append(parts[0])
                if parts[1]:
                    answer_delta = parts[1]
                    self.answer_buffer.append(parts[1])
                self.buffer = ""
            else:
                thought_delta = self.buffer
                self.thought_buffer.append(self.buffer)
                self.buffer = ""

        elif self.state == "answering":
            answer_delta = self.buffer
            self.answer_buffer.append(self.buffer)
            self.buffer = ""

        return thought_delta, answer_delta

    def finalize(self, duration_ms: float = 0.0) -> ReasoningTrace:
        """Constructs final complete ReasoningTrace from accumulated buffers."""
        raw_thought = "".join(self.thought_buffer).strip()
        final_answer = "".join(self.answer_buffer).strip()

        # Handle any leftover in buffer
        if self.buffer:
            if self.state == "thinking":
                raw_thought += self.buffer
            else:
                final_answer += self.buffer

        steps = extract_reasoning_steps(raw_thought)
        thought_tokens = max(1, len(raw_thought.split())) if raw_thought else 0

        return ReasoningTrace(
            raw_thought=raw_thought,
            steps=steps,
            final_answer=final_answer or raw_thought,
            thinking_duration_ms=duration_ms,
            thought_token_count=thought_tokens,
            has_thought=bool(raw_thought),
        )
