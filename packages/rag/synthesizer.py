"""
Libra RAG Package - Prompt Synthesizer

Formats retrieved semantic search results into structured, grounded prompts
with source citations for LLM generation.
"""

from __future__ import annotations

from typing import Optional

from packages.rag.models import SearchResult


class RAGPromptSynthesizer:
    """Combines user questions and retrieved document passages into grounded prompts."""

    DEFAULT_SYSTEM_INSTRUCTION = (
        "You are Libra, an educational AI assistant with Retrieval-Augmented Generation (RAG). "
        "Answer the user's question accurately using ONLY the provided reference context. "
        "If the context does not contain the answer, state that clearly and do not hallucinate facts."
    )

    def __init__(self, system_instruction: Optional[str] = None) -> None:
        self.system_instruction = system_instruction or self.DEFAULT_SYSTEM_INSTRUCTION

    def build_context_block(self, results: list[SearchResult]) -> str:
        """Formats matched chunks with clear source labels and rank citations."""
        if not results:
            return "No relevant context documents found."

        context_lines: list[str] = []
        for res in results:
            header = f"[Source {res.rank}: {res.chunk.doc_title} (Relevance: {res.score:.2f})]"
            context_lines.append(f"{header}\n{res.chunk.text.strip()}")

        return "\n\n".join(context_lines)

    def build_grounded_prompt(self, query: str, results: list[SearchResult]) -> str:
        """Constructs an augmented single-turn instruction prompt."""
        context_block = self.build_context_block(results)
        return (
            f"=== REFERENCE CONTEXT ===\n"
            f"{context_block}\n\n"
            f"=== USER QUESTION ===\n"
            f"{query}\n\n"
            f"=== GROUNDED ANSWER ==="
        )

    def build_chat_messages(
        self,
        query: str,
        results: list[SearchResult],
        conversation_history: Optional[list[dict[str, str]]] = None,
    ) -> list[dict[str, str]]:
        """Constructs a multi-turn chat message array with injected system prompt and grounded user turn."""
        messages: list[dict[str, str]] = []

        # System message with grounding instructions
        messages.append({"role": "system", "content": self.system_instruction})

        # Append prior conversation history if any (excluding prior system messages)
        if conversation_history:
            for turn in conversation_history:
                if turn.get("role") != "system":
                    messages.append(turn)

        # Grounded user prompt
        augmented_user_content = self.build_grounded_prompt(query, results)
        messages.append({"role": "user", "content": augmented_user_content})

        return messages
