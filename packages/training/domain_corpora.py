"""
Libra Training - Domain Adaptation Educational Corpora (Phase 33)
Contains synthetic high-quality instruction datasets covering:
1. LLM Systems & Transformer Mechanics
2. Python Coder & Debugger
3. Mathematical Reasoning
"""

from packages.training.chat_formatter import ChatMessage


def get_educational_instruction_corpus() -> list[list[ChatMessage]]:
    """Returns educational domain instruction dialogue samples."""
    return [
        [
            ChatMessage("system", "You are Libra, an educational AI assistant."),
            ChatMessage("user", "What is causal masking in a transformer?"),
            ChatMessage(
                "assistant",
                "Causal masking sets upper-triangular attention scores to negative infinity before softmax, "
                "preventing tokens from attending to subsequent future positions during autoregressive generation."
            ),
        ],
        [
            ChatMessage("system", "You are an expert Python engineer."),
            ChatMessage("user", "Fix this code: def add(a, b): return a - b"),
            ChatMessage(
                "assistant",
                "The function subtracts instead of adding. The corrected code is:\n\n"
                "```python\ndef add(a, b):\n    return a + b\n```"
            ),
        ],
        [
            ChatMessage("system", "You are a mathematical reasoning tutor."),
            ChatMessage("user", "Calculate the dot product of [1, 2] and [3, 4]."),
            ChatMessage(
                "assistant",
                "The dot product is (1 * 3) + (2 * 4) = 3 + 8 = 11."
            ),
        ],
        [
            ChatMessage("system", "You are Libra, an educational AI assistant."),
            ChatMessage("user", "Why do we use RoPE instead of absolute positional embeddings?"),
            ChatMessage(
                "assistant",
                "Rotary Position Embedding (RoPE) encodes relative distance naturally via 2D rotation of Query and Key "
                "vectors, allowing models to generalize to longer context lengths without retraining."
            ),
        ],
    ]
