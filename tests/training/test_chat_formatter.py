"""
Tests for ChatML Formatter and Prompt Loss Masker (Phase 33)
"""

from packages.training.chat_formatter import (
    ChatMessage,
    ChatMLFormatter,
    tokenize_with_loss_masking,
    IGNORE_INDEX,
)


def test_chatml_format_turn():
    formatter = ChatMLFormatter()
    msg = ChatMessage(role="user", content="Hello world")
    formatted = formatter.format_turn(msg)
    assert formatted == "<|im_start|>user\nHello world<|im_end|>\n"


def test_chatml_format_conversation():
    formatter = ChatMLFormatter()
    dialogue = [
        ChatMessage(role="system", content="You are Libra."),
        ChatMessage(role="user", content="What is 2+2?"),
        ChatMessage(role="assistant", content="4"),
    ]
    formatted = formatter.format_conversation(dialogue)
    expected = (
        "<|im_start|>system\nYou are Libra.<|im_end|>\n"
        "<|im_start|>user\nWhat is 2+2?<|im_end|>\n"
        "<|im_start|>assistant\n4<|im_end|>\n"
    )
    assert formatted == expected


def test_extract_role_spans():
    formatter = ChatMLFormatter()
    dialogue = [
        ChatMessage(role="user", content="Hi"),
        ChatMessage(role="assistant", content="Hello"),
    ]
    spans = formatter.extract_role_spans(dialogue)
    assert len(spans) == 2
    assert spans[0][0] == "user"
    assert spans[0][1] == 0
    assert spans[1][0] == "assistant"
    assert spans[1][1] == spans[0][2]


def test_tokenize_with_loss_masking():
    dialogue = [
        ChatMessage(role="system", content="System instruction"),
        ChatMessage(role="user", content="User question"),
        ChatMessage(role="assistant", content="Assistant response"),
    ]

    def mock_tokenizer(text: str) -> list[int]:
        return [ord(c) % 256 for c in text]

    input_ids, label_ids = tokenize_with_loss_masking(
        messages=dialogue,
        tokenizer=mock_tokenizer,
        max_length=512,
        ignore_index=IGNORE_INDEX,
    )

    assert len(input_ids) == len(label_ids)
    assert len(input_ids) > 0

    formatter = ChatMLFormatter()
    sys_tokens_len = len(mock_tokenizer(formatter.format_turn(dialogue[0])))
    user_tokens_len = len(mock_tokenizer(formatter.format_turn(dialogue[1])))
    prompt_len = sys_tokens_len + user_tokens_len

    assert all(label == IGNORE_INDEX for label in label_ids[:prompt_len])
    assistant_tokens = label_ids[prompt_len:]
    assert all(label != IGNORE_INDEX for label in assistant_tokens)
    assert assistant_tokens == input_ids[prompt_len:]
