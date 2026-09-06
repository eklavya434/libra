import pytest
from packages.providers.prompt_template import PromptTemplate


def test_format_chatml():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    formatted = PromptTemplate.format_chatml(messages, add_generation_prompt=True)
    assert "<|im_start|>system\nYou are a helpful assistant.<|im_end|>" in formatted
    assert "<|im_start|>user\nHello!<|im_end|>" in formatted
    assert formatted.endswith("<|im_start|>assistant\n")


def test_format_llama3():
    messages = [
        {"role": "user", "content": "Tell me a joke."},
    ]
    formatted = PromptTemplate.format_llama3(messages, add_generation_prompt=True)
    assert "<|begin_of_text|>" in formatted
    assert "<|start_header_id|>user<|end_header_id|>\n\nTell me a joke.<|eot_id|>" in formatted
    assert formatted.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")


def test_format_plain():
    messages = [
        {"role": "user", "content": "What is gravity?"},
    ]
    formatted = PromptTemplate.format_plain(messages, add_generation_prompt=True)
    assert "User: What is gravity?" in formatted
    assert formatted.endswith("Assistant: ")


def test_stop_sequences():
    chatml_stops = PromptTemplate.get_stop_sequences("chatml")
    assert "<|im_end|>" in chatml_stops

    llama_stops = PromptTemplate.get_stop_sequences("llama3")
    assert "<|eot_id|>" in llama_stops
