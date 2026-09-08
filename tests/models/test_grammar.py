"""
Tests for Phase 27: Constrained Decoding & Grammar Masking
Verifies Thompson NFA Regex Automaton, Earley CFG Parser, GrammarLogitsProcessor, and generation.
"""

import torch

from packages.core.grammar.cfg_parser import CFGGrammar
from packages.core.grammar.grammar_processor import GrammarLogitsProcessor
from packages.core.grammar.regex_automaton import RegexAutomaton
from packages.models.grammar_generation import generate_with_grammar
from packages.models.modern_config import ModernTransformerConfig
from packages.models.modern_transformer import ModernTransformerLM


def test_regex_automaton_digits_and_quantifiers():
    """Verifies Thompson NFA correctly handles digits, ranges, and character classes."""
    # Pattern: 3 digits, dash, 2 digits (e.g. 123-45)
    automaton = RegexAutomaton(r"\d{3}-\d{2}")

    assert automaton.is_valid_prefix("1")
    assert automaton.is_valid_prefix("12")
    assert automaton.is_valid_prefix("123")
    assert automaton.is_valid_prefix("123-")
    assert automaton.is_valid_prefix("123-4")
    assert automaton.is_valid_prefix("123-45")

    # Invalid continuations
    assert not automaton.is_valid_prefix("a")
    assert not automaton.is_valid_prefix("1234")
    assert not automaton.is_valid_prefix("123-456")

    # Full acceptance
    assert automaton.is_accepted("123-45")
    assert not automaton.is_accepted("123-4")
    assert not automaton.is_accepted("123-")


def test_regex_automaton_alternation_and_wildcard():
    """Verifies alternation (cat|dog) and wildcard repetition."""
    automaton = RegexAutomaton(r"(red|blue)_(circle|square)")

    assert automaton.is_valid_prefix("r")
    assert automaton.is_valid_prefix("red")
    assert automaton.is_valid_prefix("red_")
    assert automaton.is_valid_prefix("blue_s")
    assert automaton.is_valid_prefix("blue_square")

    assert not automaton.is_valid_prefix("green")
    assert not automaton.is_valid_prefix("red_triangle")

    assert automaton.is_accepted("red_circle")
    assert automaton.is_accepted("blue_square")
    assert not automaton.is_accepted("red_")


def test_cfg_earley_parser_arithmetic():
    """Verifies that the Earley parser validates arithmetic context-free grammars."""
    rules = """
    root -> expr
    expr -> expr "+" term | expr "-" term | term
    term -> term "*" factor | term "/" factor | factor
    factor -> "(" expr ")" | NUMBER
    """
    cfg = CFGGrammar(rules)

    # Valid prefixes
    assert cfg.is_valid_prefix(["NUMBER"])
    assert cfg.is_valid_prefix(["NUMBER", "+"])
    assert cfg.is_valid_prefix(["NUMBER", "+", "NUMBER"])
    assert cfg.is_valid_prefix(["(", "NUMBER", "+", "NUMBER", ")"])
    assert cfg.is_valid_prefix(["(", "NUMBER"])

    # Invalid prefixes
    assert not cfg.is_valid_prefix(["+", "NUMBER"])
    assert not cfg.is_valid_prefix(["NUMBER", ")"])
    assert not cfg.is_valid_prefix(["*", "NUMBER"])

    # Acceptance
    assert cfg.is_accepted(["NUMBER"])
    assert cfg.is_accepted(["NUMBER", "+", "NUMBER"])
    assert cfg.is_accepted(["(", "NUMBER", "*", "NUMBER", ")"])
    assert not cfg.is_accepted(["NUMBER", "+"])
    assert not cfg.is_accepted(["("])


def test_cfg_next_terminals_lookup():
    """Verifies next terminal lookahead calculation."""
    rules = """
    root -> expr
    expr -> expr "+" term | term
    term -> NUMBER
    """
    cfg = CFGGrammar(rules)

    # At start, only NUMBER is valid
    start_next = cfg.get_valid_next_terminals([])
    assert "NUMBER" in start_next
    assert "+" not in start_next

    # After NUMBER, '+' is valid
    after_num_next = cfg.get_valid_next_terminals(["NUMBER"])
    assert "+" in after_num_next


def test_grammar_logits_processor_masking():
    """Verifies that GrammarLogitsProcessor masks invalid tokens to -inf."""
    # Toy vocab
    vocab = ["<eos>", "a", "b", "1", "2", "_"]
    decode_fn = lambda ids: "".join(vocab[i] for i in ids)

    # Grammar: must start with 'a' followed by digits
    regex = RegexAutomaton(r"a[0-9]+")
    processor = GrammarLogitsProcessor(
        grammar=regex,
        decode_fn=decode_fn,
        vocab_size=len(vocab),
        eos_token_id=0,
    )

    # Step 0: empty prefix -> only 'a' (index 1) should be permitted
    scores = torch.zeros(len(vocab))
    masked_0 = processor(torch.tensor([], dtype=torch.long), scores)
    assert masked_0[1] == 0.0  # 'a'
    assert masked_0[0] == -float("inf")  # <eos>
    assert masked_0[2] == -float("inf")  # 'b'
    assert masked_0[3] == -float("inf")  # '1'

    # Step 1: prefix 'a' -> digits '1' and '2' should be permitted
    prefix_ids = torch.tensor([1], dtype=torch.long)
    masked_1 = processor(prefix_ids, scores)
    assert masked_1[1] == -float("inf")  # 'a'
    assert masked_1[3] == 0.0  # '1'
    assert masked_1[4] == 0.0  # '2'
    assert masked_1[0] == -float("inf")  # <eos> not yet accepted

    # Step 2: prefix 'a1' -> accepted! Digits AND <eos> are permitted
    prefix_ids_2 = torch.tensor([1, 3], dtype=torch.long)
    masked_2 = processor(prefix_ids_2, scores)
    assert masked_2[0] == 0.0  # <eos> allowed
    assert masked_2[3] == 0.0  # '1' allowed
    assert masked_2[1] == -float("inf")  # 'a' disallowed


def test_grammar_constrained_generation():
    """Verifies end-to-end autoregressive generation with strict regex masking."""
    torch.manual_seed(42)
    config = ModernTransformerConfig(
        vocab_size=256,
        d_model=32,
        n_heads=2,
        n_layers=2,
        max_context_length=64,
    )
    model = ModernTransformerLM(config)
    model.eval()

    # Regex: strictly digits only
    regex = RegexAutomaton(r"[0-9]+")
    decode_fn = lambda ids: bytes([i for i in ids if 0 <= i < 256]).decode(
        "utf-8", errors="replace"
    )

    processor = GrammarLogitsProcessor(
        grammar=regex,
        decode_fn=decode_fn,
        vocab_size=256,
        eos_token_id=0,
    )

    prompt = torch.tensor([[ord("5")]], dtype=torch.long)
    _, generated_text, telemetry = generate_with_grammar(
        model=model,
        idx=prompt,
        processor=processor,
        max_new_tokens=6,
        temperature=0.7,
        eos_token_id=0,
    )

    # The generated text MUST be strictly digits!
    assert len(generated_text) > 0
    assert generated_text.isdigit()
    assert telemetry["token_count"] > 0
