"""
Libra Core Grammar Package - Unified Grammar Logits Processor
Supports drop-in logit masking for Regular Expressions, Context-Free Grammars (EBNF),
and JSON state machines during autoregressive generation.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from packages.core.grammar.cfg_parser import CFGGrammar
from packages.core.grammar.json_state_machine import IncrementalJSONStateMachine
from packages.core.grammar.regex_automaton import RegexAutomaton


class GrammarLogitsProcessor:
    """Universal autoregressive logits processor for grammar-constrained generation.

    Evaluates candidate vocabulary tokens at each step and sets invalid tokens to -inf.
    Supports:
    - RegexAutomaton: Arbitrary regular expressions via Thompson NFA.
    - CFGGrammar: Context-free grammars via Earley chart parser.
    - IncrementalJSONStateMachine: Structured JSON schema validation.
    """

    def __init__(
        self,
        grammar: RegexAutomaton | CFGGrammar | IncrementalJSONStateMachine,
        decode_fn: Callable[[list[int]], str],
        vocab_size: int,
        eos_token_id: int | None = None,
    ) -> None:
        self.grammar = grammar
        self.decode_fn = decode_fn
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id

        # Pre-decode all vocabulary tokens to strings for fast matching
        self.token_strings: list[str] = []
        for token_id in range(vocab_size):
            try:
                decoded = self.decode_fn([token_id])
                self.token_strings.append(decoded if isinstance(decoded, str) else "")
            except (AttributeError, ValueError, UnicodeDecodeError):
                self.token_strings.append("")

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        prefix_offset: int = 0,
    ) -> torch.FloatTensor:
        """Masks scores based on the valid continuations allowed by the grammar."""
        if scores.dim() == 1:
            tokens = input_ids[prefix_offset:].tolist() if input_ids.numel() > prefix_offset else []
            return self._process_single(tokens, scores)

        batch_size = scores.size(0)
        output_scores = scores.clone()

        for b in range(batch_size):
            tokens = (
                input_ids[b, prefix_offset:].tolist()
                if input_ids.dim() == 2 and input_ids.size(1) > prefix_offset
                else []
            )
            output_scores[b] = self._process_single(tokens, scores[b])

        return output_scores

    def _process_single(
        self,
        generated_tokens: list[int],
        single_scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Processes logits for a single sequence."""
        prefix_text = self.decode_fn(generated_tokens) if generated_tokens else ""
        masked_scores = single_scores.clone()
        mask = torch.full_like(masked_scores, -float("inf"))
        valid_count = 0

        # Case 1: RegexAutomaton
        if isinstance(self.grammar, RegexAutomaton):
            # Check if current prefix is already an accepted complete match
            is_accepted = self.grammar.is_accepted(prefix_text)
            can_accept_more = self.grammar.can_accept_more(prefix_text)

            # If accepted, allow EOS
            if is_accepted and self.eos_token_id is not None:
                mask[self.eos_token_id] = 0.0
                valid_count += 1
                # If cannot accept more, force EOS immediately
                if not can_accept_more:
                    return masked_scores + mask

            # Evaluate which candidate tokens legally extend prefix_text
            for token_id, token_str in enumerate(self.token_strings):
                if not token_str or token_id == self.eos_token_id:
                    continue
                candidate = prefix_text + token_str
                if self.grammar.is_valid_prefix(candidate):
                    mask[token_id] = 0.0
                    valid_count += 1

        # Case 2: CFGGrammar
        elif isinstance(self.grammar, CFGGrammar):
            # Split prefix text into tokens/words
            words = prefix_text.strip().split()
            is_accepted = self.grammar.is_accepted(words) if words else False

            if is_accepted and self.eos_token_id is not None:
                mask[self.eos_token_id] = 0.0
                valid_count += 1

            for token_id, token_str in enumerate(self.token_strings):
                if not token_str or token_id == self.eos_token_id:
                    continue
                cand_text = prefix_text + token_str
                cand_words = cand_text.strip().split()
                if not cand_words or self.grammar.is_valid_prefix(cand_words):
                    mask[token_id] = 0.0
                    valid_count += 1

        # Case 3: IncrementalJSONStateMachine
        elif isinstance(self.grammar, IncrementalJSONStateMachine):
            if self.grammar.is_complete() and self.eos_token_id is not None:
                mask[self.eos_token_id] = 0.0
                valid_count += 1
            for token_id, token_str in enumerate(self.token_strings):
                if not token_str or token_id == self.eos_token_id:
                    continue
                candidate = prefix_text + token_str
                if self.grammar.is_valid_prefix(candidate):
                    mask[token_id] = 0.0
                    valid_count += 1

        # Fallback safeguard: if no candidate token is deemed valid, allow EOS or unmask
        if valid_count == 0:
            if self.eos_token_id is not None:
                mask[self.eos_token_id] = 0.0
            else:
                return single_scores

        return masked_scores + mask
