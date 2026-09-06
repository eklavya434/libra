"""
Libra Core Grammar Package - Constrained Logits Processor

Provides PyTorch-compatible logits processing that masks invalid candidate tokens
at each step of autoregressive generation using the IncrementalJSONStateMachine.
"""

from __future__ import annotations

from typing import Any, Callable, Optional
import torch

from packages.core.grammar.json_state_machine import IncrementalJSONStateMachine


class ConstrainedLogitsProcessor:
    """
    Autoregressive logits processor for grammar-constrained generation.
    Evaluates vocabulary tokens against the IncrementalJSONStateMachine prefix rules
    and sets logits of invalid next tokens to -inf.
    """

    def __init__(
        self,
        decode_fn: Callable[[list[int]], str],
        vocab_size: int,
        eos_token_id: Optional[int] = None,
        custom_state_machine: Optional[IncrementalJSONStateMachine] = None,
    ) -> None:
        self.decode_fn = decode_fn
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.state_machine = custom_state_machine or IncrementalJSONStateMachine()

        # Pre-decode single token strings for fast candidate validation
        self.token_strings: list[str] = []
        for token_id in range(vocab_size):
            try:
                decoded = self.decode_fn([token_id])
                self.token_strings.append(decoded)
            except Exception:
                self.token_strings.append("")

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        prefix_offset: int = 0,
    ) -> torch.FloatTensor:
        """
        Applies logit mask to scores based on the generated text prefix.
        Args:
            input_ids: 1D or 2D tensor of token IDs generated so far.
            scores: (batch_size, vocab_size) or (vocab_size,) logits tensor.
            prefix_offset: Index in input_ids where generation started (ignoring prompt).
        Returns:
            Masked scores tensor with invalid tokens set to -inf.
        """
        # Handle 1D or 2D tensor
        if scores.dim() == 1:
            return self._process_single(input_ids[prefix_offset:].tolist(), scores)

        # Batch processing (typically batch_size=1 on CPU)
        batch_size = scores.size(0)
        output_scores = scores.clone()

        for b in range(batch_size):
            tokens = input_ids[b, prefix_offset:].tolist() if input_ids.dim() == 2 else input_ids[prefix_offset:].tolist()
            output_scores[b] = self._process_single(tokens, scores[b])

        return output_scores

    def _process_single(self, generated_tokens: list[int], single_scores: torch.FloatTensor) -> torch.FloatTensor:
        """Processes logits for a single sequence."""
        # Decode current generated string
        prefix_text = self.decode_fn(generated_tokens) if generated_tokens else ""
        masked_scores = single_scores.clone()

        # Check if current prefix is already complete JSON
        temp_pda = IncrementalJSONStateMachine()
        is_valid = True
        for ch in prefix_text:
            if not temp_pda.feed_char(ch):
                is_valid = False
                break

        if is_valid and temp_pda.is_complete():
            # If JSON is complete, allow EOS token and whitespace
            if self.eos_token_id is not None:
                mask = torch.full_like(masked_scores, -float("inf"))
                mask[self.eos_token_id] = 0.0
                # Also allow whitespace tokens
                for tid, tstr in enumerate(self.token_strings):
                    if tstr and all(c in " \t\n\r" for c in tstr):
                        mask[tid] = 0.0
                return masked_scores + mask

        # Otherwise, evaluate which tokens can extend prefix_text legally
        valid_count = 0
        mask = torch.full_like(masked_scores, -float("inf"))

        for token_id, token_str in enumerate(self.token_strings):
            if not token_str:
                continue

            candidate = prefix_text + token_str
            if self.state_machine.is_valid_prefix(candidate):
                mask[token_id] = 0.0
                valid_count += 1

        # Fallback safeguard: if no tokens were deemed valid, don't zero out entire distribution
        if valid_count == 0:
            if self.eos_token_id is not None:
                mask[self.eos_token_id] = 0.0
            else:
                return single_scores

        return masked_scores + mask
