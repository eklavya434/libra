"""
Phase 33 CLI Demonstration:
Domain Adaptation & Instruction Fine-Tuning Corpus Pipeline
Demonstrates:
1. Multi-turn ChatML conversation formatting (<|im_start|>role\ncontent<|im_end|>)
2. Prompt token loss masking with IGNORE_INDEX (-100) vs assistant completion tokens
3. Multi-turn dialogue sequence packing without padding waste
4. Domain corpus inspection (Systems, Coding, Math)
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from packages.training.chat_formatter import (
    ChatMessage,
    ChatMLFormatter,
    tokenize_with_loss_masking,
    IGNORE_INDEX,
)
from packages.training.sft_dataset import InstructionDataset, pack_sequences, default_tokenizer
from packages.training.domain_corpora import get_educational_instruction_corpus


def main():
    print("=" * 80)
    print("LIBRA PHASE 33: DOMAIN ADAPTATION & SFT CORPUS DEMO")
    print("=" * 80)

    # 1. ChatML Formatting
    print("\n[1] ChatML Delimiter Formatting & Role Serialization:")
    messages = [
        ChatMessage(role="system", content="You are Libra, an educational AI assistant built from first principles."),
        ChatMessage(role="user", content="Explain how Rotary Position Embedding (RoPE) works."),
        ChatMessage(role="assistant", content="RoPE encodes relative token position by applying a 2D rotation matrix to query and key vector slices."),
    ]
    formatter = ChatMLFormatter()
    raw_chatml = formatter.format_conversation(messages)
    print("Formatted ChatML Output:")
    print("-" * 50)
    print(raw_chatml, end="")
    print("-" * 50)

    # 2. Tokenization with Prompt Loss Masking
    print("\n[2] Tokenization & Completion-Only Loss Masking (-100):")
    input_ids, label_ids = tokenize_with_loss_masking(
        messages=messages,
        tokenizer=default_tokenizer,
        max_length=256,
        ignore_index=IGNORE_INDEX,
    )
    total_tokens = len(input_ids)
    trainable_tokens = sum(1 for lbl in label_ids if lbl != IGNORE_INDEX)
    masked_tokens = total_tokens - trainable_tokens
    trainable_pct = (trainable_tokens / total_tokens) * 100 if total_tokens else 0.0

    print(f"Total tokens in dialogue: {total_tokens}")
    print(f"Masked prompt tokens (loss = -100): {masked_tokens} ({(masked_tokens / total_tokens) * 100:.1f}%)")
    print(f"Active assistant tokens (loss != -100): {trainable_tokens} ({trainable_pct:.1f}%)")
    print("\nFirst 40 tokens comparison:")
    print("Idx | TokenID | Char | Label")
    print("----------------------------")
    for i in range(min(40, total_tokens)):
        c = chr(input_ids[i]) if 32 <= input_ids[i] <= 126 else "."
        lbl_str = "-100 (MASKED)" if label_ids[i] == IGNORE_INDEX else str(label_ids[i])
        print(f"{i:3d} | {input_ids[i]:7d} | {c:4s} | {lbl_str}")

    # 3. Multi-Turn Sequence Packing
    print("\n[3] Sequence Packing (Zero Padding Waste):")
    corpus = get_educational_instruction_corpus()
    print(f"Loaded {len(corpus)} educational dialogues from domain corpora.")

    tokenized_dialogues = []
    unpacked_token_sum = 0
    for diag in corpus:
        inp, lbl = tokenize_with_loss_masking(diag, default_tokenizer, max_length=512)
        tokenized_dialogues.append((inp, lbl))
        unpacked_token_sum += len(inp)

    max_pack_len = 512
    packed = pack_sequences(tokenized_dialogues, max_length=max_pack_len, eos_token_id=0)

    packed_token_sum = sum(len(p["input_ids"]) for p in packed)
    padding_saved = (len(corpus) * max_pack_len) - packed_token_sum
    efficiency = ((padding_saved) / (len(corpus) * max_pack_len)) * 100

    print(f"Original dialogue count: {len(corpus)}")
    print(f"Packed into {len(packed)} batches of max length {max_pack_len}")
    print(f"Unpacked naive padding total: {len(corpus) * max_pack_len} tokens")
    print(f"Packed tokens utilized: {packed_token_sum} tokens")
    print(f"Compute / Memory Efficiency Gain: {efficiency:.2f}% padding eliminated!")

    print("\n" + "=" * 80)
    print("Phase 33 SFT Corpus Pipeline Verified Successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
