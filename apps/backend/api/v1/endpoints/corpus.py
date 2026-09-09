"""
Libra Backend API - Instruction Corpus & SFT Endpoints (Phase 33)
Provides ChatML formatting, loss masking inspection, and sequence packing simulation.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from packages.training.chat_formatter import (
    IGNORE_INDEX,
    ChatMessage,
    ChatMLFormatter,
    tokenize_with_loss_masking,
)
from packages.training.domain_corpora import get_educational_instruction_corpus
from packages.training.sft_dataset import pack_sequences

router = APIRouter(prefix="/corpus", tags=["Instruction Corpus & Fine-Tuning"])


class MessageInput(BaseModel):
    role: str = Field(..., description="system, user, assistant")
    content: str = Field(...)


class FormatChatRequest(BaseModel):
    messages: list[MessageInput] = Field(
        default=[
            MessageInput(role="system", content="You are Libra, an educational AI assistant."),
            MessageInput(role="user", content="Explain gradient descent in one sentence."),
            MessageInput(
                role="assistant",
                content="Gradient descent iteratively updates parameters in the direction of steepest descent to minimize loss.",
            ),
        ]
    )
    max_length: int = Field(default=256)


class TokenInspectionItem(BaseModel):
    token_idx: int
    token_id: int
    char_repr: str
    is_masked: bool
    label: int


class FormatChatResponse(BaseModel):
    raw_chatml: str
    total_tokens: int
    trainable_tokens: int
    masked_tokens: int
    trainable_fraction: float
    tokens: list[TokenInspectionItem]


class PackRequest(BaseModel):
    max_length: int = Field(default=256)


class PackResponse(BaseModel):
    original_dialogues: int
    packed_sequences: int
    tokens_unpacked_with_padding: int
    tokens_packed: int
    efficiency_gain_percent: float
    packed_batches: list[dict[str, Any]]


def simple_tokenizer(text: str) -> list[int]:
    return list(text.encode("utf-8"))


@router.post("/format_chat", response_model=FormatChatResponse)
async def format_chat_endpoint(req: FormatChatRequest) -> FormatChatResponse:
    """Formats messages into ChatML and inspects token loss masking."""
    chat_messages = [ChatMessage(role=m.role, content=m.content) for m in req.messages]
    formatter = ChatMLFormatter()
    raw_chatml = formatter.format_conversation(chat_messages)

    input_ids, label_ids = tokenize_with_loss_masking(
        messages=chat_messages,
        tokenizer=simple_tokenizer,
        max_length=req.max_length,
    )

    tokens = []
    trainable_count = 0
    for idx, (tok_id, lbl) in enumerate(zip(input_ids, label_ids, strict=False)):
        is_masked = lbl == IGNORE_INDEX
        if not is_masked:
            trainable_count += 1
        char_repr = chr(tok_id) if 32 <= tok_id <= 126 else f"\\x{tok_id:02x}"
        tokens.append(
            TokenInspectionItem(
                token_idx=idx,
                token_id=tok_id,
                char_repr=char_repr,
                is_masked=is_masked,
                label=lbl,
            )
        )

    total = len(input_ids)
    masked_count = total - trainable_count
    fraction = (trainable_count / total) if total > 0 else 0.0

    return FormatChatResponse(
        raw_chatml=raw_chatml,
        total_tokens=total,
        trainable_tokens=trainable_count,
        masked_tokens=masked_count,
        trainable_fraction=round(fraction, 3),
        tokens=tokens,
    )


@router.post("/pack", response_model=PackResponse)
async def pack_dialogues_endpoint(req: PackRequest) -> PackResponse:
    """Demonstrates multi-turn sequence packing eliminating padding waste."""
    corpus = get_educational_instruction_corpus()

    dialogue_tokens = []
    for dialogue in corpus:
        in_ids, lbl_ids = tokenize_with_loss_masking(
            dialogue, simple_tokenizer, max_length=req.max_length
        )
        dialogue_tokens.append((in_ids, lbl_ids))

    packed = pack_sequences(dialogue_tokens, max_length=req.max_length, eos_token_id=0)

    # In unpacked training, each sample would be padded to max_length
    padded_tokens_count = len(corpus) * req.max_length
    packed_tokens_count = sum(len(p["input_ids"]) for p in packed)
    efficiency = ((padded_tokens_count - packed_tokens_count) / padded_tokens_count) * 100.0

    return PackResponse(
        original_dialogues=len(corpus),
        packed_sequences=len(packed),
        tokens_unpacked_with_padding=padded_tokens_count,
        tokens_packed=packed_tokens_count,
        efficiency_gain_percent=round(efficiency, 2),
        packed_batches=[
            {
                "batch_idx": idx,
                "length": len(p["input_ids"]),
                "active_labels": int((p["labels"] != IGNORE_INDEX).sum().item()),
            }
            for idx, p in enumerate(packed)
        ],
    )


@router.get("/sample_dataset")
async def get_sample_dataset() -> list[dict[str, Any]]:
    """Returns sample educational instruction records."""
    corpus = get_educational_instruction_corpus()
    formatter = ChatMLFormatter()
    return [
        {
            "turns": len(d),
            "chatml": formatter.format_conversation(d),
        }
        for d in corpus
    ]
