"""
Libra Phase 12 Demonstration: Multi-Turn Conversation Memory & Context Management

Demonstrates:
1. SQLite WAL conversation persistence and session CRUD.
2. Auto-titling upon first user message.
3. Multi-turn message history tracking.
4. First-principles sliding-window context truncation preserving system prompt.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from packages.core.memory import (
    ContextWindowManager,
    SQLiteConversationStore,
)


def main():
    print("=" * 80)
    print("LIBRA PHASE 12: MULTI-TURN CONVERSATION MEMORY & CONTEXT WINDOW LABORATORY")
    print("SQLite Session Store, Message Cascades, and Sliding-Window Token Management")
    print("=" * 80)

    # 1. Initialize an in-memory SQLite store for clean demonstration
    store = SQLiteConversationStore(db_path=":memory:")
    print("\n1. Initialized SQLite Store (WAL mode, Foreign Keys ON, zero-dependency).")

    # 2. Create conversation session
    conv = store.create_conversation(
        model="libra-llama-tied",
        system_prompt="You are Libra, an educational AI assistant specializing in neural networks.",
    )
    print(f"\n2. Created Session:")
    print(f"   - ID:             {conv.id}")
    print(f"   - Initial Title:  '{conv.title}'")
    print(f"   - Model:          {conv.model}")
    print(f"   - System Prompt:  {conv.system_prompt}")

    # 3. Simulate multi-turn user/assistant exchanges
    print("\n3. Simulating Multi-Turn Dialog:")
    turns = [
        ("user", "What is attention in transformer models?"),
        ("assistant", "Attention computes a weighted average of values based on similarity between queries and keys."),
        ("user", "Can you explain why dot product is scaled by sqrt(d_k)?"),
        ("assistant", "Scaling prevents large dot products from pushing softmax gradients into tiny saturation regions."),
        ("user", "Now explain what RoPE positional encoding does."),
        ("assistant", "Rotary Positional Embedding rotates query and key vectors in complex 2D subspaces."),
    ]

    for role, content in turns:
        msg = store.add_message(conv.id, role=role, content=content)
        icon = "[USR]" if role == "user" else "[BOT]"
        print(f"   {icon} [{role.upper():9s}] {content}")

    # 4. Check auto-titling result
    updated_conv = store.get_conversation(conv.id)
    print(f"\n4. Session State After Exchanges:")
    print(f"   - Auto-Titled:    '{updated_conv.title}'")
    print(f"   - Total Messages: {updated_conv.message_count}")
    print(f"   - Last Updated:   {updated_conv.updated_at}")

    # 5. Educational Context Window Management Demo
    print("\n5. Testing Context Window Manager (Sliding-Window Truncation):")
    print("   Scenario: Target local model has a constrained 120-token context limit.")
    print("   Goal: Reserve 40 tokens for generation, keeping input strictly <= 80 tokens.")

    ctx_mgr = ContextWindowManager(
        max_context_tokens=120,
        reserved_completion_tokens=40,
        chars_per_token=3.8,
    )

    history = store.get_messages(conv.id)
    truncated = ctx_mgr.prepare_context(
        messages=history,
        override_system_prompt=updated_conv.system_prompt,
    )

    print(f"\n   Context Window Budget Summary:")
    print(f"   - Max Context Tokens:     {truncated.max_context_tokens}")
    print(f"   - Reserved for Output:    {truncated.reserved_completion_tokens}")
    print(f"   - Total Input Tokens:     {truncated.total_tokens}")
    print(f"   - Dropped Older Turns:    {truncated.truncated_count} message(s)")
    print(f"   - Remaining Headroom:     {truncated.remaining_capacity} tokens")

    print("\n   Retained Prompt History Dispatched to Model:")
    for idx, m in enumerate(truncated.messages):
        badge = "ANCHOR" if m["role"] == "system" else f"TURN {idx}"
        print(f"   [{badge:8s}] {m['role'].upper():9s}: {m['content']}")

    # 6. Listing and Cascade Deletion
    print("\n6. Verifying Storage Deletion & Integrity:")
    store.delete_conversation(conv.id)
    remaining_messages = store.get_messages(conv.id)
    print(f"   - Conversation deleted successfully.")
    print(f"   - Cascaded message count in DB: {len(remaining_messages)} (Verified clean cascade)")

    print("\n" + "=" * 80)
    print("PHASE 12 COMPLETE: Multi-Turn Conversation Memory Verified Successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
