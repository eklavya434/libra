"""
Libra Core Grammar Package - Constrained Decoding & Schema Validation

Exports:
- IncrementalJSONStateMachine: Pushdown automaton tracking partial JSON
- SchemaCompiler, SchemaConstraint: JSON schema & Pydantic rule compiler
- ConstrainedLogitsProcessor: PyTorch autoregressive logits mask processor
"""

from packages.core.grammar.json_state_machine import (
    IncrementalJSONStateMachine,
    JSONState,
)
from packages.core.grammar.logits_processor import ConstrainedLogitsProcessor
from packages.core.grammar.schema_compiler import SchemaCompiler, SchemaConstraint

__all__ = [
    "IncrementalJSONStateMachine",
    "JSONState",
    "SchemaCompiler",
    "SchemaConstraint",
    "ConstrainedLogitsProcessor",
]
