"""
Libra Core Grammar Package - Constrained Decoding & Schema Validation

Exports:
- IncrementalJSONStateMachine: Pushdown automaton tracking partial JSON
- SchemaCompiler, SchemaConstraint: JSON schema & Pydantic rule compiler
- ConstrainedLogitsProcessor: PyTorch autoregressive logits mask processor
"""

from packages.core.grammar.cfg_parser import CFGGrammar, EarleyItem, Production
from packages.core.grammar.grammar_processor import GrammarLogitsProcessor
from packages.core.grammar.json_state_machine import (
    IncrementalJSONStateMachine,
    JSONState,
)
from packages.core.grammar.logits_processor import ConstrainedLogitsProcessor
from packages.core.grammar.regex_automaton import NFAFragment, NFAState, RegexAutomaton
from packages.core.grammar.schema_compiler import SchemaCompiler, SchemaConstraint

__all__ = [
    "CFGGrammar",
    "ConstrainedLogitsProcessor",
    "EarleyItem",
    "GrammarLogitsProcessor",
    "IncrementalJSONStateMachine",
    "JSONState",
    "NFAFragment",
    "NFAState",
    "Production",
    "RegexAutomaton",
    "SchemaCompiler",
    "SchemaConstraint",
]
