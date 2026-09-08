"""
Libra Core Grammar Package - Context-Free Grammar (CFG) Earley Parser
First-principles implementation of Jay Earley's (1970) chart parsing algorithm
for incremental prefix validation and next-token constraint masking.
"""

from __future__ import annotations

import re


class Production:
    """A context-free grammar production rule: LHS -> RHS[0] RHS[1] ..."""

    def __init__(self, lhs: str, rhs: tuple[str, ...]) -> None:
        self.lhs: str = lhs
        self.rhs: tuple[str, ...] = rhs

    def __repr__(self) -> str:
        rhs_str = " ".join(self.rhs) if self.rhs else "ε"
        return f"{self.lhs} -> {rhs_str}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Production):
            return False
        return self.lhs == other.lhs and self.rhs == other.rhs

    def __hash__(self) -> int:
        return hash((self.lhs, self.rhs))


class EarleyItem:
    """An item in an Earley chart: [A -> alpha . beta, origin]."""

    def __init__(self, production: Production, dot: int, origin: int) -> None:
        self.production: Production = production
        self.dot: int = dot
        self.origin: int = origin

    @property
    def next_symbol(self) -> str | None:
        if self.dot < len(self.production.rhs):
            return self.production.rhs[self.dot]
        return None

    @property
    def is_complete(self) -> bool:
        return self.dot >= len(self.production.rhs)

    def advance(self) -> EarleyItem:
        """Returns a new item with the dot shifted one position to the right."""
        return EarleyItem(self.production, self.dot + 1, self.origin)

    def __repr__(self) -> str:
        rhs = list(self.production.rhs)
        rhs.insert(self.dot, "•")
        return f"[{self.production.lhs} -> {' '.join(rhs)}, {self.origin}]"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EarleyItem):
            return False
        return (
            self.production == other.production
            and self.dot == other.dot
            and self.origin == other.origin
        )

    def __hash__(self) -> int:
        return hash((self.production, self.dot, self.origin))


class CFGGrammar:
    """Context-Free Grammar representation and Earley prefix parser.

    Can be specified via simple text rule definitions:
        root -> expr
        expr -> expr "+" term | expr "-" term | term
        term -> term "*" factor | term "/" factor | factor
        factor -> "(" expr ")" | "[0-9]+"
    """

    def __init__(self, rules_text: str | None = None, start_symbol: str = "root") -> None:
        self.productions: list[Production] = []
        self.non_terminals: set[str] = set()
        self.terminals: set[str] = set()
        self.start_symbol: str = start_symbol

        # Augmented start rule: __START__ -> start_symbol
        self.augmented_start = "__START__"
        self.augmented_production = Production(self.augmented_start, (self.start_symbol,))

        if rules_text:
            self.parse_rules(rules_text)

    def add_production(self, lhs: str, rhs: list[str] | tuple[str, ...]) -> None:
        prod = Production(lhs, tuple(rhs))
        self.productions.append(prod)
        self.non_terminals.add(lhs)
        for sym in rhs:
            if not sym.isidentifier() or sym.isupper() or sym.startswith(('"', "'")):
                self.terminals.add(sym)

    def parse_rules(self, rules_text: str) -> None:
        """Parses standard BNF / EBNF multi-line grammar declarations."""
        lines = rules_text.strip().splitlines()
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Supports 'lhs -> rhs' or 'lhs = rhs' or 'lhs ::= rhs'
            sep = None
            for candidate_sep in ["->", "::=", "="]:
                if candidate_sep in line:
                    sep = candidate_sep
                    break

            if not sep:
                continue

            lhs, rhs_all = line.split(sep, 1)
            lhs = lhs.strip()

            # Split alternations '|'
            alternatives = rhs_all.split("|")
            for alt in alternatives:
                symbols = self._tokenize_rhs(alt.strip())
                self.add_production(lhs, symbols)

        # Infer non-terminals vs terminals
        for prod in self.productions:
            for sym in prod.rhs:
                if sym not in self.non_terminals:
                    self.terminals.add(sym)

    def _tokenize_rhs(self, rhs_str: str) -> list[str]:
        """Tokenizes RHS string respecting quoted literals and symbols."""
        tokens: list[str] = []
        # Match quoted literals ("foo" or 'foo') or whitespace-separated tokens
        pattern = r'"([^"]*)"|\'([^\']*)\'|(\S+)'
        for match in re.finditer(pattern, rhs_str):
            double_q, single_q, bare = match.groups()
            if double_q is not None:
                tokens.append(double_q)
            elif single_q is not None:
                tokens.append(single_q)
            elif bare is not None:
                tokens.append(bare)
        return tokens

    def _get_productions_for(self, non_terminal: str) -> list[Production]:
        if non_terminal == self.augmented_start:
            return [self.augmented_production]
        return [p for p in self.productions if p.lhs == non_terminal]

    def _is_terminal(self, symbol: str) -> bool:
        return symbol not in self.non_terminals and symbol != self.augmented_start

    def _matches_terminal(self, terminal: str, token: str) -> bool:
        """Checks if a literal string or terminal symbol matches the token."""
        # Exact match
        if terminal == token:
            return True
        # Quoted literal match
        if (terminal.startswith('"') and terminal.endswith('"')) or (
            terminal.startswith("'") and terminal.endswith("'")
        ):
            return terminal[1:-1] == token
        # Common pre-defined terminal tokens
        if terminal == "NUMBER":
            return token.isdigit()
        if terminal == "IDENTIFIER":
            return token.isidentifier()
        if terminal == "STRING":
            return (token.startswith('"') and token.endswith('"')) or (
                token.startswith("'") and token.endswith("'")
            )
        # Character class regex like [0-9]+
        if terminal.startswith("[") and terminal.endswith("]"):
            return bool(re.fullmatch(terminal, token))
        return False

    def build_chart(self, tokens: list[str]) -> list[list[EarleyItem]]:
        """Executes the Earley parsing algorithm across the token sequence.

        Returns the full chart containing item sets at each token boundary.
        """
        n = len(tokens)
        chart: list[list[EarleyItem]] = [[] for _ in range(n + 1)]
        seen: list[set[EarleyItem]] = [set() for _ in range(n + 1)]

        def add_item(item: EarleyItem, index: int) -> bool:
            if item not in seen[index]:
                seen[index].add(item)
                chart[index].append(item)
                return True
            return False

        # Initialize chart[0] with augmented start production
        add_item(EarleyItem(self.augmented_production, 0, 0), 0)

        # Process each chart position 0 ... n
        for i in range(n + 1):
            queue_idx = 0
            while queue_idx < len(chart[i]):
                item = chart[i][queue_idx]
                queue_idx += 1

                if not item.is_complete:
                    sym = item.next_symbol
                    if sym in self.non_terminals or sym == self.start_symbol:
                        # 1. PREDICTOR: expand non-terminal productions
                        for prod in self._get_productions_for(sym):
                            add_item(EarleyItem(prod, 0, i), i)
                    elif (
                        self._is_terminal(sym) and i < n and self._matches_terminal(sym, tokens[i])
                    ):
                        # 2. SCANNER: match terminal against current input token
                        add_item(item.advance(), i + 1)
                else:
                    # 3. COMPLETER: advance waiting items from origin
                    origin = item.origin
                    for prev_item in chart[origin]:
                        if (
                            not prev_item.is_complete
                            and prev_item.next_symbol == item.production.lhs
                        ):
                            add_item(prev_item.advance(), i)

        return chart

    def is_valid_prefix(self, tokens: list[str]) -> bool:
        """Determines if the token sequence is a mathematically valid prefix in the grammar."""
        chart = self.build_chart(tokens)
        last_index = len(tokens)
        return len(chart[last_index]) > 0

    def is_accepted(self, tokens: list[str]) -> bool:
        """Determines if the token sequence is a complete, accepted sentence in the grammar."""
        chart = self.build_chart(tokens)
        last_index = len(tokens)
        # Search for [__START__ -> start_symbol •, 0]
        for item in chart[last_index]:
            if (
                item.production.lhs == self.augmented_start
                and item.is_complete
                and item.origin == 0
            ):
                return True
        return False

    def get_valid_next_terminals(self, tokens: list[str]) -> set[str]:
        """Computes the exact set of valid next terminals at the current prefix."""
        chart = self.build_chart(tokens)
        last_index = len(tokens)
        allowed: set[str] = set()

        for item in chart[last_index]:
            if not item.is_complete:
                sym = item.next_symbol
                if sym and self._is_terminal(sym):
                    # Strip quotes if literal
                    clean_sym = (
                        sym[1:-1]
                        if (sym.startswith('"') and sym.endswith('"'))
                        or (sym.startswith("'") and sym.endswith("'"))
                        else sym
                    )
                    allowed.add(clean_sym)

        return allowed
