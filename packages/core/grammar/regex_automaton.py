"""
Libra Core Grammar Package - First-Principles Thompson NFA Regex Automaton
Enables incremental prefix validation and state-based constrained decoding for regular expressions.
"""

from __future__ import annotations


class NFAState:
    """A single state in Thompson's Non-Deterministic Finite Automaton (NFA)."""

    _id_counter: int = 0

    def __init__(self, is_accept: bool = False) -> None:
        self.id: int = NFAState._id_counter
        NFAState._id_counter += 1
        self.is_accept: bool = is_accept
        # Transitions on specific characters: char -> set of target states
        self.transitions: dict[str, set[NFAState]] = {}
        # Transitions on character predicate functions: (func, repr_name) -> set of target states
        self.predicate_transitions: list[tuple[callable, set[NFAState]]] = []
        # Epsilon transitions (no character consumed): set of target states
        self.epsilon_transitions: set[NFAState] = set()

    def add_char_transition(self, char: str, target: NFAState) -> None:
        if char not in self.transitions:
            self.transitions[char] = set()
        self.transitions[char].add(target)

    def add_predicate_transition(self, pred: callable, target: NFAState) -> None:
        self.predicate_transitions.append((pred, {target}))

    def add_epsilon_transition(self, target: NFAState) -> None:
        self.epsilon_transitions.add(target)

    def __repr__(self) -> str:
        return f"NFAState({self.id}, accept={self.is_accept})"


class NFAFragment:
    """An NFA sub-graph with a defined entry state and exit state."""

    def __init__(self, start: NFAState, end: NFAState) -> None:
        self.start: NFAState = start
        self.end: NFAState = end


def epsilon_closure(states: set[NFAState]) -> set[NFAState]:
    """Computes the set of states reachable via zero or more epsilon transitions."""
    closure = set(states)
    stack = list(states)

    while stack:
        current = stack.pop()
        for next_state in current.epsilon_transitions:
            if next_state not in closure:
                closure.add(next_state)
                stack.append(next_state)

    return closure


class RegexParser:
    """First-principles parser and Thompson NFA compiler for regular expressions.

    Supports:
    - Literal characters and escape codes (\\d, \\w, \\s, \\., \\-, etc.)
    - Wildcard dot (.)
    - Character classes ([a-z], [0-9], [abc], [^0-9])
    - Concatenation (ab)
    - Alternation (a|b)
    - Kleene Star (*), Plus (+), Optional (?)
    - Repetition ranges ({n,m}, {n})
    - Grouping with parentheses ((...))
    """

    def __init__(self, pattern: str) -> None:
        # Strip anchor characters if present (the automaton matches the full pattern by default)
        clean_pat = pattern
        clean_pat = clean_pat.removeprefix("^")
        clean_pat = clean_pat.removesuffix("$")

        self.pattern: str = clean_pat
        self.pos: int = 0
        self.length: int = len(self.pattern)

    def compile(self) -> NFAFragment:
        """Parses the regex string and compiles it into an NFA fragment."""
        if not self.pattern:
            # Empty regex accepts empty string
            s = NFAState()
            e = NFAState(is_accept=True)
            s.add_epsilon_transition(e)
            return NFAFragment(s, e)

        frag = self._parse_alternation()
        frag.end.is_accept = True
        return frag

    def _peek(self) -> str | None:
        return self.pattern[self.pos] if self.pos < self.length else None

    def _get(self) -> str:
        ch = self.pattern[self.pos]
        self.pos += 1
        return ch

    def _parse_alternation(self) -> NFAFragment:
        left = self._parse_concatenation()

        while self._peek() == "|":
            self._get()  # Consume '|'
            right = self._parse_concatenation()

            start = NFAState()
            end = NFAState()

            start.add_epsilon_transition(left.start)
            start.add_epsilon_transition(right.start)

            left.end.add_epsilon_transition(end)
            right.end.add_epsilon_transition(end)

            left = NFAFragment(start, end)

        return left

    def _parse_concatenation(self) -> NFAFragment:
        fragments: list[NFAFragment] = []

        while self.pos < self.length and self._peek() not in ")|":
            frag = self._parse_quantifier()
            fragments.append(frag)

        if not fragments:
            s = NFAState()
            e = NFAState()
            s.add_epsilon_transition(e)
            return NFAFragment(s, e)

        # Chain all fragments in series
        for i in range(len(fragments) - 1):
            fragments[i].end.add_epsilon_transition(fragments[i + 1].start)

        return NFAFragment(fragments[0].start, fragments[-1].end)

    def _parse_quantifier(self) -> NFAFragment:
        atom = self._parse_atom()
        peek = self._peek()

        if peek == "*":
            self._get()
            start = NFAState()
            end = NFAState()

            start.add_epsilon_transition(atom.start)
            start.add_epsilon_transition(end)
            atom.end.add_epsilon_transition(atom.start)
            atom.end.add_epsilon_transition(end)

            return NFAFragment(start, end)

        elif peek == "+":
            self._get()
            start = NFAState()
            end = NFAState()

            start.add_epsilon_transition(atom.start)
            atom.end.add_epsilon_transition(atom.start)
            atom.end.add_epsilon_transition(end)

            return NFAFragment(start, end)

        elif peek == "?":
            self._get()
            start = NFAState()
            end = NFAState()

            start.add_epsilon_transition(atom.start)
            start.add_epsilon_transition(end)
            atom.end.add_epsilon_transition(end)

            return NFAFragment(start, end)

        elif peek == "{":
            # Range quantifier: {n}, {n,}, {n,m}
            self._get()  # Consume '{'
            range_str = ""
            while self.pos < self.length and self._peek() != "}":
                range_str += self._get()
            if self._peek() == "}":
                self._get()  # Consume '}'

            return self._expand_range(atom, range_str)

        return atom

    def _expand_range(self, atom: NFAFragment, range_str: str) -> NFAFragment:
        """Expands repetition range like {2,4} by cloning or chaining fragments."""
        parts = [p.strip() for p in range_str.split(",")]
        min_count = int(parts[0]) if parts[0] else 0
        max_count = None
        if len(parts) > 1:
            max_count = int(parts[1]) if parts[1] else None
        else:
            max_count = min_count

        # For simple bounded ranges up to 16, expand sequentially
        min_count = min(min_count, 16)
        if max_count is not None:
            max_count = min(max_count, 16)

        # Helper to deep copy single transitions for atomic repeat
        start = NFAState()
        curr = start

        # Exact repeat for min_count
        for _ in range(min_count):
            sub_start = NFAState()
            sub_end = NFAState()
            self._copy_fragment_edges(atom, sub_start, sub_end)
            curr.add_epsilon_transition(sub_start)
            curr = sub_end

        if max_count is None:
            # Unbounded: min_count followed by *
            star_start = NFAState()
            star_end = NFAState()
            sub_start = NFAState()
            sub_end = NFAState()
            self._copy_fragment_edges(atom, sub_start, sub_end)

            star_start.add_epsilon_transition(sub_start)
            star_start.add_epsilon_transition(star_end)
            sub_end.add_epsilon_transition(sub_start)
            sub_end.add_epsilon_transition(star_end)

            curr.add_epsilon_transition(star_start)
            curr = star_end
        else:
            # Optional repetitions up to max_count
            end_sink = NFAState()
            curr.add_epsilon_transition(end_sink)

            for _ in range(max_count - min_count):
                opt_start = NFAState()
                opt_end = NFAState()
                self._copy_fragment_edges(atom, opt_start, opt_end)
                curr.add_epsilon_transition(opt_start)
                opt_end.add_epsilon_transition(end_sink)
                curr = opt_end

            curr = end_sink

        return NFAFragment(start, curr)

    def _copy_fragment_edges(
        self, src: NFAFragment, new_start: NFAState, new_end: NFAState
    ) -> None:
        """Copies edges from src into new_start and new_end for character atoms."""
        # For atomic single-char fragments
        for ch, targets in src.start.transitions.items():
            for _ in targets:
                new_start.add_char_transition(ch, new_end)
        for pred, _ in src.start.predicate_transitions:
            new_start.add_predicate_transition(pred, new_end)
        for _ in src.start.epsilon_transitions:
            new_start.add_epsilon_transition(new_end)

    def _parse_atom(self) -> NFAFragment:
        ch = self._get()

        if ch == "(":
            frag = self._parse_alternation()
            if self._peek() == ")":
                self._get()  # Consume ')'
            return frag

        elif ch == "[":
            return self._parse_character_class()

        elif ch == "\\":
            return self._parse_escape()

        elif ch == ".":
            # Wildcard: matches any character except newline
            start = NFAState()
            end = NFAState()
            start.add_predicate_transition(lambda c: c != "\n", end)
            return NFAFragment(start, end)

        else:
            # Literal character
            start = NFAState()
            end = NFAState()
            start.add_char_transition(ch, end)
            return NFAFragment(start, end)

    def _parse_character_class(self) -> NFAFragment:
        negate = False
        if self._peek() == "^":
            negate = True
            self._get()

        allowed_chars: set[str] = set()
        class_chars = ""

        while self.pos < self.length and self._peek() != "]":
            c = self._get()
            if c == "\\" and self.pos < self.length:
                esc = self._get()
                if esc == "d":
                    class_chars += "0123456789"
                elif esc == "w":
                    class_chars += "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
                elif esc == "s":
                    class_chars += " \t\n\r"
                else:
                    class_chars += esc
            elif (
                self._peek() == "-"
                and self.pos + 1 < self.length
                and self.pattern[self.pos + 1] != "]"
            ):
                self._get()  # Consume '-'
                end_range = self._get()
                # Expand ASCII range
                for code in range(ord(c), ord(end_range) + 1):
                    class_chars += chr(code)
            else:
                class_chars += c

        if self._peek() == "]":
            self._get()  # Consume ']'

        allowed_chars.update(class_chars)

        start = NFAState()
        end = NFAState()

        if negate:
            start.add_predicate_transition(
                lambda c, ac=allowed_chars: c not in ac and c != "\n", end
            )
        else:
            # Direct char lookup or set lookup
            start.add_predicate_transition(lambda c, ac=allowed_chars: c in ac, end)

        return NFAFragment(start, end)

    def _parse_escape(self) -> NFAFragment:
        if self.pos >= self.length:
            start = NFAState()
            end = NFAState()
            start.add_char_transition("\\", end)
            return NFAFragment(start, end)

        esc = self._get()
        start = NFAState()
        end = NFAState()

        if esc == "d":
            # Digit
            start.add_predicate_transition(lambda c: c.isdigit(), end)
        elif esc == "w":
            # Word character
            start.add_predicate_transition(lambda c: c.isalnum() or c == "_", end)
        elif esc == "s":
            # Whitespace
            start.add_predicate_transition(lambda c: c.isspace(), end)
        elif esc == "D":
            start.add_predicate_transition(lambda c: not c.isdigit(), end)
        elif esc == "W":
            start.add_predicate_transition(lambda c: not (c.isalnum() or c == "_"), end)
        elif esc == "S":
            start.add_predicate_transition(lambda c: not c.isspace(), end)
        else:
            # Literal escaped character (e.g. \., \-, \/)
            start.add_char_transition(esc, end)

        return NFAFragment(start, end)


class RegexAutomaton:
    """Maintains active state sets for incremental prefix testing and validation against a regex."""

    def __init__(self, pattern: str) -> None:
        self.pattern: str = pattern
        parser = RegexParser(pattern)
        self.fragment: NFAFragment = parser.compile()
        self.initial_states: set[NFAState] = epsilon_closure({self.fragment.start})

    def step(self, current_states: set[NFAState], char: str) -> set[NFAState]:
        """Steps from a set of states consuming character char."""
        next_states: set[NFAState] = set()

        for state in current_states:
            # Direct character match
            if char in state.transitions:
                next_states.update(state.transitions[char])

            # Predicate matches
            for pred, targets in state.predicate_transitions:
                if pred(char):
                    next_states.update(targets)

        return epsilon_closure(next_states)

    def is_valid_prefix(self, prefix: str) -> bool:
        """Determines if the string prefix can be extended into a valid matching string."""
        states = self.initial_states

        for ch in prefix:
            states = self.step(states, ch)
            if not states:
                return False

        return len(states) > 0

    def is_accepted(self, text: str) -> bool:
        """Determines if the string text is fully accepted by the regex."""
        states = self.initial_states

        for ch in text:
            states = self.step(states, ch)
            if not states:
                return False

        return any(s.is_accept for s in states)

    def can_accept_more(self, text: str) -> bool:
        """Checks if the automaton can accept additional characters beyond text."""
        states = self.initial_states
        for ch in text:
            states = self.step(states, ch)
            if not states:
                return False

        # Check if there are non-accepting active states or outgoing transitions
        return any(bool(s.transitions or s.predicate_transitions) for s in states)
