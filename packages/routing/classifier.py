"""
Libra Routing - Query Complexity & Intent Classifier

Evaluates prompt semantics, syntax, structure, and constraints to classify
user queries into intent categories and calculate multi-factor complexity scores,
recommending optimal execution tiers (Fast/Local, Balanced, Frontier, Multi-Agent).
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, ClassVar

from pydantic import BaseModel, Field


class ExecutionTier(str, Enum):
    """Execution tiers representing latency, cost, and cognitive capability levels."""

    FAST_LOCAL = "fast_local"  # Tiny local models / mocks (sub-50ms, zero-cost)
    BALANCED = "balanced"  # Mid-sized local 3B-7B or fast cloud (standard tasks)
    FRONTIER_REASONING = (
        "frontier_reasoning"  # High-reasoning models (code synthesis, complex math)
    )
    MULTI_AGENT = "multi_agent"  # Collaborative multi-agent teams (multi-step architecture)


class IntentType(str, Enum):
    """Categorization of user intent."""

    GREETING = "greeting"
    FACTUAL_QA = "factual_qa"
    CODE_GENERATION = "code_generation"
    REASONING_MATH = "reasoning_math"
    CREATIVE_WRITING = "creative_writing"
    DATA_EXTRACTION = "data_extraction"
    MULTI_STEP_PLANNING = "multi_step_planning"
    GENERAL = "general"


class ComplexityBreakdown(BaseModel):
    """Detailed breakdown of multi-factor complexity scores (0.0 to 1.0)."""

    length_score: float = Field(..., ge=0.0, le=1.0, description="Normalized input length score")
    code_score: float = Field(
        ..., ge=0.0, le=1.0, description="Code syntax and programming density score"
    )
    reasoning_score: float = Field(
        ..., ge=0.0, le=1.0, description="Logical, mathematical, and reasoning density"
    )
    constraint_score: float = Field(
        ..., ge=0.0, le=1.0, description="Formatting, schema, and operational constraints"
    )
    overall_score: float = Field(
        ..., ge=0.0, le=1.0, description="Composite weighted complexity score"
    )


class QueryClassification(BaseModel):
    """Result of prompt classification and tier recommendation."""

    prompt_snippet: str
    intent: IntentType
    domain: str
    complexity: ComplexityBreakdown
    recommended_tier: ExecutionTier
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str


class QueryClassifier:
    """Classifies user queries by intent and calculates multi-factor complexity."""

    # Intent detection patterns
    GREETING_PATTERNS: ClassVar[list[str]] = [
        r"\b(hi|hello|hey|good\s+(morning|afternoon|evening)|howdy|greetings)\b",
        r"\b(how\s+are\s+you|who\s+are\s+you|what\s+is\s+your\s+name|what\s+can\s+you\s+do)\b",
        r"\b(thanks|thank\s+you|bye|goodbye)\b",
    ]

    CODE_PATTERNS: ClassVar[list[str]] = [
        r"\b(def|class|import|return|function|const|let|var|async|await|public|private)\b",
        r"\b(python|javascript|typescript|c\+\+|rust|java|go|html|css|sql|json|yaml)\b",
        r"\b(debug|refactor|compile|syntax|exception|stack\s*trace|unit\s*test|pytest)\b",
        r"```[\w]*\n[\s\S]*?\n```",
        r"[\{\}\[\]\(\);=>]{3,}",
    ]

    REASONING_MATH_PATTERNS: ClassVar[list[str]] = [
        r"\b(prove|proof|derive|derivation|calculate|integral|derivative|theorem|lemma)\b",
        r"\b(step-by-step|chain\s+of\s+thought|deduce|logic|induction|combinatorics|probability)\b",
        r"\b(why\s+does|explain\s+the\s+mechanism|root\s+cause|compare\s+and\s+contrast|trade-?offs)\b",
        r"[\+\-\*\/\^=<>]{2,}",
    ]

    MULTI_STEP_PLANNING_PATTERNS: ClassVar[list[str]] = [
        r"\b(architect|architecture|design\s+system|multi-?agent|collaboration|end-to-end)\b",
        r"\b(full-?stack|comprehensive\s+plan|workflow|pipeline|orchestrate|milestones)\b",
        r"\b(step\s+1|phase\s+1|firstly|sub-?tasks|decomposition)\b",
    ]

    CREATIVE_PATTERNS: ClassVar[list[str]] = [
        r"\b(write\s+a\s+(story|poem|essay|script|dialogue|novel|haiku))\b",
        r"\b(metaphor|creative|fictional|character|plot|rhyme)\b",
    ]

    DATA_EXTRACTION_PATTERNS: ClassVar[list[str]] = [
        r"\b(extract|parse|convert\s+to\s+json|format\s+as\s+table|summarize\s+table|csv|regex)\b",
        r"\b(find\s+all\s+(emails|dates|numbers|names|keys))\b",
    ]

    CONSTRAINT_PATTERNS: ClassVar[list[str]] = [
        r"\b(must|never|only|strictly|exactly|json\s+format|schema|valid\s+json|in\s+\d+\s+words)\b",
        r"\b(without\s+using|do\s+not\s+include|enforce|constraint|requirement)\b",
    ]

    def __init__(
        self,
        length_weight: float = 0.15,
        code_weight: float = 0.35,
        reasoning_weight: float = 0.35,
        constraint_weight: float = 0.15,
    ) -> None:
        self.w_len = length_weight
        self.w_code = code_weight
        self.w_reason = reasoning_weight
        self.w_const = constraint_weight

    def _score_length(self, text: str) -> float:
        """Scores length logarithmically up to ~1000 characters."""
        length = len(text.strip())
        if length <= 20:
            return 0.05
        if length <= 100:
            return 0.15
        if length <= 300:
            return 0.40
        if length <= 800:
            return 0.70
        return min(1.0, 0.70 + (length - 800) / 2000.0)

    def _score_matches(self, text: str, patterns: list[str]) -> float:
        """Computes density of regex pattern matches."""
        text_lower = text.lower()
        matches = 0
        for pat in patterns:
            if re.search(pat, text_lower, re.IGNORECASE):
                matches += 1
        return min(1.0, matches / max(1, len(patterns) * 0.5))

    def compute_complexity(self, prompt: str) -> ComplexityBreakdown:
        """Calculates fine-grained multi-factor complexity scores."""
        len_score = self._score_length(prompt)
        code_score = self._score_matches(prompt, self.CODE_PATTERNS)
        reason_score = self._score_matches(prompt, self.REASONING_MATH_PATTERNS)
        const_score = self._score_matches(prompt, self.CONSTRAINT_PATTERNS)

        # Boost code score if explicit code blocks are present
        if "```" in prompt:
            code_score = min(1.0, code_score + 0.3)

        # Composite score
        overall = (
            self.w_len * len_score
            + self.w_code * code_score
            + self.w_reason * reason_score
            + self.w_const * const_score
        )
        overall = max(0.0, min(1.0, round(overall, 3)))

        return ComplexityBreakdown(
            length_score=round(len_score, 3),
            code_score=round(code_score, 3),
            reasoning_score=round(reason_score, 3),
            constraint_score=round(const_score, 3),
            overall_score=overall,
        )

    def classify_intent(self, prompt: str) -> tuple[IntentType, str, float]:
        """Identifies primary intent, domain, and classification confidence."""
        p_clean = prompt.strip().lower()

        # Check for presence of code or math/logic patterns first to avoid greeting false positives
        has_code = "```" in prompt or any(re.search(pat, p_clean) for pat in self.CODE_PATTERNS)
        has_planning = (
            sum(1 for pat in self.MULTI_STEP_PLANNING_PATTERNS if re.search(pat, p_clean)) >= 2
        )
        has_math = sum(1 for pat in self.REASONING_MATH_PATTERNS if re.search(pat, p_clean)) >= 2

        # 1. Multi-step architectural / collaboration planning
        if has_planning:
            return IntentType.MULTI_STEP_PLANNING, "systems", 0.90

        # 2. Code Generation / Debugging
        if has_code:
            return IntentType.CODE_GENERATION, "software_engineering", 0.88

        # 3. Mathematical & Scientific Reasoning
        math_terms = len(re.findall(r"\b(prove|proof|derive|derivation|calculate|integral|derivative|theorem|lemma|gradient|loss|softmax|matrix|eigen|probability)\b", p_clean))
        has_math = has_math or (math_terms >= 2)
        if has_math:
            return IntentType.REASONING_MATH, "mathematics_and_logic", 0.85

        # 4. Greetings / Conversational pleasantries (if not a heavy technical query)
        for pat in self.GREETING_PATTERNS:
            if re.search(pat, p_clean):
                return IntentType.GREETING, "conversational", 0.95

        # 5. Data extraction & structured output
        extraction_matches = sum(
            1 for pat in self.DATA_EXTRACTION_PATTERNS if re.search(pat, p_clean)
        )
        if extraction_matches >= 1 and any(
            k in p_clean for k in ["json", "table", "csv", "extract", "parse"]
        ):
            return IntentType.DATA_EXTRACTION, "data_engineering", 0.85

        # 6. Creative writing
        creative_matches = sum(1 for pat in self.CREATIVE_PATTERNS if re.search(pat, p_clean))
        if creative_matches >= 1:
            return IntentType.CREATIVE_WRITING, "creative", 0.80

        # 7. Factual QA vs General
        if re.search(r"\b(what|when|where|who|how|is|are|can|which|difference)\b", p_clean):
            return IntentType.FACTUAL_QA, "general_knowledge", 0.75

        return IntentType.GENERAL, "general", 0.65

    def classify(
        self,
        prompt: str,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> QueryClassification:
        """Performs full query classification and execution tier recommendation."""
        intent, domain, confidence = self.classify_intent(prompt)
        complexity = self.compute_complexity(prompt)

        # Incorporate multi-turn conversation depth into complexity
        if conversation_history and len(conversation_history) > 4:
            history_boost = min(0.15, len(conversation_history) * 0.02)
            adjusted_score = min(1.0, complexity.overall_score + history_boost)
            complexity.overall_score = round(adjusted_score, 3)

        # Tier mapping logic prioritizing intent semantics then complexity
        if intent == IntentType.GREETING:
            tier = ExecutionTier.FAST_LOCAL
            rationale = (
                "Low complexity conversational greeting; served via zero-cost local/fast engine."
            )
        elif intent == IntentType.MULTI_STEP_PLANNING or complexity.overall_score >= 0.80:
            tier = ExecutionTier.MULTI_AGENT
            rationale = "High complexity multi-component task; requires collaborative multi-agent orchestration."
        elif (
            intent in (IntentType.CODE_GENERATION, IntentType.REASONING_MATH)
            or complexity.overall_score >= 0.55
        ):
            tier = ExecutionTier.FRONTIER_REASONING
            rationale = "Complex reasoning, algorithm synthesis, or strict constraints; routed to frontier tier."
        elif (
            intent
            in (IntentType.FACTUAL_QA, IntentType.DATA_EXTRACTION, IntentType.CREATIVE_WRITING)
            or complexity.overall_score >= 0.20
        ):
            tier = ExecutionTier.BALANCED
            rationale = "Standard informational or task query; routed to balanced tier model."
        else:
            tier = ExecutionTier.FAST_LOCAL
            rationale = "Simple general query; served via zero-cost local/fast engine."

        snippet = prompt.strip()[:60] + ("..." if len(prompt.strip()) > 60 else "")

        return QueryClassification(
            prompt_snippet=snippet,
            intent=intent,
            domain=domain,
            complexity=complexity,
            recommended_tier=tier,
            confidence=confidence,
            rationale=rationale,
        )


_default_classifier: QueryClassifier | None = None


def get_query_classifier() -> QueryClassifier:
    """Singleton getter for QueryClassifier."""
    global _default_classifier
    if _default_classifier is None:
        _default_classifier = QueryClassifier()
    return _default_classifier
