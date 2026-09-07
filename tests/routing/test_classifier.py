"""
Tests for Query Complexity & Intent Classifier (Phase 21)
"""

import pytest

from packages.routing.classifier import (
    ExecutionTier,
    IntentType,
    QueryClassifier,
    get_query_classifier,
)


@pytest.fixture
def classifier() -> QueryClassifier:
    return get_query_classifier()


def test_classify_greeting(classifier: QueryClassifier):
    result = classifier.classify("Hello there! How are you doing today?")
    assert result.intent == IntentType.GREETING
    assert result.recommended_tier == ExecutionTier.FAST_LOCAL
    assert result.complexity.overall_score < 0.35


def test_classify_factual_qa(classifier: QueryClassifier):
    result = classifier.classify("What is the difference between RoPE and sinusoidal embeddings?")
    assert result.intent == IntentType.FACTUAL_QA
    assert result.recommended_tier in (ExecutionTier.BALANCED, ExecutionTier.FRONTIER_REASONING)
    assert result.domain == "general_knowledge"


def test_classify_code_generation(classifier: QueryClassifier):
    prompt = """
    Write a Python function `quicksort(arr: list[int]) -> list[int]` that implements in-place quicksort.
    Include unit tests with pytest and handle empty lists.
    ```python
    def quicksort(arr):
        pass
    ```
    """
    result = classifier.classify(prompt)
    assert result.intent == IntentType.CODE_GENERATION
    assert result.recommended_tier == ExecutionTier.FRONTIER_REASONING
    assert result.complexity.code_score > 0.4
    assert result.domain == "software_engineering"


def test_classify_mathematical_reasoning(classifier: QueryClassifier):
    prompt = (
        "Prove step-by-step why the gradient of the cross-entropy loss with softmax "
        "reduces to p_i - y_i. Derive every algebraic step carefully."
    )
    result = classifier.classify(prompt)
    assert result.intent == IntentType.REASONING_MATH
    assert result.recommended_tier == ExecutionTier.FRONTIER_REASONING
    assert result.complexity.reasoning_score > 0.4


def test_classify_multi_step_planning(classifier: QueryClassifier):
    prompt = (
        "Architect an end-to-end full-stack data pipeline with multi-agent orchestration. "
        "Provide a comprehensive plan with milestones, step 1 data ingestion, and decomposition."
    )
    result = classifier.classify(prompt)
    assert result.intent == IntentType.MULTI_STEP_PLANNING
    assert result.recommended_tier == ExecutionTier.MULTI_AGENT


def test_complexity_score_monotonicity(classifier: QueryClassifier):
    simple = "Hi"
    medium = "Explain how backpropagation works in neural networks."
    complex_query = (
        "Architect and implement a thread-safe distributed cache in Rust with LRU eviction, "
        "consistent hashing across 16 nodes, and formal verification. The code must compile "
        "without warnings, adhere strictly to zero-allocation principles, and pass all stress tests."
    )

    c_simple = classifier.compute_complexity(simple)
    c_med = classifier.compute_complexity(medium)
    c_complex = classifier.compute_complexity(complex_query)

    assert c_simple.overall_score < c_med.overall_score
    assert c_med.overall_score < c_complex.overall_score


def test_conversation_history_depth_boost(classifier: QueryClassifier):
    prompt = "Can you refine that?"
    res_no_history = classifier.classify(prompt, conversation_history=None)
    history = [{"role": "user", "content": f"msg {i}"} for i in range(8)]
    res_with_history = classifier.classify(prompt, conversation_history=history)

    assert res_with_history.complexity.overall_score >= res_no_history.complexity.overall_score
