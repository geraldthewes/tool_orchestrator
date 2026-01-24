"""
DSPy Optimization Metrics for ToolOrchestra.

Provides metric functions for evaluating DSPy module performance
during optimization.
"""

import logging
from typing import Any

import dspy

logger = logging.getLogger(__name__)


def routing_accuracy(example: dspy.Example, prediction: Any, trace=None) -> float:
    """
    Metric for evaluating query routing accuracy.

    Compares predicted routing decision with expected decision.

    Args:
        example: DSPy Example with expected values
        prediction: Model prediction with routing decision
        trace: Optional trace information

    Returns:
        1.0 if routing decision matches expected, 0.0 otherwise
    """
    expected_needs_tools = example.get("needs_tools", None)
    if expected_needs_tools is None:
        logger.warning("Example missing 'needs_tools' field")
        return 0.0

    # Get predicted value
    predicted_needs_tools = getattr(prediction, "needs_tools", None)
    if predicted_needs_tools is None:
        return 0.0

    # Handle string "True"/"False"
    if isinstance(predicted_needs_tools, str):
        predicted_needs_tools = predicted_needs_tools.lower() in ("true", "yes", "1")

    if isinstance(expected_needs_tools, str):
        expected_needs_tools = expected_needs_tools.lower() in ("true", "yes", "1")

    return 1.0 if predicted_needs_tools == expected_needs_tools else 0.0


def routing_with_reasoning_quality(
    example: dspy.Example, prediction: Any, trace=None
) -> float:
    """
    Metric for evaluating routing with reasoning quality.

    Considers both the routing decision accuracy and whether
    reasoning is provided.

    Args:
        example: DSPy Example with expected values
        prediction: Model prediction
        trace: Optional trace information

    Returns:
        Score between 0.0 and 1.0
    """
    # Base accuracy
    accuracy = routing_accuracy(example, prediction, trace)

    # Bonus for having reasoning
    reasoning = getattr(prediction, "reasoning", None)
    reasoning_bonus = 0.1 if reasoning and len(reasoning.strip()) > 10 else 0.0

    return min(1.0, accuracy + reasoning_bonus)


def orchestration_quality(example: dspy.Example, prediction: Any, trace=None) -> float:
    """
    Metric for evaluating orchestration answer quality.

    Evaluates whether the final answer is relevant and complete.

    Args:
        example: DSPy Example with expected answer or keywords
        prediction: Model prediction with answer
        trace: Optional trace information

    Returns:
        Score between 0.0 and 1.0
    """
    expected_answer = example.get("answer", None)
    expected_keywords = example.get("expected_keywords", [])

    predicted_answer = getattr(prediction, "answer", None)
    if predicted_answer is None:
        # Debug: Log what we actually received
        attrs = [a for a in dir(prediction) if not a.startswith("_")]
        logger.warning(
            f"Prediction missing 'answer' field. "
            f"Type: {type(prediction).__name__}, "
            f"Attrs: {attrs}"
        )
        # Also log if there's any useful content
        if hasattr(prediction, "__dict__"):
            logger.debug(f"Prediction dict: {prediction.__dict__}")
        return 0.0

    predicted_answer_lower = predicted_answer.lower()

    # If expected answer is provided, check for similarity
    if expected_answer:
        expected_lower = expected_answer.lower()
        # Simple overlap scoring
        expected_words = set(expected_lower.split())
        predicted_words = set(predicted_answer_lower.split())
        if not expected_words:
            return 0.0
        overlap = len(expected_words & predicted_words) / len(expected_words)
        return overlap

    # If keywords are provided, check for presence
    if expected_keywords:
        found = sum(
            1 for kw in expected_keywords if kw.lower() in predicted_answer_lower
        )
        return found / len(expected_keywords)

    # If no expected values, return 0.5 for non-empty answer
    return 0.5 if predicted_answer.strip() else 0.0


def orchestration_completeness(
    example: dspy.Example, prediction: Any, trace=None
) -> float:
    """
    Metric for evaluating orchestration completeness.

    Checks if the answer addresses the question without being too brief.

    Args:
        example: DSPy Example with question
        prediction: Model prediction with answer
        trace: Optional trace information

    Returns:
        Score between 0.0 and 1.0
    """
    predicted_answer = getattr(prediction, "answer", None)
    if not predicted_answer:
        return 0.0

    # Penalize very short answers
    word_count = len(predicted_answer.split())
    if word_count < 5:
        return 0.3
    elif word_count < 20:
        return 0.6
    else:
        return 1.0
