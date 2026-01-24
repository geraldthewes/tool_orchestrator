"""
Tests for DSPy Optimization Metrics.

Tests cover tool selection accuracy and combined orchestration metrics.
"""

from unittest.mock import MagicMock

import dspy
import pytest

from src.prompts.optimization.metrics import (
    orchestration_quality,
    orchestration_quality_with_tools,
    tool_selection_accuracy,
)


class TestToolSelectionAccuracy:
    """Tests for the tool_selection_accuracy metric."""

    def test_no_tool_expectation_returns_one(self):
        """When example has no tool field, metric should return 1.0."""
        example = dspy.Example(question="What is 2+2?", answer="4")
        prediction = MagicMock()
        prediction.tool = "calculate"

        score = tool_selection_accuracy(example, prediction)
        assert score == 1.0

    def test_no_tool_predicted_returns_zero(self):
        """When tools expected but none predicted, return 0.0."""
        example = dspy.Example(question="What is 2+2?", answer="4", tool="calculate")
        prediction = MagicMock()
        prediction.tool = None
        prediction.tools = None

        score = tool_selection_accuracy(example, prediction)
        assert score == 0.0

    def test_exact_single_tool_match(self):
        """Exact match on single tool should return 1.0."""
        example = dspy.Example(
            question="What is 2+2?", answer="4", tool="calculate"
        ).with_inputs("question")
        prediction = MagicMock()
        prediction.tool = "calculate"
        prediction.tools = None

        score = tool_selection_accuracy(example, prediction)
        assert score == 1.0

    def test_wrong_single_tool(self):
        """Wrong tool should return 0.0 (no intersection)."""
        example = dspy.Example(
            question="What is 2+2?", answer="4", tool="calculate"
        ).with_inputs("question")
        prediction = MagicMock()
        prediction.tool = "web_search"
        prediction.tools = None

        score = tool_selection_accuracy(example, prediction)
        assert score == 0.0

    def test_exact_multi_tool_match(self):
        """Exact match on multiple tools should return 1.0."""
        example = dspy.Example(
            question="Search for X and calculate Y",
            answer="result",
            tools=["web_search", "calculate"],
        ).with_inputs("question")
        prediction = MagicMock()
        prediction.tool = None
        prediction.tools = ["calculate", "web_search"]  # Order shouldn't matter

        score = tool_selection_accuracy(example, prediction)
        assert score == 1.0

    def test_partial_tool_match(self):
        """Partial match should return partial credit (IoU)."""
        example = dspy.Example(
            question="Do multiple things",
            answer="result",
            tools=["web_search", "calculate"],
        ).with_inputs("question")
        prediction = MagicMock()
        prediction.tool = None
        prediction.tools = ["web_search"]

        # IoU = 1 intersection / 2 union = 0.5
        score = tool_selection_accuracy(example, prediction)
        assert score == 0.5

    def test_extra_predicted_tool(self):
        """Extra predicted tool should reduce score (IoU penalizes extras)."""
        example = dspy.Example(
            question="Search for X",
            answer="result",
            tools=["web_search"],
        ).with_inputs("question")
        prediction = MagicMock()
        prediction.tool = None
        prediction.tools = ["web_search", "calculate"]

        # IoU = 1 intersection / 2 union = 0.5
        score = tool_selection_accuracy(example, prediction)
        assert score == 0.5

    def test_tools_field_takes_precedence_over_tool(self):
        """When example has both 'tools' and 'tool', 'tools' should take precedence."""
        example = dspy.Example(
            question="Do stuff",
            answer="result",
            tools=["web_search", "calculate"],
            tool="python_execute",  # Should be ignored
        ).with_inputs("question")
        prediction = MagicMock()
        prediction.tool = None
        prediction.tools = ["web_search", "calculate"]

        score = tool_selection_accuracy(example, prediction)
        assert score == 1.0

    def test_prediction_tools_precedence(self):
        """When prediction has both 'tools' and 'tool', 'tools' should take precedence."""
        example = dspy.Example(
            question="Do stuff", answer="result", tool="web_search"
        ).with_inputs("question")
        prediction = MagicMock()
        prediction.tool = "python_execute"  # Should be ignored
        prediction.tools = ["web_search"]

        score = tool_selection_accuracy(example, prediction)
        assert score == 1.0

    def test_string_tool_normalized_to_list(self):
        """String tool values should be normalized to lists."""
        example = dspy.Example(
            question="Search for X",
            answer="result",
            tool="web_search",
        ).with_inputs("question")
        prediction = MagicMock()
        prediction.tool = "web_search"
        prediction.tools = None

        score = tool_selection_accuracy(example, prediction)
        assert score == 1.0


class TestOrchestrationQualityWithTools:
    """Tests for the combined orchestration_quality_with_tools metric."""

    def test_both_scores_high(self):
        """High tool and answer scores should result in high combined score."""
        example = dspy.Example(
            question="What is 2+2?",
            answer="The answer is 4",
            tool="calculate",
        ).with_inputs("question")
        prediction = MagicMock()
        prediction.answer = "The answer is 4"
        prediction.tool = "calculate"
        prediction.tools = None

        score = orchestration_quality_with_tools(example, prediction)
        assert score == 1.0

    def test_correct_answer_wrong_tool(self):
        """Correct answer but wrong tool should result in low score."""
        example = dspy.Example(
            question="What is 2+2?",
            answer="The answer is 4",
            tool="calculate",
        ).with_inputs("question")
        prediction = MagicMock()
        prediction.answer = "The answer is 4"
        prediction.tool = "web_search"
        prediction.tools = None

        score = orchestration_quality_with_tools(example, prediction)
        # answer_score = 1.0, tool_score = 0.0, combined = 0.0
        assert score == 0.0

    def test_correct_tool_wrong_answer(self):
        """Correct tool but wrong answer should result in low score."""
        example = dspy.Example(
            question="What is 2+2?",
            answer="The answer is 4",
            tool="calculate",
        ).with_inputs("question")
        prediction = MagicMock()
        prediction.answer = "I don't know"
        prediction.tool = "calculate"
        prediction.tools = None

        score = orchestration_quality_with_tools(example, prediction)
        # tool_score = 1.0, answer_score = low, combined = low
        assert score < 0.5

    def test_no_tool_expected_answer_only(self):
        """When no tool expected, combined score equals answer score."""
        example = dspy.Example(
            question="What is 2+2?",
            answer="The answer is 4",
        ).with_inputs("question")
        prediction = MagicMock()
        prediction.answer = "The answer is 4"
        prediction.tool = None
        prediction.tools = None

        combined_score = orchestration_quality_with_tools(example, prediction)
        answer_score = orchestration_quality(example, prediction)

        # tool_score = 1.0 (no expectation), so combined = answer_score * 1.0
        assert combined_score == answer_score

    def test_partial_tool_partial_answer(self):
        """Partial matches on both should multiply together."""
        example = dspy.Example(
            question="Search and calculate",
            answer="result one two three",
            tools=["web_search", "calculate"],
        ).with_inputs("question")
        prediction = MagicMock()
        prediction.answer = "result one"  # Partial match
        prediction.tool = None
        prediction.tools = ["web_search"]  # Partial match

        score = orchestration_quality_with_tools(example, prediction)
        # tool_score = 0.5, answer_score partial
        # Combined should be < max(tool_score, answer_score)
        tool_score = tool_selection_accuracy(example, prediction)
        answer_score = orchestration_quality(example, prediction)
        expected = tool_score * answer_score
        assert score == pytest.approx(expected)

    def test_multiplicative_prevents_gaming(self):
        """Verify that gaming with one metric is prevented by the other."""
        # Game 1: Try to get high score with only correct tool
        example = dspy.Example(
            question="Calculate 2+2",
            answer="4",
            tool="calculate",
        ).with_inputs("question")

        prediction_wrong_answer = MagicMock()
        prediction_wrong_answer.answer = "completely wrong unrelated text"
        prediction_wrong_answer.tool = "calculate"
        prediction_wrong_answer.tools = None

        score_wrong_answer = orchestration_quality_with_tools(
            example, prediction_wrong_answer
        )

        # Game 2: Try to get high score with only correct answer
        prediction_wrong_tool = MagicMock()
        prediction_wrong_tool.answer = "4"
        prediction_wrong_tool.tool = "wrong_tool"
        prediction_wrong_tool.tools = None

        score_wrong_tool = orchestration_quality_with_tools(
            example, prediction_wrong_tool
        )

        # Both gaming attempts should fail (score < 0.5)
        assert score_wrong_answer < 0.5
        assert score_wrong_tool < 0.5


class TestOrchestrationQuality:
    """Tests for the base orchestration_quality metric."""

    def test_exact_answer_match(self):
        """Exact answer match should return 1.0."""
        example = dspy.Example(
            question="What is 2+2?",
            answer="The answer is 4",
        ).with_inputs("question")
        prediction = MagicMock()
        prediction.answer = "The answer is 4"

        score = orchestration_quality(example, prediction)
        assert score == 1.0

    def test_no_answer_predicted(self):
        """No answer predicted should return 0.0."""
        example = dspy.Example(
            question="What is 2+2?",
            answer="4",
        ).with_inputs("question")
        prediction = MagicMock()
        prediction.answer = None

        score = orchestration_quality(example, prediction)
        assert score == 0.0

    def test_keyword_match(self):
        """Keyword matching should work when expected_keywords provided."""
        example = dspy.Example(
            question="What is Python?",
            expected_keywords=["programming", "language"],
        ).with_inputs("question")
        prediction = MagicMock()
        prediction.answer = "Python is a programming language"

        score = orchestration_quality(example, prediction)
        assert score == 1.0

    def test_partial_keyword_match(self):
        """Partial keyword match should return partial score."""
        example = dspy.Example(
            question="What is Python?",
            expected_keywords=["programming", "language", "snake"],
        ).with_inputs("question")
        prediction = MagicMock()
        prediction.answer = "Python is a programming language"

        score = orchestration_quality(example, prediction)
        # 2/3 keywords found
        assert score == pytest.approx(2 / 3)
