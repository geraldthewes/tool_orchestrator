"""
Tests for ToolOrchestra orchestration.

These tests cover the orchestration logic using mocked OpenAI client.
"""

import json
from unittest.mock import Mock, patch

from src.orchestrator import ToolOrchestrator, OrchestrationStep, ToolResult


class TestOrchestrationStep:
    """Tests for OrchestrationStep dataclass."""

    def test_default_values(self):
        """Test default values for OrchestrationStep."""
        step = OrchestrationStep(step_number=1)
        assert step.step_number == 1
        assert step.reasoning is None
        assert step.action is None
        assert step.action_input is None
        assert step.observation is None
        assert step.is_final is False
        assert step.final_answer is None

    def test_final_step(self):
        """Test final answer step."""
        step = OrchestrationStep(
            step_number=3,
            reasoning="I have enough information",
            action="Final Answer",
            is_final=True,
            final_answer="The answer is 42",
        )
        assert step.is_final is True
        assert step.final_answer == "The answer is 42"


class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_success_result(self):
        """Test successful tool result."""
        result = ToolResult(
            tool_name="calculate",
            success=True,
            result="2 + 2 = 4",
            raw_data={"expression": "2 + 2", "result": 4},
        )
        assert result.success is True
        assert result.tool_name == "calculate"
        assert "4" in result.result

    def test_failure_result(self):
        """Test failed tool result."""
        result = ToolResult(
            tool_name="web_search",
            success=False,
            result="Connection timeout",
        )
        assert result.success is False
        assert "timeout" in result.result.lower()


def _make_tool_call_text(name: str, arguments: dict) -> str:
    """Build a <tool_call> XML block for mock LLM output."""
    return (
        "<tool_call>\n"
        + json.dumps({"name": name, "arguments": arguments})
        + "\n</tool_call>"
    )


def _make_mock_response(content: str, usage: Mock | None = None) -> Mock:
    """Create a mock chat completion response returning plain text."""
    msg = Mock()
    msg.content = content
    response = Mock()
    response.choices = [Mock(message=msg)]
    response.usage = usage
    return response


class TestOrchestrator:
    """Tests for ToolOrchestrator."""

    @patch("src.orchestration.loop.OpenAI")
    def test_initialization(self, mock_openai_cls):
        """Test orchestrator initialization."""
        orchestrator = ToolOrchestrator(max_steps=5, verbose=True)
        assert orchestrator.max_steps == 5
        assert orchestrator.verbose is True
        assert len(orchestrator.steps) == 0

    @patch("src.orchestration.loop.OpenAI")
    def test_initialization_with_execution_id(self, mock_openai_cls):
        """Test orchestrator initialization with execution_id."""
        orchestrator = ToolOrchestrator(execution_id="exec-test123")
        assert orchestrator.execution_id == "exec-test123"

    @patch("src.orchestration.loop.OpenAI")
    def test_initialization_without_execution_id(self, mock_openai_cls):
        """Test orchestrator initialization without execution_id defaults to None."""
        orchestrator = ToolOrchestrator()
        assert orchestrator.execution_id is None

    @patch("src.orchestration.loop.OpenAI")
    def test_get_trace_empty(self, mock_openai_cls):
        """Test getting trace with no steps."""
        orchestrator = ToolOrchestrator()
        trace = orchestrator.get_trace()
        assert trace == []

    @patch("src.orchestration.loop.OpenAI")
    def test_get_trace_with_steps(self, mock_openai_cls):
        """Test getting trace with steps."""
        orchestrator = ToolOrchestrator()
        # Directly set steps on the underlying loop
        orchestrator._loop.steps = [
            OrchestrationStep(
                step_number=1,
                reasoning="First step",
                action="calculate",
                action_input={"expression": "1 + 1"},
                observation="1 + 1 = 2",
            ),
            OrchestrationStep(
                step_number=2,
                reasoning="Final step",
                action="answer",
                is_final=True,
                final_answer="The answer is 2",
            ),
        ]

        trace = orchestrator.get_trace()

        assert len(trace) == 2
        assert trace[0]["step"] == 1
        assert trace[0]["reasoning"] == "First step"
        assert trace[1]["is_final"] is True

    @patch("src.orchestration.loop.OpenAI")
    def test_run_with_answer_tool(self, mock_openai_cls):
        """Test orchestration run with mocked OpenAI client."""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client

        tc_text = _make_tool_call_text("answer", {"content": "The answer is 42"})
        content = f"<think>Simple question</think>\n\n{tc_text}"
        response = _make_mock_response(content)
        mock_client.chat.completions.create.return_value = response

        orchestrator = ToolOrchestrator()
        result = orchestrator.run("What is the meaning of life?")

        assert "42" in result

    @patch("src.orchestration.loop.OpenAI")
    def test_run_handles_errors(self, mock_openai_cls):
        """Test orchestration handles errors gracefully."""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("LLM error")

        orchestrator = ToolOrchestrator()
        result = orchestrator.run("Test query")

        assert "Error" in result or "error" in result.lower()

    @patch("src.orchestration.loop.OpenAI")
    def test_run_populates_steps(self, mock_openai_cls):
        """Test that run() populates steps from tool calls."""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client

        # Step 1: calculate
        calc_tc = _make_tool_call_text("calculate", {"expression": "6 * 7"})
        calc_content = f"<think>I need to calculate</think>\n\n{calc_tc}"
        calc_response = _make_mock_response(calc_content)

        # Step 2: answer
        answer_tc = _make_tool_call_text("answer", {"content": "The answer is 42"})
        answer_content = f"<think>Now I have the answer</think>\n\n{answer_tc}"
        answer_response = _make_mock_response(answer_content)

        mock_client.chat.completions.create.side_effect = [
            calc_response,
            answer_response,
        ]

        orchestrator = ToolOrchestrator()
        result = orchestrator.run("What is 6 times 7?")

        assert "42" in result
        assert len(orchestrator.steps) == 2
        assert orchestrator.steps[0].action == "calculate"
        assert orchestrator.steps[1].is_final is True

    @patch("src.orchestration.loop.OpenAI")
    def test_run_no_tool_calls_returns_content(self, mock_openai_cls):
        """Test that run handles LLM returning content without tool calls."""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client

        response = _make_mock_response("Direct answer")
        mock_client.chat.completions.create.return_value = response

        orchestrator = ToolOrchestrator()
        result = orchestrator.run("Simple question")

        assert result == "Direct answer"

    @patch("src.orchestration.loop.OpenAI")
    def test_delegate_handlers_backward_compat(self, mock_openai_cls):
        """Test delegate_handlers property for backward compatibility."""
        orchestrator = ToolOrchestrator()
        assert orchestrator.delegate_handlers == {}

    @patch("src.orchestration.loop.OpenAI")
    def test_llm_client_backward_compat(self, mock_openai_cls):
        """Test llm_client property for backward compatibility."""
        orchestrator = ToolOrchestrator()
        assert orchestrator.llm_client is None
        # Setting should not raise
        orchestrator.llm_client = "ignored"
        assert orchestrator.llm_client is None

    @patch("src.orchestration.loop.OpenAI")
    def test_kwargs_ignored_for_backward_compat(self, mock_openai_cls):
        """Test that unknown kwargs are ignored (backward compat)."""
        # Should not raise
        orchestrator = ToolOrchestrator(llm_client="ignored_value")
        assert orchestrator.llm_client is None


class TestRunQueryConvenience:
    """Tests for the run_query convenience function."""

    @patch("src.orchestrator.ToolOrchestrator")
    def test_run_query_creates_orchestrator(self, mock_orchestrator_class):
        """Test that run_query creates an orchestrator and runs."""
        from src.orchestrator import run_query

        mock_instance = Mock()
        mock_instance.run.return_value = "test result"
        mock_orchestrator_class.return_value = mock_instance

        result = run_query("test query", verbose=True)

        mock_orchestrator_class.assert_called_once_with(verbose=True)
        mock_instance.run.assert_called_once_with("test query")
        assert result == "test result"
