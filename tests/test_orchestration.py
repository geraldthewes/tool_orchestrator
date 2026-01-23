"""
Tests for ToolOrchestra orchestration.

These tests cover the orchestration logic using mocked DSPy components.
"""

from unittest.mock import Mock, patch, MagicMock

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


class TestOrchestrator:
    """Tests for ToolOrchestrator."""

    def test_initialization(self):
        """Test orchestrator initialization."""
        orchestrator = ToolOrchestrator(max_steps=5, verbose=True)
        assert orchestrator.max_steps == 5
        assert orchestrator.verbose is True
        assert len(orchestrator.steps) == 0

    def test_initialization_with_execution_id(self):
        """Test orchestrator initialization with execution_id."""
        orchestrator = ToolOrchestrator(execution_id="exec-test123")
        assert orchestrator.execution_id == "exec-test123"

    def test_initialization_without_execution_id(self):
        """Test orchestrator initialization without execution_id defaults to None."""
        orchestrator = ToolOrchestrator()
        assert orchestrator.execution_id is None

    def test_get_trace_empty(self):
        """Test getting trace with no steps."""
        orchestrator = ToolOrchestrator()
        trace = orchestrator.get_trace()
        assert trace == []

    def test_get_trace_with_steps(self):
        """Test getting trace with steps."""
        orchestrator = ToolOrchestrator()
        # Directly set steps on the underlying module
        orchestrator._module.steps = [
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
                action="Final Answer",
                is_final=True,
                final_answer="The answer is 2",
            ),
        ]

        trace = orchestrator.get_trace()

        assert len(trace) == 2
        assert trace[0]["step"] == 1
        assert trace[0]["reasoning"] == "First step"
        assert trace[1]["is_final"] is True

    @patch("src.prompts.modules.orchestrator.get_orchestrator_lm")
    @patch("src.prompts.modules.orchestrator.dspy.context")
    def test_run_with_mocked_dspy(self, mock_context, mock_get_lm):
        """Test orchestration run with mocked DSPy."""
        mock_lm = MagicMock()
        mock_get_lm.return_value = mock_lm

        mock_context.return_value.__enter__ = Mock()
        mock_context.return_value.__exit__ = Mock(return_value=False)

        orchestrator = ToolOrchestrator()

        # Mock the react module
        mock_react_result = Mock()
        mock_react_result.answer = "The answer is 42"
        orchestrator._module.react = Mock(return_value=mock_react_result)

        result = orchestrator.run("What is the meaning of life?")

        assert "42" in result

    @patch("src.prompts.modules.orchestrator.get_orchestrator_lm")
    @patch("src.prompts.modules.orchestrator.dspy.context")
    def test_run_handles_errors(self, mock_context, mock_get_lm):
        """Test orchestration handles errors gracefully."""
        mock_lm = MagicMock()
        mock_get_lm.return_value = mock_lm

        mock_context.return_value.__enter__ = Mock()
        mock_context.return_value.__exit__ = Mock(return_value=False)

        orchestrator = ToolOrchestrator()

        # Mock react to raise an exception
        orchestrator._module.react = Mock(side_effect=Exception("LLM error"))

        result = orchestrator.run("Test query")

        assert "Error" in result or "error" in result.lower()

    def test_delegate_handlers_backward_compat(self):
        """Test delegate_handlers property for backward compatibility."""
        orchestrator = ToolOrchestrator()
        # Should return empty dict for backward compatibility
        assert orchestrator.delegate_handlers == {}


class TestToolHandlers:
    """Tests for tool handler integration."""

    def test_dspy_tools_are_registered(self):
        """Test that DSPy tools are registered in the module."""
        orchestrator = ToolOrchestrator()
        module = orchestrator._module

        # Should have tools registered
        assert len(module._tools) > 0

        # Check that common tools are present by checking function names
        tool_names = [t.__name__ for t in module._tools]
        assert "calculate" in tool_names
        assert "web_search" in tool_names
        assert "python_execute" in tool_names

    def test_delegate_tools_are_registered(self):
        """Test that delegate tools are registered."""
        orchestrator = ToolOrchestrator()
        module = orchestrator._module

        tool_names = [t.__name__ for t in module._tools]

        # Check for delegate tools (ask_*)
        delegate_tools = [name for name in tool_names if name.startswith("ask_")]
        assert len(delegate_tools) > 0


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


class TestToolExecution:
    """Tests for tool execution via DSPy."""

    def test_calculate_tool_function(self):
        """Test the calculate tool works through DSPy adapter."""
        from src.prompts.modules.orchestrator import create_dspy_tool
        from src.tools.math_solver import _handle_calculate, format_result_for_llm

        # Use _handle_calculate which expects a dict, not calculate which expects a string
        tool = create_dspy_tool(
            name="calculate",
            description="Calculate mathematical expressions",
            handler=_handle_calculate,
            formatter=format_result_for_llm,
        )

        result = tool(expression="10 * 5")
        assert "50" in result

    def test_tool_error_handling(self):
        """Test that tool errors are handled gracefully."""
        from src.prompts.modules.orchestrator import create_dspy_tool

        def failing_handler(params):
            raise ValueError("Test error")

        tool = create_dspy_tool(
            name="failing_tool",
            description="A tool that fails",
            handler=failing_handler,
            formatter=str,
        )

        result = tool()
        assert "error" in result.lower()
