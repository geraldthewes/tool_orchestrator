"""
Tests for ToolOrchestra orchestration.

These tests cover the orchestration logic using mocked LLM responses.
"""

import pytest
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


class TestOrchestrator:
    """Tests for ToolOrchestrator."""

    def test_initialization(self):
        """Test orchestrator initialization."""
        orchestrator = ToolOrchestrator(max_steps=5, verbose=True)
        assert orchestrator.max_steps == 5
        assert orchestrator.verbose is True
        assert len(orchestrator.steps) == 0

    def test_parse_thought_action_input(self):
        """Test parsing a standard response."""
        orchestrator = ToolOrchestrator()

        response = """
**Thought**: I need to calculate this expression.
**Action**: calculate
**Action Input**: {"expression": "2 + 2"}
"""
        step = orchestrator._parse_response(response)

        assert step.reasoning == "I need to calculate this expression."
        assert step.action == "calculate"
        assert step.action_input == {"expression": "2 + 2"}
        assert step.is_final is False

    def test_parse_final_answer(self):
        """Test parsing a final answer response."""
        orchestrator = ToolOrchestrator()

        response = """
**Thought**: I have calculated the result.
**Action**: Final Answer
**Action Input**: The sum of 2 and 2 is 4.
"""
        step = orchestrator._parse_response(response)

        assert step.is_final is True
        assert step.final_answer == "The sum of 2 and 2 is 4."

    def test_parse_json_with_code_block(self):
        """Test parsing JSON wrapped in code block."""
        orchestrator = ToolOrchestrator()

        response = """
**Thought**: Let me search for this.
**Action**: web_search
**Action Input**: ```json
{"query": "Python latest version"}
```
"""
        step = orchestrator._parse_response(response)

        assert step.action_input == {"query": "Python latest version"}

    def test_execute_calculate_tool(self):
        """Test executing the calculate tool."""
        orchestrator = ToolOrchestrator()

        result = orchestrator._execute_tool("calculate", {"expression": "10 * 5"})

        assert result.success is True
        assert "50" in result.result

    @patch("src.tools.python_executor.requests.post")
    def test_execute_python_tool(self, mock_post):
        """Test executing the Python executor tool."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "stdout": "Hello\n",
            "stderr": "",
            "exit_code": 0,
        }
        mock_post.return_value = mock_response

        orchestrator = ToolOrchestrator()

        result = orchestrator._execute_tool(
            "python_execute",
            {"code": "print('Hello')"},
        )

        assert result.success is True
        assert "Hello" in result.result

    def test_execute_unknown_tool(self):
        """Test handling of unknown tool."""
        orchestrator = ToolOrchestrator()

        result = orchestrator._execute_tool("nonexistent_tool", {})

        assert result.success is False
        assert "Unknown tool" in result.result

    def test_build_system_prompt(self):
        """Test system prompt generation."""
        orchestrator = ToolOrchestrator()

        prompt = orchestrator._build_system_prompt()

        # Check that key tools are mentioned
        assert "web_search" in prompt
        assert "python_execute" in prompt
        assert "calculate" in prompt
        assert "ask_reasoner" in prompt
        assert "ask_coder" in prompt

        # Check format instructions
        assert "**Thought**" in prompt
        assert "**Action**" in prompt
        assert "Final Answer" in prompt

    def test_build_messages_initial(self):
        """Test building messages for initial query."""
        orchestrator = ToolOrchestrator()

        messages = orchestrator._build_messages("What is 2 + 2?")

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "2 + 2" in messages[1]["content"]

    def test_build_messages_with_history(self):
        """Test building messages with step history."""
        orchestrator = ToolOrchestrator()

        # Add a previous step
        orchestrator.steps.append(
            OrchestrationStep(
                step_number=1,
                reasoning="I need to calculate this",
                action="calculate",
                action_input={"expression": "2 + 2"},
                observation="2 + 2 = 4",
            )
        )

        messages = orchestrator._build_messages("What is 2 + 2?")

        # Should have: system, user, assistant (step 1), user (observation)
        assert len(messages) == 4
        assert messages[2]["role"] == "assistant"
        assert messages[3]["role"] == "user"
        assert "Observation" in messages[3]["content"]

    def test_get_trace_empty(self):
        """Test getting trace with no steps."""
        orchestrator = ToolOrchestrator()

        trace = orchestrator.get_trace()

        assert trace == []

    def test_get_trace_with_steps(self):
        """Test getting trace with steps."""
        orchestrator = ToolOrchestrator()

        orchestrator.steps = [
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

    @patch("src.orchestrator.LLMClient")
    def test_run_with_mocked_llm(self, mock_llm_class):
        """Test full orchestration run with mocked LLM."""
        # Setup mock
        mock_client = Mock()
        mock_llm_class.return_value = mock_client

        # First call: return a calculation action
        # Second call: return final answer
        mock_client.call_orchestrator.side_effect = [
            {
                "success": True,
                "response": """
**Thought**: I need to calculate this.
**Action**: calculate
**Action Input**: {"expression": "5 + 5"}
""",
            },
            {
                "success": True,
                "response": """
**Thought**: I have the result.
**Action**: Final Answer
**Action Input**: The result of 5 + 5 is 10.
""",
            },
        ]

        orchestrator = ToolOrchestrator(llm_client=mock_client)
        result = orchestrator.run("What is 5 + 5?")

        assert "10" in result
        assert len(orchestrator.steps) == 2
        assert mock_client.call_orchestrator.call_count == 2

    @patch("src.orchestrator.LLMClient")
    def test_run_max_steps_reached(self, mock_llm_class):
        """Test orchestration stops at max steps."""
        mock_client = Mock()
        mock_llm_class.return_value = mock_client

        # Always return a non-final action
        mock_client.call_orchestrator.return_value = {
            "success": True,
            "response": """
**Thought**: Keep calculating.
**Action**: calculate
**Action Input**: {"expression": "1 + 1"}
""",
        }

        orchestrator = ToolOrchestrator(llm_client=mock_client, max_steps=3)
        result = orchestrator.run("Loop forever")

        assert len(orchestrator.steps) == 3
        assert "unable to complete" in result.lower()

    @patch("src.orchestrator.LLMClient")
    def test_run_llm_error(self, mock_llm_class):
        """Test handling LLM errors."""
        mock_client = Mock()
        mock_llm_class.return_value = mock_client

        mock_client.call_orchestrator.return_value = {
            "success": False,
            "error": "Connection refused",
            "response": None,
        }

        orchestrator = ToolOrchestrator(llm_client=mock_client)
        result = orchestrator.run("Test query")

        assert "Error" in result
        assert "Connection refused" in result


class TestToolHandlers:
    """Tests for tool handler mappings."""

    def test_all_tools_have_handlers(self):
        """Test that all tools have handlers and formatters."""
        orchestrator = ToolOrchestrator()

        expected_tools = [
            "web_search",
            "python_execute",
            "calculate",
            "ask_reasoner",
            "ask_coder",
            "ask_fast",
        ]

        for tool in expected_tools:
            assert tool in orchestrator.tool_handlers, f"Missing handler for {tool}"
            assert tool in orchestrator.tool_formatters, f"Missing formatter for {tool}"


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
