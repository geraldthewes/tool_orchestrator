"""Tests for the orchestration loop."""

import json
from unittest.mock import Mock, patch

from src.orchestration.loop import (
    OrchestrationLoop,
    OrchestrationStep,
    SYSTEM_PROMPT,
    MAX_CONSECUTIVE_CALLS,
)


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
            action="answer",
            is_final=True,
            final_answer="The answer is 42",
        )
        assert step.is_final is True
        assert step.final_answer == "The answer is 42"


class TestMessageBuilding:
    """Tests for message construction."""

    @patch("src.orchestration.loop.OpenAI")
    def test_build_messages_no_observations(self, mock_openai_cls):
        """Without observations or system_prompt, messages use base SYSTEM_PROMPT."""
        loop = OrchestrationLoop()
        messages = loop._build_messages("What is 2+2?")

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == SYSTEM_PROMPT
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "What is 2+2?"

    @patch("src.orchestration.loop.OpenAI")
    def test_build_messages_with_system_prompt(self, mock_openai_cls):
        """Custom system_prompt should be used when provided."""
        loop = OrchestrationLoop()
        custom = SYSTEM_PROMPT + "\n\n# Tools\n<tools>...</tools>"
        messages = loop._build_messages("Hello", system_prompt=custom)

        assert messages[0]["content"] == custom

    @patch("src.orchestration.loop.OpenAI")
    def test_build_messages_with_observations(self, mock_openai_cls):
        """With observations, user message should include context."""
        loop = OrchestrationLoop()
        loop._buffers.add_observation("calculate", "2 + 2 = 4", 1)

        messages = loop._build_messages("What is 2+2?")

        assert len(messages) == 2
        user_content = messages[1]["content"]
        assert "What is 2+2?" in user_content
        assert "Previous observations" in user_content
        assert "2 + 2 = 4" in user_content


class TestThinkBlockExtraction:
    """Tests for <think> block parsing."""

    def test_extract_think_block(self):
        """Should extract content from <think> tags."""
        content = "<think>I need to calculate this</think>The answer is 4."
        reasoning = OrchestrationLoop._extract_think_block(content)
        assert reasoning == "I need to calculate this"

    def test_extract_think_block_multiline(self):
        """Should handle multiline think blocks."""
        content = "<think>\nStep 1: think\nStep 2: plan\n</think>Answer."
        reasoning = OrchestrationLoop._extract_think_block(content)
        assert "Step 1: think" in reasoning
        assert "Step 2: plan" in reasoning

    def test_no_think_block(self):
        """Should return None when no think block present."""
        content = "Just a plain answer."
        reasoning = OrchestrationLoop._extract_think_block(content)
        assert reasoning is None

    def test_strip_think_block(self):
        """Should remove think tags from content."""
        content = "<think>reasoning here</think>The answer is 42."
        stripped = OrchestrationLoop._strip_think_block(content)
        assert "<think>" not in stripped
        assert "The answer is 42." in stripped


class TestToolCallParsing:
    """Tests for <tool_call> XML parsing."""

    def test_parse_tool_call_basic(self):
        """Should parse a basic tool_call block."""
        content = (
            '<tool_call>\n{"name": "calculate", "arguments": {"expression": "3^8"}}\n'
            "</tool_call>"
        )
        result = OrchestrationLoop._parse_tool_call(content)
        assert result is not None
        name, args = result
        assert name == "calculate"
        assert args == {"expression": "3^8"}

    def test_parse_tool_call_with_think(self):
        """Should parse tool_call even when preceded by <think> block."""
        content = (
            "<think>I need to search</think>\n\n"
            '<tool_call>\n{"name": "web_search", "arguments": {"query": "test"}}\n'
            "</tool_call>"
        )
        result = OrchestrationLoop._parse_tool_call(content)
        assert result is not None
        name, args = result
        assert name == "web_search"
        assert args == {"query": "test"}

    def test_parse_tool_call_no_match(self):
        """Should return None when no tool_call present."""
        content = "Just a plain text answer."
        assert OrchestrationLoop._parse_tool_call(content) is None

    def test_parse_tool_call_invalid_json(self):
        """Should return None for malformed JSON inside tool_call."""
        content = "<tool_call>\nnot valid json\n</tool_call>"
        assert OrchestrationLoop._parse_tool_call(content) is None

    def test_parse_tool_call_missing_name(self):
        """Should return None when name is empty."""
        content = '<tool_call>\n{"name": "", "arguments": {}}\n</tool_call>'
        assert OrchestrationLoop._parse_tool_call(content) is None

    def test_parse_tool_call_string_arguments(self):
        """Should handle arguments as a JSON string (double-encoded)."""
        inner_args = json.dumps({"expression": "2+2"})
        content = (
            "<tool_call>\n"
            + json.dumps({"name": "calculate", "arguments": inner_args})
            + "\n</tool_call>"
        )
        result = OrchestrationLoop._parse_tool_call(content)
        assert result is not None
        _, args = result
        assert args == {"expression": "2+2"}


class TestStripTags:
    """Tests for _strip_tags helper."""

    def test_strip_all_tags(self):
        """Should strip think, tool_call, and unwrap message tags."""
        content = (
            "<think>reasoning</think>"
            "<message>The answer is 42</message>"
            '<tool_call>{"name": "answer", "arguments": {}}</tool_call>'
        )
        stripped = OrchestrationLoop._strip_tags(content)
        assert "<think>" not in stripped
        assert "<tool_call>" not in stripped
        assert "<message>" not in stripped
        assert "The answer is 42" in stripped

    def test_strip_tags_plain_text(self):
        """Plain text should pass through unchanged."""
        content = "The answer is 42."
        assert OrchestrationLoop._strip_tags(content) == content


class TestAntiRepetition:
    """Tests for anti-repetition tracking."""

    @patch("src.orchestration.loop.OpenAI")
    def test_no_exclusion_below_threshold(self, mock_openai_cls):
        """No tools should be excluded below the consecutive call threshold."""
        loop = OrchestrationLoop()
        loop._call_history = ["calculate"]
        assert loop._get_excluded_tools() == set()

    @patch("src.orchestration.loop.OpenAI")
    def test_exclude_after_consecutive_calls(self, mock_openai_cls):
        """Tool should be excluded after MAX_CONSECUTIVE_CALLS."""
        loop = OrchestrationLoop()
        loop._call_history = ["calculate"] * MAX_CONSECUTIVE_CALLS
        excluded = loop._get_excluded_tools()
        assert "calculate" in excluded

    @patch("src.orchestration.loop.OpenAI")
    def test_no_exclusion_mixed_calls(self, mock_openai_cls):
        """No exclusion when recent calls are mixed."""
        loop = OrchestrationLoop()
        loop._call_history = ["calculate", "web_search"]
        assert loop._get_excluded_tools() == set()

    @patch("src.orchestration.loop.OpenAI")
    def test_answer_never_excluded(self, mock_openai_cls):
        """The answer tool should never be excluded."""
        loop = OrchestrationLoop()
        loop._call_history = ["answer"] * MAX_CONSECUTIVE_CALLS
        excluded = loop._get_excluded_tools()
        assert "answer" not in excluded


class TestLoopExecution:
    """Tests for end-to-end loop execution."""

    @patch("src.orchestration.loop.OpenAI")
    def test_answer_tool_terminates_loop(self, mock_openai_cls):
        """Calling the answer tool should terminate the loop."""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client

        # LLM returns text with <tool_call> for the answer tool
        tc_text = _make_tool_call_text("answer", {"content": "The answer is 42"})
        content = f"<think>I know the answer</think>\n\n{tc_text}"
        response = _make_mock_response(content)
        mock_client.chat.completions.create.return_value = response

        loop = OrchestrationLoop()
        result = loop.run("What is the meaning of life?")

        assert result.answer == "The answer is 42"
        assert len(result.steps) == 1
        assert result.steps[0].is_final is True
        assert result.steps[0].action == "answer"

    @patch("src.orchestration.loop.OpenAI")
    def test_no_tool_calls_uses_content_as_answer(self, mock_openai_cls):
        """When LLM returns no tool_call tags, content is the answer."""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client

        response = _make_mock_response("The answer is simply 42.")
        mock_client.chat.completions.create.return_value = response

        loop = OrchestrationLoop()
        result = loop.run("What is 42?")

        assert result.answer == "The answer is simply 42."
        assert result.steps[0].is_final is True

    @patch("src.orchestration.loop.OpenAI")
    def test_max_steps_forces_answer(self, mock_openai_cls):
        """Max steps should force a final answer."""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client

        # LLM always calls calculate (never answers)
        tc_text = _make_tool_call_text("calculate", {"expression": "1+1"})
        content = f"<think>Let me calculate</think>\n\n{tc_text}"
        response = _make_mock_response(content)
        mock_client.chat.completions.create.return_value = response

        loop = OrchestrationLoop(max_steps=2)
        result = loop.run("What is 1+1?")

        # Should have max_steps + 1 steps (the forced answer step)
        assert len(result.steps) == 3
        assert result.steps[-1].is_final is True
        assert "Unable to determine" in result.answer or "Based on" in result.answer

    @patch("src.orchestration.loop.OpenAI")
    def test_tool_execution_with_observation(self, mock_openai_cls):
        """Tool execution should add observation to buffers."""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client

        # Step 1: LLM calls calculate
        calc_tc = _make_tool_call_text("calculate", {"expression": "3^8"})
        calc_content = f"<think>Need to calculate 3^8</think>\n\n{calc_tc}"
        calc_response = _make_mock_response(calc_content)

        # Step 2: LLM returns answer
        answer_tc = _make_tool_call_text("answer", {"content": "3^8 = 6561"})
        answer_content = f"<think>Now I have the answer</think>\n\n{answer_tc}"
        answer_response = _make_mock_response(answer_content)

        mock_client.chat.completions.create.side_effect = [
            calc_response,
            answer_response,
        ]

        loop = OrchestrationLoop()
        result = loop.run("What is 3^8?")

        assert result.answer == "3^8 = 6561"
        assert len(result.steps) == 2
        assert result.steps[0].action == "calculate"
        assert result.steps[0].observation is not None
        assert result.steps[1].is_final is True
        assert "calculate" in result.tools_used

    @patch("src.orchestration.loop.OpenAI")
    def test_llm_failure_returns_error(self, mock_openai_cls):
        """LLM call failure should produce an error answer."""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception(
            "Connection refused"
        )

        loop = OrchestrationLoop()
        result = loop.run("Hello")

        assert "Error" in result.answer or "error" in result.answer.lower()
        assert result.steps[-1].is_final is True

    @patch("src.orchestration.loop.OpenAI")
    def test_think_block_captured_as_reasoning(self, mock_openai_cls):
        """<think> blocks should be captured as step reasoning."""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client

        tc_text = _make_tool_call_text("answer", {"content": "42"})
        content = f"<think>The user wants to know the answer</think>\n\n{tc_text}"
        response = _make_mock_response(content)
        mock_client.chat.completions.create.return_value = response

        loop = OrchestrationLoop()
        result = loop.run("What is 42?")

        assert result.steps[0].reasoning == "The user wants to know the answer"

    @patch("src.orchestration.loop.OpenAI")
    def test_system_prompt_includes_tools_block(self, mock_openai_cls):
        """System prompt sent to LLM should include <tools> block."""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client

        # Return a direct answer so the loop runs one step
        response = _make_mock_response("The answer is 42.")
        mock_client.chat.completions.create.return_value = response

        loop = OrchestrationLoop()
        loop.run("What is 42?")

        # Inspect the messages sent to the LLM
        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        system_content = messages[0]["content"]

        assert "# Tools" in system_content
        assert "<tools>" in system_content
        assert "</tools>" in system_content
        assert "<tool_call>" in system_content
        assert "answer" in system_content

    @patch("src.orchestration.loop.OpenAI")
    def test_no_tools_api_parameter(self, mock_openai_cls):
        """LLM call should NOT include 'tools' or 'tool_choice' API params."""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client

        response = _make_mock_response("The answer is 42.")
        mock_client.chat.completions.create.return_value = response

        loop = OrchestrationLoop()
        loop.run("What is 42?")

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert "tools" not in call_kwargs
        assert "tool_choice" not in call_kwargs


class TestGetTrace:
    """Tests for trace generation."""

    @patch("src.orchestration.loop.OpenAI")
    def test_get_trace_format(self, mock_openai_cls):
        """Trace should return list of step dicts."""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client

        tc_text = _make_tool_call_text("answer", {"content": "42"})
        content = f"<think>thinking</think>\n\n{tc_text}"
        response = _make_mock_response(content)
        mock_client.chat.completions.create.return_value = response

        loop = OrchestrationLoop()
        loop.run("What is 42?")
        trace = loop.get_trace()

        assert len(trace) == 1
        step = trace[0]
        assert "step" in step
        assert "reasoning" in step
        assert "action" in step
        assert "action_input" in step
        assert "observation" in step
        assert "is_final" in step
        assert "final_answer" in step

    @patch("src.orchestration.loop.OpenAI")
    def test_get_trace_empty(self, mock_openai_cls):
        """Empty loop should return empty trace."""
        loop = OrchestrationLoop()
        assert loop.get_trace() == []


class TestClose:
    """Tests for resource cleanup."""

    @patch("src.orchestration.loop.OpenAI")
    def test_close_calls_client_close(self, mock_openai_cls):
        """Close should call the underlying client's close method."""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client

        loop = OrchestrationLoop()
        loop.close()

        mock_client.close.assert_called_once()
