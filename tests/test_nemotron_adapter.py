"""
Tests for NemotronJSONAdapter.

Tests the custom adapter that handles Nemotron-Orchestrator's "final" wrapper format.
"""

import logging
from typing import Any

import dspy
import pytest

from src.prompts.adapters.nemotron_adapter import NemotronJSONAdapter


class MockSignature(dspy.Signature):
    """Mock signature for testing with reasoning and answer fields."""

    question: str = dspy.InputField()
    reasoning: str = dspy.OutputField(desc="Step-by-step reasoning")
    answer: str = dspy.OutputField(desc="Final answer")


class TestNemotronJSONAdapter:
    """Tests for NemotronJSONAdapter parsing logic."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.adapter = NemotronJSONAdapter()
        self.signature = MockSignature

    def test_parse_standard_json(self) -> None:
        """Test parsing standard JSON without 'final' wrapper."""
        completion = '{"reasoning": "I calculated 2+2", "answer": "4"}'

        result = self.adapter.parse(self.signature, completion)

        assert result["reasoning"] == "I calculated 2+2"
        assert result["answer"] == "4"

    def test_parse_final_wrapper(self) -> None:
        """Test parsing JSON with Nemotron's 'final' wrapper."""
        completion = '{"final": {"reasoning": "I looked it up", "answer": "Paris"}}'

        result = self.adapter.parse(self.signature, completion)

        assert result["reasoning"] == "I looked it up"
        assert result["answer"] == "Paris"

    def test_parse_final_wrapper_with_extra_fields(self) -> None:
        """Test that extra fields in 'final' are filtered out."""
        completion = """{
            "final": {
                "reasoning": "Step by step",
                "answer": "42",
                "extra_field": "should be ignored"
            }
        }"""

        result = self.adapter.parse(self.signature, completion)

        assert result["reasoning"] == "Step by step"
        assert result["answer"] == "42"
        assert "extra_field" not in result

    def test_parse_json_in_code_block(self) -> None:
        """Test parsing JSON embedded in markdown code block."""
        completion = """Here is the answer:
```json
{"final": {"reasoning": "Computed", "answer": "Result"}}
```
"""

        result = self.adapter.parse(self.signature, completion)

        assert result["reasoning"] == "Computed"
        assert result["answer"] == "Result"

    def test_parse_json_with_surrounding_text(self) -> None:
        """Test parsing JSON surrounded by other text."""
        completion = """Based on my analysis:
{"final": {"reasoning": "Analysis complete", "answer": "Done"}}
That's the result."""

        result = self.adapter.parse(self.signature, completion)

        assert result["reasoning"] == "Analysis complete"
        assert result["answer"] == "Done"

    def test_parse_complex_nested_final(self) -> None:
        """Test parsing real-world Nemotron response with complex content."""
        completion = """{
            "final": {
                "reasoning": "I first attempted a web search but initially omitted the query field. After correcting, I retrieved results showing the official site.",
                "answer": "**Eiffel Tower**\\n\\n- Location: Paris\\n- Height: 330m"
            }
        }"""

        result = self.adapter.parse(self.signature, completion)

        assert "web search" in result["reasoning"]
        assert "Eiffel Tower" in result["answer"]

    def test_parse_empty_completion(self) -> None:
        """Test parsing empty completion returns empty dict."""
        completion = ""

        result = self.adapter.parse(self.signature, completion)

        assert result == {}

    def test_parse_invalid_json(self) -> None:
        """Test parsing invalid JSON returns empty dict."""
        completion = "This is not JSON at all"

        result = self.adapter.parse(self.signature, completion)

        assert result == {}

    def test_parse_partial_fields(self) -> None:
        """Test parsing when only some expected fields are present."""
        completion = '{"final": {"answer": "Just the answer"}}'

        result = self.adapter.parse(self.signature, completion)

        assert result.get("answer") == "Just the answer"
        assert "reasoning" not in result

    def test_extract_json_nested_braces(self) -> None:
        """Test JSON extraction with nested braces in content."""
        completion = """{
            "final": {
                "reasoning": "Code example: {x: 1}",
                "answer": "The object has {key: value}"
            }
        }"""

        result = self.adapter.parse(self.signature, completion)

        assert "{x: 1}" in result["reasoning"]
        assert "{key: value}" in result["answer"]


class TestThinkBlockParsing:
    """Tests for parsing responses with <think>...</think> blocks."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.adapter = NemotronJSONAdapter()
        self.signature = MockSignature

    def test_parse_with_think_block(self) -> None:
        """Test parsing JSON after <think> block."""
        completion = """<think>
Let me think about this question. The user is asking about Paris.
I should provide the answer.
</think>

{"reasoning": "The capital of France is Paris", "answer": "Paris"}"""

        result = self.adapter.parse(self.signature, completion)

        assert result["reasoning"] == "The capital of France is Paris"
        assert result["answer"] == "Paris"

    def test_parse_with_think_block_and_final_wrapper(self) -> None:
        """Test parsing <think> block followed by 'final' wrapped JSON."""
        completion = """<think>
I need to calculate 2+2.
2+2 equals 4.
</think>

{"final": {"reasoning": "Simple arithmetic", "answer": "4"}}"""

        result = self.adapter.parse(self.signature, completion)

        assert result["reasoning"] == "Simple arithmetic"
        assert result["answer"] == "4"

    def test_parse_long_think_block(self) -> None:
        """Test parsing with a verbose <think> block."""
        completion = """<think>
Okay, the user is asking "What is 2+2?" which is a basic math problem.
I need to respond with a JSON object.
First, the answer should be calculated. 2+2 equals 4.
Let me make sure I format this correctly.
The user wants reasoning and answer fields.
</think>

{"reasoning": "2+2 is basic arithmetic that equals 4", "answer": "4"}"""

        result = self.adapter.parse(self.signature, completion)

        assert "4" in result["answer"]


class TestExtractJson:
    """Tests for the _extract_json helper method."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.adapter = NemotronJSONAdapter()

    def test_extract_from_code_block(self) -> None:
        """Test extraction from markdown code block."""
        text = '```json\n{"key": "value"}\n```'

        result = self.adapter._extract_json(text)

        assert result == '{"key": "value"}'

    def test_extract_from_code_block_no_language(self) -> None:
        """Test extraction from code block without language specifier."""
        text = '```\n{"key": "value"}\n```'

        result = self.adapter._extract_json(text)

        assert result == '{"key": "value"}'

    def test_extract_raw_json(self) -> None:
        """Test extraction of raw JSON object."""
        text = 'Some text {"key": "value"} more text'

        result = self.adapter._extract_json(text)

        assert result == '{"key": "value"}'

    def test_extract_no_json(self) -> None:
        """Test extraction when no JSON present."""
        text = "Just plain text"

        result = self.adapter._extract_json(text)

        assert result is None

    def test_extract_nested_json(self) -> None:
        """Test extraction of nested JSON object."""
        text = '{"outer": {"inner": "value"}}'

        result = self.adapter._extract_json(text)

        assert result == '{"outer": {"inner": "value"}}'


class ReactSignature(dspy.Signature):
    """Mock signature matching DSPy ReAct internal fields."""

    trajectory: str = dspy.InputField()
    next_thought: str = dspy.OutputField(desc="The next thought")
    next_tool_name: str = dspy.OutputField(desc="The tool to call")
    next_tool_args: dict[str, Any] = dspy.OutputField(desc="Arguments for the tool")


class TestReActFieldsParsing:
    """Tests for parsing ReAct module's internal signature fields."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.adapter = NemotronJSONAdapter()
        self.signature = ReactSignature

    def test_parse_react_standard_json(self) -> None:
        """Test parsing standard JSON with ReAct fields."""
        completion = """{
            "next_thought": "I need to calculate the average",
            "next_tool_name": "python_executor",
            "next_tool_args": {"code": "sum([3,7,2,9,4])/5"}
        }"""
        result = self.adapter.parse(self.signature, completion)
        assert result["next_thought"] == "I need to calculate the average"
        assert result["next_tool_name"] == "python_executor"
        assert result["next_tool_args"] == {"code": "sum([3,7,2,9,4])/5"}

    def test_parse_react_with_final_wrapper(self) -> None:
        """Test parsing ReAct fields with Nemotron's 'final' wrapper."""
        completion = """{
            "final": {
                "next_thought": "Using python to compute",
                "next_tool_name": "python_executor",
                "next_tool_args": {"code": "2+2"}
            }
        }"""
        result = self.adapter.parse(self.signature, completion)
        assert result["next_thought"] == "Using python to compute"
        assert result["next_tool_name"] == "python_executor"
        assert isinstance(result["next_tool_args"], dict)

    def test_parse_react_with_think_block(self) -> None:
        """Test parsing ReAct fields after <think> block."""
        completion = """<think>
I should use the python executor to calculate this.
</think>

{
    "next_thought": "Computing average",
    "next_tool_name": "python_executor",
    "next_tool_args": {"code": "avg = sum([3,7,2,9,4])/5"}
}"""
        result = self.adapter.parse(self.signature, completion)
        assert "Computing" in result["next_thought"]
        assert result["next_tool_name"] == "python_executor"

    def test_parse_react_nested_dict_args(self) -> None:
        """Test parsing when next_tool_args contains nested dict."""
        completion = """{
            "next_thought": "Need to search",
            "next_tool_name": "web_search",
            "next_tool_args": {"query": "test", "options": {"limit": 10}}
        }"""
        result = self.adapter.parse(self.signature, completion)
        assert result["next_tool_args"]["options"]["limit"] == 10

    def test_parse_react_stringified_final_wrapper(self) -> None:
        """Test parsing when 'final' value is a stringified JSON (double-encoded)."""
        # This is the actual format observed in production - the "final" value
        # is a JSON string, not a nested object
        completion = (
            '{"final":"{\\"next_thought\\": \\"I need to calculate\\", '
            '\\"next_tool_name\\": \\"python_execute\\", '
            '\\"next_tool_args\\": {\\"code\\": \\"print(2+2)\\"}}"}'
        )
        result = self.adapter.parse(self.signature, completion)
        assert result["next_thought"] == "I need to calculate"
        assert result["next_tool_name"] == "python_execute"
        assert result["next_tool_args"] == {"code": "print(2+2)"}

    def test_parse_react_stringified_final_with_newlines(self) -> None:
        """Test parsing stringified 'final' with embedded newlines."""
        completion = (
            '{"final":"{\n  \\"next_thought\\": \\"Compute the result\\",\n  '
            '\\"next_tool_name\\": \\"calculate\\",\n  '
            '\\"next_tool_args\\": {\\"expression\\": \\"2+2\\"}\n}"}'
        )
        result = self.adapter.parse(self.signature, completion)
        assert result["next_thought"] == "Compute the result"
        assert result["next_tool_name"] == "calculate"
        assert result["next_tool_args"] == {"expression": "2+2"}


class TestLoggingContext:
    """Tests for logging context in parse failure warnings."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.adapter = NemotronJSONAdapter()
        self.signature = MockSignature

    def test_parse_failure_logs_with_context(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that parse failure logs include model/signature context."""
        caplog.set_level(logging.WARNING)

        # Completion with wrong fields triggers warning
        completion = '{"final": {"wrong_field": "value"}}'
        result = self.adapter.parse(self.signature, completion)

        # Should return empty result since expected fields are missing
        assert result == {}

        # Verify warning was logged with context
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) > 0
        warning_text = warnings[-1].message
        assert "Failed to parse Nemotron response" in warning_text
        assert "signature=" in warning_text  # Context should be present

    def test_standard_parsing_failure_logs_exception_type(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that standard parsing failure logs include exception type."""
        caplog.set_level(logging.WARNING)

        # Malformed JSON that would cause JSONAdapter to throw
        completion = '{"partial": true'  # Missing closing brace
        self.adapter.parse(self.signature, completion)

        # Check if any warning about standard parsing failure was logged
        warning_messages = [
            r.message for r in caplog.records if r.levelno == logging.WARNING
        ]
        # At minimum we should get the final "Failed to parse" warning
        assert any("Failed to parse" in msg for msg in warning_messages)
