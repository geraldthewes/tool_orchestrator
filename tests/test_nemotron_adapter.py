"""
Tests for NemotronJSONAdapter.

Tests the custom adapter that handles Nemotron-Orchestrator's "final" wrapper format.
"""

import dspy

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
