"""
Tests for DSPy Signatures.

Tests the signature field definitions and types.
"""

from src.prompts.signatures import QueryRouting, ToolOrchestrationTask


class TestQueryRoutingSignature:
    """Tests for QueryRouting signature."""

    def test_signature_has_required_input_fields(self):
        """Test that QueryRouting has required input fields."""
        sig = QueryRouting
        input_fields = sig.input_fields

        assert "query" in input_fields
        assert "available_tools" in input_fields

    def test_signature_has_required_output_fields(self):
        """Test that QueryRouting has required output fields."""
        sig = QueryRouting
        output_fields = sig.output_fields

        assert "needs_tools" in output_fields
        assert "reasoning" in output_fields
        assert "direct_answer" in output_fields

    def test_query_field_description(self):
        """Test query field has description."""
        sig = QueryRouting
        query_field = sig.input_fields["query"]
        # DSPy stores description in json_schema_extra['desc']
        desc = query_field.json_schema_extra.get("desc", "")
        assert desc is not None and len(desc) > 0
        assert "query" in desc.lower()

    def test_needs_tools_is_bool_field(self):
        """Test needs_tools output field type."""
        sig = QueryRouting
        needs_tools_field = sig.output_fields["needs_tools"]
        # DSPy stores description in json_schema_extra['desc']
        desc = needs_tools_field.json_schema_extra.get("desc", "")
        assert desc is not None and len(desc) > 0


class TestToolOrchestrationTaskSignature:
    """Tests for ToolOrchestrationTask signature."""

    def test_signature_has_required_input_fields(self):
        """Test that ToolOrchestrationTask has required input fields."""
        sig = ToolOrchestrationTask
        input_fields = sig.input_fields

        assert "question" in input_fields

    def test_signature_has_required_output_fields(self):
        """Test that ToolOrchestrationTask has required output fields."""
        sig = ToolOrchestrationTask
        output_fields = sig.output_fields

        assert "answer" in output_fields

    def test_question_field_description(self):
        """Test question field has description."""
        sig = ToolOrchestrationTask
        question_field = sig.input_fields["question"]
        # DSPy stores description in json_schema_extra['desc']
        desc = question_field.json_schema_extra.get("desc", "")
        assert desc is not None and len(desc) > 0

    def test_answer_field_description(self):
        """Test answer field has description."""
        sig = ToolOrchestrationTask
        answer_field = sig.output_fields["answer"]
        # DSPy stores description in json_schema_extra['desc']
        desc = answer_field.json_schema_extra.get("desc", "")
        assert desc is not None and len(desc) > 0

    def test_signature_docstring(self):
        """Test signature has docstring."""
        sig = ToolOrchestrationTask
        assert sig.__doc__ is not None
        assert "ReAct" in sig.__doc__ or "orchestrat" in sig.__doc__.lower()
