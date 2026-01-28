"""
Tests for tool input validation.

These tests verify that tools return helpful error messages when given
empty or malformed input, allowing the LLM to self-correct.
"""

from unittest.mock import patch, Mock

from src.tools import ToolRegistry
from src.tools.search import search
from src.tools.math_solver import calculate
from src.tools.python_executor import execute_python


class TestSearchInputValidation:
    """Tests for web_search input validation."""

    def test_empty_query_string(self):
        """Test that empty query returns helpful error message."""
        result = search(query="")

        assert "error" in result
        assert "empty" in result["error"].lower()
        assert "query" in result["error"].lower()
        assert result["results"] == []

    def test_whitespace_only_query(self):
        """Test that whitespace-only query returns error."""
        result = search(query="   ")

        assert "error" in result
        assert "empty" in result["error"].lower()

    def test_handler_with_missing_query_key(self):
        """Test handler when 'query' key is missing (malformed input)."""
        tool = ToolRegistry.get("web_search")
        result = tool.handler({"raw": "some malformed input"})

        assert "error" in result
        assert (
            "invalid" in result["error"].lower() or "format" in result["error"].lower()
        )
        assert "query" in result["error"].lower()
        assert result["results"] == []

    def test_handler_with_empty_params(self):
        """Test handler with completely empty params dict."""
        tool = ToolRegistry.get("web_search")
        result = tool.handler({})

        assert "error" in result
        assert result["results"] == []

    @patch("src.tools.search.requests.get")
    def test_valid_query_still_works(self, mock_get):
        """Test that valid queries still work after adding validation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"title": "Test", "url": "http://test.com", "content": "Test content"}
            ]
        }
        mock_get.return_value = mock_response

        result = search(query="test query")

        assert "error" not in result or result.get("error") is None
        assert result["total"] >= 0

    @patch("src.tools.search.requests.get")
    def test_handler_with_string_num_results(self, mock_get):
        """Test that num_results as string is converted to int (DSPy compatibility)."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"title": "Test1", "url": "http://test1.com", "content": "Content 1"},
                {"title": "Test2", "url": "http://test2.com", "content": "Content 2"},
                {"title": "Test3", "url": "http://test3.com", "content": "Content 3"},
            ]
        }
        mock_get.return_value = mock_response

        tool = ToolRegistry.get("web_search")
        # DSPy passes params as strings from LLM output
        result = tool.handler({"query": "test", "num_results": "2"})

        assert "error" not in result or result.get("error") is None
        assert result["total"] == 2  # Should respect string "2" converted to int

    def test_handler_with_invalid_num_results_falls_back_to_default(self):
        """Test that invalid num_results falls back to default."""
        tool = ToolRegistry.get("web_search")
        # Invalid string that can't be converted to int should use default
        result = tool.handler({"query": "", "num_results": "invalid"})

        # Will fail validation for empty query, but num_results shouldn't cause crash
        assert "error" in result


class TestMathSolverInputValidation:
    """Tests for calculate tool input validation."""

    def test_empty_expression_string(self):
        """Test that empty expression returns helpful error message."""
        result = calculate(expression="")

        assert result["success"] is False
        assert "empty" in result["error"].lower()
        assert "expression" in result["error"].lower()

    def test_whitespace_only_expression(self):
        """Test that whitespace-only expression returns error."""
        result = calculate(expression="   ")

        assert result["success"] is False
        assert "empty" in result["error"].lower()

    def test_handler_with_missing_expression_key(self):
        """Test handler when 'expression' key is missing (malformed input)."""
        tool = ToolRegistry.get("calculate")
        result = tool.handler({"raw": "some malformed input"})

        assert result["success"] is False
        assert (
            "invalid" in result["error"].lower() or "format" in result["error"].lower()
        )
        assert "expression" in result["error"].lower()

    def test_handler_with_empty_params(self):
        """Test handler with completely empty params dict."""
        tool = ToolRegistry.get("calculate")
        result = tool.handler({})

        assert result["success"] is False
        assert "error" in result

    def test_valid_expression_still_works(self):
        """Test that valid expressions still work after adding validation."""
        result = calculate(expression="2 + 2")

        assert result["success"] is True
        assert result["result"] == 4


class TestPythonExecutorInputValidation:
    """Tests for python_execute tool input validation."""

    def test_empty_code_string(self):
        """Test that empty code returns helpful error message."""
        result = execute_python(code="")

        assert result["success"] is False
        assert "empty" in result["error"].lower()
        assert "code" in result["error"].lower()

    def test_whitespace_only_code(self):
        """Test that whitespace-only code returns error."""
        result = execute_python(code="   ")

        assert result["success"] is False
        assert "empty" in result["error"].lower()

    def test_handler_with_missing_code_key(self):
        """Test handler when 'code' key is missing (malformed input)."""
        tool = ToolRegistry.get("python_execute")
        result = tool.handler({"raw": "some malformed input"})

        assert result["success"] is False
        assert (
            "invalid" in result["error"].lower() or "format" in result["error"].lower()
        )
        assert "code" in result["error"].lower()

    def test_handler_with_empty_params(self):
        """Test handler with completely empty params dict."""
        tool = ToolRegistry.get("python_execute")
        result = tool.handler({})

        assert result["success"] is False
        assert "error" in result

    @patch("src.tools.python_executor.requests.post")
    def test_valid_code_still_works(self, mock_post):
        """Test that valid code still works after adding validation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "completed",
            "stdout": "hello\n",
            "stderr": "",
            "exit_code": 0,
        }
        mock_post.return_value = mock_response

        result = execute_python(code="print('hello')")

        assert result["success"] is True
        assert "hello" in result["output"]

    @patch("src.tools.python_executor.requests.post")
    def test_handler_with_string_timeout(self, mock_post):
        """Test that timeout as string is converted to int (DSPy compatibility)."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "completed",
            "stdout": "done\n",
            "stderr": "",
            "exit_code": 0,
        }
        mock_post.return_value = mock_response

        tool = ToolRegistry.get("python_execute")
        # DSPy passes params as strings from LLM output
        result = tool.handler({"code": "print('done')", "timeout": "60"})

        assert result["success"] is True
        # Verify the timeout was converted (60 + 5 = 65 seconds for network buffer)
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["timeout"] == 65

    def test_handler_with_invalid_timeout_falls_back_to_default(self):
        """Test that invalid timeout falls back to default."""
        tool = ToolRegistry.get("python_execute")
        # Invalid string that can't be converted to int should use default (30)
        result = tool.handler({"code": "", "timeout": "invalid"})

        # Will fail validation for empty code, but timeout shouldn't cause crash
        assert result["success"] is False
        assert "empty" in result["error"].lower()


class TestDelegateInputValidation:
    """Tests for delegate LLM tool input validation."""

    def test_delegate_handler_rejects_malformed_input(self):
        """Test that delegate handlers reject malformed input with helpful error."""
        from src.orchestrator import ToolOrchestrator

        orchestrator = ToolOrchestrator()

        # Get a delegate handler (e.g., ask_reasoner)
        delegate_handlers = orchestrator.delegate_handlers
        if delegate_handlers:
            handler_name = next(iter(delegate_handlers))
            handler = delegate_handlers[handler_name]

            result = handler({"raw": "malformed input"})

            assert result["success"] is False
            assert "prompt" in result["error"].lower()

    def test_delegate_handler_rejects_empty_prompt(self):
        """Test that delegate handlers reject empty prompts."""
        from src.orchestrator import ToolOrchestrator

        orchestrator = ToolOrchestrator()

        delegate_handlers = orchestrator.delegate_handlers
        if delegate_handlers:
            handler_name = next(iter(delegate_handlers))
            handler = delegate_handlers[handler_name]

            result = handler({"prompt": ""})

            assert result["success"] is False
            assert "empty" in result["error"].lower()

    def test_delegate_handler_rejects_whitespace_prompt(self):
        """Test that delegate handlers reject whitespace-only prompts."""
        from src.orchestrator import ToolOrchestrator

        orchestrator = ToolOrchestrator()

        delegate_handlers = orchestrator.delegate_handlers
        if delegate_handlers:
            handler_name = next(iter(delegate_handlers))
            handler = delegate_handlers[handler_name]

            result = handler({"prompt": "   "})

            assert result["success"] is False
            assert "empty" in result["error"].lower()


