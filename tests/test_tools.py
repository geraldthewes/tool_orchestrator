"""
Tests for ToolOrchestra tools.

These tests cover the local tools (math solver) and the remote python executor
(mocked HTTP calls) without requiring external services.
"""

from unittest.mock import patch, Mock

import requests

from src.tools.math_solver import calculate, format_result_for_llm as format_math_result
from src.tools.python_executor import (
    execute_python,
    format_result_for_llm as format_python_result,
)


class TestMathSolver:
    """Tests for the math solver tool."""

    def test_basic_arithmetic(self):
        """Test basic arithmetic operations."""
        result = calculate("2 + 2")
        assert result["success"] is True
        assert result["result"] == 4

    def test_multiplication(self):
        """Test multiplication."""
        result = calculate("7 * 8")
        assert result["success"] is True
        assert result["result"] == 56

    def test_division(self):
        """Test division."""
        result = calculate("100 / 4")
        assert result["success"] is True
        assert result["result"] == 25

    def test_power(self):
        """Test exponentiation."""
        result = calculate("2 ** 10")
        assert result["success"] is True
        assert result["result"] == 1024

    def test_complex_expression(self):
        """Test complex mathematical expression."""
        result = calculate("(10 + 5) * 2 - 3")
        assert result["success"] is True
        assert result["result"] == 27

    def test_sqrt_function(self):
        """Test square root function."""
        result = calculate("sqrt(144)")
        assert result["success"] is True
        assert result["result"] == 12

    def test_trigonometric(self):
        """Test trigonometric functions."""
        result = calculate("sin(0)")
        assert result["success"] is True
        assert result["result"] == 0

    def test_pi_constant(self):
        """Test pi constant."""
        result = calculate("pi")
        assert result["success"] is True
        assert abs(result["result"] - 3.14159265358979) < 0.0001

    def test_floor_ceil(self):
        """Test floor and ceiling functions."""
        floor_result = calculate("floor(3.7)")
        ceil_result = calculate("ceil(3.2)")
        assert floor_result["success"] is True
        assert floor_result["result"] == 3
        assert ceil_result["success"] is True
        assert ceil_result["result"] == 4

    def test_factorial(self):
        """Test factorial function."""
        result = calculate("factorial(5)")
        assert result["success"] is True
        assert result["result"] == 120

    def test_syntax_error(self):
        """Test handling of syntax errors."""
        result = calculate("2 +* 2")
        assert result["success"] is False
        assert "Syntax error" in result["error"]

    def test_unknown_function(self):
        """Test handling of unknown functions."""
        result = calculate("unknown_func(5)")
        assert result["success"] is False
        assert "Unknown function" in result["error"]

    def test_format_success(self):
        """Test formatting successful result."""
        result = calculate("2 + 2")
        formatted = format_math_result(result)
        assert "2 + 2 = 4" in formatted

    def test_format_error(self):
        """Test formatting error result."""
        result = calculate("invalid")
        formatted = format_math_result(result)
        assert "failed" in formatted.lower()


class TestPythonExecutor:
    """Tests for the Python executor tool (remote service)."""

    @patch("src.tools.python_executor.requests.post")
    def test_simple_print(self, mock_post):
        """Test simple print statement."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "completed",
            "stdout": "Hello, World!\n",
            "stderr": "",
            "exit_code": 0,
        }
        mock_post.return_value = mock_response

        result = execute_python('print("Hello, World!")')

        assert result["success"] is True
        assert "Hello, World!" in result["output"]
        mock_post.assert_called_once()

    @patch("src.tools.python_executor.requests.post")
    def test_arithmetic(self, mock_post):
        """Test arithmetic in Python."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "completed",
            "stdout": "4\n",
            "stderr": "",
            "exit_code": 0,
        }
        mock_post.return_value = mock_response

        result = execute_python("print(2 + 2)")

        assert result["success"] is True
        assert "4" in result["output"]

    @patch("src.tools.python_executor.requests.post")
    def test_loop(self, mock_post):
        """Test loop execution."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "completed",
            "stdout": "10\n",
            "stderr": "",
            "exit_code": 0,
        }
        mock_post.return_value = mock_response

        code = """
total = 0
for i in range(5):
    total += i
print(total)
"""
        result = execute_python(code)

        assert result["success"] is True
        assert "10" in result["output"]

    @patch("src.tools.python_executor.requests.post")
    def test_runtime_error(self, mock_post):
        """Test handling of runtime errors."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "failed",
            "stdout": "",
            "stderr": 'Traceback (most recent call last):\n  File "main.py", line 1\nZeroDivisionError: division by zero',
            "exit_code": 1,
        }
        mock_post.return_value = mock_response

        result = execute_python("1 / 0")

        assert result["success"] is False
        assert "ZeroDivision" in result["error"]

    @patch("src.tools.python_executor.requests.post")
    def test_syntax_error(self, mock_post):
        """Test handling of syntax errors."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "failed",
            "stdout": "",
            "stderr": "SyntaxError: unexpected EOF while parsing",
            "exit_code": 1,
        }
        mock_post.return_value = mock_response

        result = execute_python("def broken(")

        assert result["success"] is False
        assert "Syntax" in result["error"]

    @patch("src.tools.python_executor.requests.post")
    def test_connection_error(self, mock_post):
        """Test handling of connection errors."""
        mock_post.side_effect = requests.exceptions.ConnectionError(
            "Connection refused"
        )

        result = execute_python("print('test')")

        assert result["success"] is False
        assert "connect" in result["error"].lower()

    @patch("src.tools.python_executor.requests.post")
    def test_timeout_error(self, mock_post):
        """Test handling of timeout errors."""
        mock_post.side_effect = requests.exceptions.Timeout()

        result = execute_python("print('test')", timeout_seconds=5)

        assert result["success"] is False
        assert "timed out" in result["error"].lower()

    @patch("src.tools.python_executor.requests.post")
    def test_http_error(self, mock_post):
        """Test handling of HTTP errors."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "500 Server Error"
        )
        mock_post.return_value = mock_response

        result = execute_python("print('test')")

        assert result["success"] is False
        assert "error" in result["error"].lower()

    @patch("src.tools.python_executor.requests.post")
    def test_request_payload_format(self, mock_post):
        """Test that correct JSON payload is sent to remote service."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "completed",
            "stdout": "",
            "stderr": "",
            "exit_code": 0,
        }
        mock_post.return_value = mock_response

        code = "print('hello')"
        execute_python(code, timeout_seconds=60)

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args

        # Verify JSON format (uses 'json' parameter, not 'files')
        assert "json" in call_kwargs.kwargs
        assert "files" not in call_kwargs.kwargs

        json_payload = call_kwargs.kwargs["json"]
        assert json_payload == {"code": code}

        # Verify endpoint URL uses /api/v1/eval
        call_args = mock_post.call_args
        endpoint_url = (
            call_args.args[0] if call_args.args else call_kwargs.kwargs.get("url", "")
        )
        assert "/api/v1/eval" in endpoint_url

    @patch("src.tools.python_executor.requests.post")
    def test_format_output(self, mock_post):
        """Test formatting output with print."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "completed",
            "stdout": "test output\n",
            "stderr": "",
            "exit_code": 0,
        }
        mock_post.return_value = mock_response

        result = execute_python('print("test output")')
        formatted = format_python_result(result)

        assert "test output" in formatted

    @patch("src.tools.python_executor.requests.post")
    def test_format_error(self, mock_post):
        """Test formatting error."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "failed",
            "stdout": "",
            "stderr": "NameError: name 'undefined' is not defined",
            "exit_code": 1,
        }
        mock_post.return_value = mock_response

        result = execute_python("undefined")
        formatted = format_python_result(result)

        assert "failed" in formatted.lower() or "error" in formatted.lower()

    @patch("src.tools.python_executor.requests.post")
    def test_stderr_appended_on_success(self, mock_post):
        """Test that stderr is appended to output on successful execution."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "completed",
            "stdout": "result\n",
            "stderr": "warning message",
            "exit_code": 0,
        }
        mock_post.return_value = mock_response

        result = execute_python("print('result')")

        assert result["success"] is True
        assert "result" in result["output"]
        assert "warning message" in result["output"]


class TestToolFormatters:
    """Tests for tool result formatters."""

    def test_math_format_with_integers(self):
        """Test math formatter converts float to int when appropriate."""
        result = calculate("10.0 + 5.0")
        # Should be formatted as integer
        assert result["result"] == 15

    @patch("src.tools.python_executor.requests.post")
    def test_python_no_output(self, mock_post):
        """Test Python formatter with no output."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "completed",
            "stdout": "",
            "stderr": "",
            "exit_code": 0,
        }
        mock_post.return_value = mock_response

        result = execute_python("x = 5")  # Assignment, no print
        formatted = format_python_result(result)

        assert "success" in formatted.lower() or "no output" in formatted.lower()
