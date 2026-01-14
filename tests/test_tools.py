"""
Tests for ToolOrchestra tools.

These tests cover the local tools (math solver, python executor)
without requiring external services.
"""

import pytest
from src.tools.math_solver import calculate, format_result_for_llm as format_math_result
from src.tools.python_executor import execute_python, format_result_for_llm as format_python_result


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
    """Tests for the Python executor tool."""

    def test_simple_print(self):
        """Test simple print statement."""
        result = execute_python('print("Hello, World!")')
        assert result["success"] is True
        assert "Hello, World!" in result["output"]

    def test_arithmetic(self):
        """Test arithmetic in Python."""
        result = execute_python("2 + 2")
        assert result["success"] is True
        assert result["result"] == "4"

    def test_variable_assignment(self):
        """Test variable assignment and usage."""
        code = """
x = 10
y = 20
x + y
"""
        result = execute_python(code)
        assert result["success"] is True
        assert result["result"] == "30"

    def test_list_operations(self):
        """Test list operations."""
        code = """
numbers = [1, 2, 3, 4, 5]
sum(numbers)
"""
        result = execute_python(code)
        assert result["success"] is True
        assert result["result"] == "15"

    def test_loop(self):
        """Test loop execution."""
        code = """
total = 0
for i in range(5):
    total += i
print(total)
"""
        result = execute_python(code)
        assert result["success"] is True
        assert "10" in result["output"]

    def test_function_definition(self):
        """Test function definition and calling."""
        code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

factorial(5)
"""
        result = execute_python(code)
        assert result["success"] is True
        assert result["result"] == "120"

    def test_allowed_import_math(self):
        """Test allowed import (math module)."""
        code = """
import math
math.sqrt(16)
"""
        result = execute_python(code)
        assert result["success"] is True
        assert result["result"] == "4.0"

    def test_allowed_import_json(self):
        """Test allowed import (json module)."""
        code = """
import json
data = {"key": "value"}
json.dumps(data)
"""
        result = execute_python(code)
        assert result["success"] is True
        assert '"key"' in result["result"]

    def test_disallowed_import(self):
        """Test that disallowed imports are blocked."""
        result = execute_python("import os")
        assert result["success"] is False
        assert "not allowed" in result["error"]

    def test_disallowed_import_from(self):
        """Test that disallowed from imports are blocked."""
        result = execute_python("from subprocess import run")
        assert result["success"] is False
        assert "not allowed" in result["error"]

    def test_syntax_error(self):
        """Test handling of syntax errors."""
        result = execute_python("def broken(")
        assert result["success"] is False
        assert "Syntax error" in result["error"]

    def test_runtime_error(self):
        """Test handling of runtime errors."""
        result = execute_python("1 / 0")
        assert result["success"] is False
        assert "ZeroDivision" in result["error"]

    def test_name_error(self):
        """Test handling of undefined variables."""
        result = execute_python("undefined_variable")
        assert result["success"] is False
        assert "NameError" in result["error"]

    def test_format_output(self):
        """Test formatting output with print."""
        result = execute_python('print("test output")')
        formatted = format_python_result(result)
        assert "test output" in formatted

    def test_format_result(self):
        """Test formatting result expression."""
        result = execute_python("42")
        formatted = format_python_result(result)
        assert "42" in formatted

    def test_format_error(self):
        """Test formatting error."""
        result = execute_python("undefined")
        formatted = format_python_result(result)
        assert "failed" in formatted.lower() or "error" in formatted.lower()

    def test_list_comprehension(self):
        """Test list comprehension."""
        code = "[x**2 for x in range(5)]"
        result = execute_python(code)
        assert result["success"] is True
        assert "[0, 1, 4, 9, 16]" in result["result"]

    def test_dict_operations(self):
        """Test dictionary operations."""
        code = """
d = {"a": 1, "b": 2}
d["c"] = 3
len(d)
"""
        result = execute_python(code)
        assert result["success"] is True
        assert result["result"] == "3"


class TestToolFormatters:
    """Tests for tool result formatters."""

    def test_math_format_with_integers(self):
        """Test math formatter converts float to int when appropriate."""
        result = calculate("10.0 + 5.0")
        # Should be formatted as integer
        assert result["result"] == 15

    def test_python_no_output(self):
        """Test Python formatter with no output."""
        result = execute_python("x = 5")  # Assignment, no print or expression
        formatted = format_python_result(result)
        assert "success" in formatted.lower() or "no output" in formatted.lower()
