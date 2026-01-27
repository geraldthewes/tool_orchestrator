"""
Mathematical Expression Solver

Safely evaluates mathematical expressions using SymPy's parser.
Supports scientific calculator syntax including:
- Factorial notation: 5!
- Caret exponentiation: 2^16
- Degree notation: sin(30 degrees)
"""

import logging
import re

from sympy import N
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
    factorial_notation,
)

logger = logging.getLogger(__name__)

# Transformations for scientific calculator syntax
TRANSFORMATIONS = (
    standard_transformations
    + (implicit_multiplication_application,)
    + (convert_xor,)  # 2^16 -> 2**16
    + (factorial_notation,)  # 5! -> factorial(5)
)


def preprocess_expression(expression: str) -> str:
    """
    Preprocess expression for SymPy compatibility.

    Handles:
        - Degree notation: sin(30 degrees) -> sin(30 * pi / 180)
        - ceil function: ceil(x) -> ceiling(x) (SymPy naming)
    """
    # Convert degree notation to radians
    # Match number followed by 'degrees' or 'deg'
    degree_pattern = r"(\d+(?:\.\d+)?)\s*(?:degrees?|deg)\b"
    expression = re.sub(degree_pattern, r"(\1 * pi / 180)", expression, flags=re.IGNORECASE)

    # Convert ceil to ceiling (SymPy uses 'ceiling')
    expression = re.sub(r"\bceil\b", "ceiling", expression)

    return expression


def calculate(expression: str) -> dict:
    """
    Safely evaluate a mathematical expression.

    Supports scientific calculator syntax:
    - Basic arithmetic: 2 + 2, 10 * 5
    - Exponentiation: 2^10 or 2**10
    - Factorial: 5! or factorial(5)
    - Trig functions: sin(30 degrees), cos(pi/4)
    - Math functions: sqrt(16), log(100), exp(2)
    - Constants: pi, e, E (Euler's number)

    Args:
        expression: Mathematical expression as a string

    Returns:
        Dictionary with result or error
    """
    # Validate expression is not empty
    if not expression or not expression.strip():
        return {
            "success": False,
            "expression": expression,
            "result": None,
            "error": 'Expression is empty. Please provide a math expression in format: {"expression": "2+2"}',
        }

    try:
        # Preprocess for SymPy compatibility
        processed_expr = preprocess_expression(expression)

        # Parse with scientific calculator transformations
        expr = parse_expr(
            processed_expr,
            transformations=TRANSFORMATIONS,
            evaluate=True,
        )

        # Evaluate numerically
        result = complex(N(expr))

        # Convert to real if no imaginary component
        if result.imag == 0:
            result = result.real

        # Format result - convert to int if it's a whole number
        if isinstance(result, float) and result.is_integer():
            result = int(result)

        return {
            "success": True,
            "expression": expression,
            "result": result,
            "error": None,
        }

    except SyntaxError as e:
        logger.debug("Syntax error parsing expression '%s': %s", expression, e)
        return {
            "success": False,
            "expression": expression,
            "result": None,
            "error": f"Syntax error: {e}",
        }
    except (ValueError, TypeError) as e:
        logger.debug("Value/Type error evaluating '%s': %s", expression, e)
        return {
            "success": False,
            "expression": expression,
            "result": None,
            "error": str(e),
        }
    except Exception as e:
        logger.debug("Calculation error for '%s': %s", expression, e)
        return {
            "success": False,
            "expression": expression,
            "result": None,
            "error": f"Calculation error: {e}",
        }


def format_result_for_llm(calc_result: dict) -> str:
    """
    Format calculation result for LLM consumption.

    Args:
        calc_result: Result from calculate()

    Returns:
        Formatted string
    """
    if not calc_result["success"]:
        return f"Calculation failed: {calc_result['error']}"

    return f"{calc_result['expression']} = {calc_result['result']}"


def _handle_calculate(params: dict) -> dict:
    """Handle calculate tool invocation with input validation."""
    if "raw" in params and "expression" not in params:
        return {
            "success": False,
            "expression": None,
            "result": None,
            "error": 'Invalid input format. Expected JSON: {"expression": "math expression like 2+2 or sqrt(16)"}',
        }
    return calculate(params.get("expression", ""))


# Register tool with the registry
def _register():
    from .registry import ToolRegistry

    ToolRegistry.register(
        name="calculate",
        description="Perform mathematical calculations",
        parameters={"expression": "math expression like 2+2 or sqrt(16)"},
        handler=_handle_calculate,
        formatter=format_result_for_llm,
    )


_register()
