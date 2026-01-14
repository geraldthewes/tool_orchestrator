"""
Mathematical Expression Solver

Safely evaluates mathematical expressions.
"""

import logging
import ast
import math
import operator as op

logger = logging.getLogger(__name__)

# Supported operators
OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}

# Supported math functions
MATH_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "pow": pow,
    "floor": math.floor,
    "ceil": math.ceil,
    "factorial": math.factorial,
    "gcd": math.gcd,
}

# Supported constants
CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
}


def _eval_node(node: ast.AST) -> float:
    """Recursively evaluate an AST node."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")

    elif isinstance(node, ast.Name):
        name = node.id.lower()
        if name in CONSTANTS:
            return CONSTANTS[name]
        raise ValueError(f"Unknown variable: {node.id}")

    elif isinstance(node, ast.BinOp):
        if type(node.op) not in OPERATORS:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return OPERATORS[type(node.op)](left, right)

    elif isinstance(node, ast.UnaryOp):
        if type(node.op) not in OPERATORS:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        operand = _eval_node(node.operand)
        return OPERATORS[type(node.op)](operand)

    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are supported")

        func_name = node.func.id.lower()
        if func_name not in MATH_FUNCTIONS:
            raise ValueError(f"Unknown function: {node.func.id}")

        args = [_eval_node(arg) for arg in node.args]
        return MATH_FUNCTIONS[func_name](*args)

    elif isinstance(node, ast.Expression):
        return _eval_node(node.body)

    else:
        raise ValueError(f"Unsupported node type: {type(node).__name__}")


def calculate(expression: str) -> dict:
    """
    Safely evaluate a mathematical expression.

    Args:
        expression: Mathematical expression as a string

    Returns:
        Dictionary with result or error
    """
    try:
        # Parse the expression
        tree = ast.parse(expression, mode="eval")

        # Evaluate it
        result = _eval_node(tree)

        # Format result
        if isinstance(result, float) and result.is_integer():
            result = int(result)

        return {
            "success": True,
            "expression": expression,
            "result": result,
            "error": None,
        }

    except SyntaxError as e:
        return {
            "success": False,
            "expression": expression,
            "result": None,
            "error": f"Syntax error: {e}",
        }
    except ValueError as e:
        return {
            "success": False,
            "expression": expression,
            "result": None,
            "error": str(e),
        }
    except Exception as e:
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
