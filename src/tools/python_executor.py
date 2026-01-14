"""
Safe Python Code Executor

Executes Python code in a restricted sandbox environment.
"""

import logging
import ast
import sys
from io import StringIO
from typing import Any

logger = logging.getLogger(__name__)

# Allowed built-in functions for the sandbox
ALLOWED_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bin": bin,
    "bool": bool,
    "chr": chr,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "format": format,
    "frozenset": frozenset,
    "getattr": getattr,
    "hasattr": hasattr,
    "hash": hash,
    "hex": hex,
    "int": int,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "iter": iter,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "oct": oct,
    "ord": ord,
    "pow": pow,
    "print": print,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "zip": zip,
}

# Allowed modules
ALLOWED_MODULES = {
    "math",
    "statistics",
    "random",
    "datetime",
    "json",
    "re",
    "collections",
    "itertools",
    "functools",
}


class RestrictedImport:
    """Restricted import that only allows safe modules."""

    def __init__(self, allowed_modules: set):
        self.allowed_modules = allowed_modules

    def __call__(self, name: str, *args, **kwargs):
        if name not in self.allowed_modules:
            raise ImportError(f"Import of '{name}' is not allowed in sandbox")
        return __builtins__["__import__"](name, *args, **kwargs)


def execute_python(code: str, timeout_seconds: int = 30) -> dict:
    """
    Execute Python code in a sandboxed environment.

    Args:
        code: Python code to execute
        timeout_seconds: Maximum execution time (not enforced in this simple impl)

    Returns:
        Dictionary with output, errors, and return value
    """
    # Validate code doesn't contain dangerous patterns
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {
            "success": False,
            "error": f"Syntax error: {e}",
            "output": "",
            "result": None,
        }

    # Check for dangerous operations
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in ALLOWED_MODULES:
                    return {
                        "success": False,
                        "error": f"Import of '{alias.name}' is not allowed",
                        "output": "",
                        "result": None,
                    }
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] not in ALLOWED_MODULES:
                return {
                    "success": False,
                    "error": f"Import from '{node.module}' is not allowed",
                    "output": "",
                    "result": None,
                }

    # Prepare sandbox environment
    sandbox_globals = {
        "__builtins__": ALLOWED_BUILTINS.copy(),
    }
    sandbox_globals["__builtins__"]["__import__"] = RestrictedImport(ALLOWED_MODULES)

    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    result = None
    error = None

    try:
        # Execute the code
        exec(code, sandbox_globals)

        # Try to get a result if there's an expression at the end
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            result = eval(compile(ast.Expression(tree.body[-1].value), "<string>", "eval"), sandbox_globals)

    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        logger.error(f"Code execution failed: {error}")

    finally:
        sys.stdout = old_stdout

    output = captured_output.getvalue()

    return {
        "success": error is None,
        "error": error,
        "output": output,
        "result": str(result) if result is not None else None,
    }


def format_result_for_llm(execution_result: dict) -> str:
    """
    Format execution result for LLM consumption.

    Args:
        execution_result: Result from execute_python()

    Returns:
        Formatted string
    """
    if not execution_result["success"]:
        return f"Execution failed with error: {execution_result['error']}"

    parts = []
    if execution_result["output"]:
        parts.append(f"Output:\n{execution_result['output']}")
    if execution_result["result"]:
        parts.append(f"Result: {execution_result['result']}")

    return "\n".join(parts) if parts else "Code executed successfully (no output)"
