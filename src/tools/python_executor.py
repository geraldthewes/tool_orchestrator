"""
Remote Python Code Executor

Executes Python code via a remote python-executor service for secure,
sandboxed code execution outside the container.
"""

import logging

import requests

from ..config import config

logger = logging.getLogger(__name__)


def execute_python(code: str, timeout_seconds: int = 30) -> dict:
    """
    Execute Python code via remote python-executor service.

    Args:
        code: Python code to execute
        timeout_seconds: Maximum execution time

    Returns:
        Dictionary with output, errors, and return value
    """
    # Validate code is not empty
    if not code or not code.strip():
        return {
            "success": False,
            "error": 'Code is empty. Please provide Python code in format: {"code": "print(\'hello\')"}',
            "output": "",
            "result": None,
        }

    base_url = config.tools.python_executor_url.rstrip("/")
    endpoint = f"{base_url}/api/v1/eval"

    payload = {"code": code}

    # Log code execution (truncate to avoid log bloat)
    code_preview = code[:200] + "..." if len(code) > 200 else code
    logger.info(f"Executing Python code: {code_preview}")

    try:
        response = requests.post(
            endpoint,
            json=payload,
            timeout=timeout_seconds + 5,  # Add buffer for network latency
        )
        response.raise_for_status()
        data = response.json()

        return _adapt_response(data)

    except requests.exceptions.Timeout:
        logger.error(f"Python executor request timed out after {timeout_seconds}s")
        return {
            "success": False,
            "error": f"Execution timed out after {timeout_seconds} seconds",
            "output": "",
            "result": None,
        }
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Failed to connect to python-executor service: {e}")
        return {
            "success": False,
            "error": f"Failed to connect to python-executor service: {e}",
            "output": "",
            "result": None,
        }
    except requests.exceptions.HTTPError as e:
        logger.error(f"Python executor HTTP error: {e}")
        return {
            "success": False,
            "error": f"Python executor service error: {e}",
            "output": "",
            "result": None,
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"Python executor request failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "output": "",
            "result": None,
        }


def _adapt_response(remote_response: dict) -> dict:
    """
    Adapt the remote service response to the expected format.

    Remote service returns: status, stdout, stderr, exit_code, error
    We need to return: success, error, output, result
    """
    status = remote_response.get("status", "failed")
    success = status == "completed"
    stdout = remote_response.get("stdout", "")
    stderr = remote_response.get("stderr", "")
    exit_code = remote_response.get("exit_code", -1)
    error_msg = remote_response.get("error", "")

    error = None
    if not success:
        if error_msg:
            error = error_msg
        elif stderr:
            error = stderr.strip()
        elif exit_code != 0:
            error = f"Process exited with code {exit_code}"
        else:
            error = f"Execution {status}"

    output = stdout
    if success and stderr:
        # Append stderr as informational if execution succeeded
        output = f"{stdout}\n[stderr: {stderr}]" if stdout else f"[stderr: {stderr}]"

    return {
        "success": success,
        "error": error,
        "output": output,
        "result": None,  # Remote executor doesn't extract last expression
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


def _handle_python_execute(params: dict) -> dict:
    """Handle python_execute tool invocation with input validation."""
    if "raw" in params and "code" not in params:
        return {
            "success": False,
            "error": 'Invalid input format. Expected JSON: {"code": "python code to execute"}',
            "output": "",
            "result": None,
        }
    return execute_python(
        code=params.get("code", ""),
        timeout_seconds=params.get("timeout", 30),
    )


# Register tool with the registry
def _register():
    from .registry import ToolRegistry

    ToolRegistry.register(
        name="python_execute",
        description="Execute Python code safely in a sandbox",
        parameters={
            "code": "python code to execute",
            "timeout": "execution timeout in seconds (default 30)",
        },
        handler=_handle_python_execute,
        formatter=format_result_for_llm,
    )


_register()
