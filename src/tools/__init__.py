"""
ToolOrchestra Tools Package

Available tools:
- search: Web search via SearXNG
- python_executor: Safe Python code execution
- math_solver: Mathematical expression evaluation
- llm_delegate: Delegation to specialized LLMs
"""

from .search import search, format_results_for_llm as format_search_results
from .python_executor import execute_python, format_result_for_llm as format_python_result
from .math_solver import calculate, format_result_for_llm as format_math_result
from .llm_delegate import (
    call_gpt_oss,
    call_qwen_coder,
    call_nemotron_nano,
    format_result_for_llm as format_delegate_result,
)

__all__ = [
    "search",
    "format_search_results",
    "execute_python",
    "format_python_result",
    "calculate",
    "format_math_result",
    "call_gpt_oss",
    "call_qwen_coder",
    "call_nemotron_nano",
    "format_delegate_result",
]
