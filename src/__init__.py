"""
ToolOrchestra - LLM Tool Orchestration Framework

This package provides:
- Tool implementations (search, python executor, math solver)
- Nemotron-native orchestration loop with OpenAI function-calling
- Interactive CLI for testing
"""

from .orchestrator import ToolOrchestrator, run_query

__all__ = [
    "ToolOrchestrator",
    "run_query",
]

__version__ = "0.1.0"
