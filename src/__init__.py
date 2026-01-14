"""
ToolOrchestra - NVIDIA Nemotron-Orchestrator-8B Testing Framework

This package provides:
- LLM client for orchestrator and delegate models
- Tool implementations (search, python executor, math solver)
- ReAct-style orchestration loop
- Interactive CLI for testing
"""

from .orchestrator import ToolOrchestrator, run_query
from .llm_call import LLMClient

__all__ = [
    "ToolOrchestrator",
    "LLMClient",
    "run_query",
]

__version__ = "0.1.0"
