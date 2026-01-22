"""
DSPy Prompts Package for ToolOrchestra.

Provides declarative prompt programming using DSPy signatures and modules
for query routing and tool orchestration.
"""

from .signatures import QueryRouting, ToolOrchestrationTask
from .modules import QueryRouterModule, ToolOrchestratorModule
from .adapters import get_orchestrator_lm, get_delegate_lm

__all__ = [
    "QueryRouting",
    "ToolOrchestrationTask",
    "QueryRouterModule",
    "ToolOrchestratorModule",
    "get_orchestrator_lm",
    "get_delegate_lm",
]
