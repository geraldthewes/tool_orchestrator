"""
DSPy Prompts Package for ToolOrchestra.

Provides declarative prompt programming using DSPy signatures and modules
for query routing.
"""

from .signatures import QueryRouting
from .modules import QueryRouterModule
from .adapters import get_orchestrator_lm, get_delegate_lm

__all__ = [
    "QueryRouting",
    "QueryRouterModule",
    "get_orchestrator_lm",
    "get_delegate_lm",
]
