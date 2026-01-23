"""
DSPy Language Model Adapters for ToolOrchestra.

Provides factory functions for creating DSPy LM instances
from existing configuration.
"""

from .lm_factory import (
    get_orchestrator_lm,
    get_teacher_lm,
    get_delegate_lm,
    get_fast_lm,
    configure_dspy_default,
    TracedLM,
)

__all__ = [
    "get_orchestrator_lm",
    "get_teacher_lm",
    "get_delegate_lm",
    "get_fast_lm",
    "configure_dspy_default",
    "TracedLM",
]
