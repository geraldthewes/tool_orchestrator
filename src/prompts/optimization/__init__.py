"""
DSPy Optimization Infrastructure for ToolOrchestra.

Provides metrics, datasets, and optimizers for tuning DSPy modules.
"""

from .metrics import routing_accuracy, orchestration_quality
from .datasets import load_routing_dataset, load_orchestration_dataset
from .optimizer import PromptOptimizer

__all__ = [
    "routing_accuracy",
    "orchestration_quality",
    "load_routing_dataset",
    "load_orchestration_dataset",
    "PromptOptimizer",
]
