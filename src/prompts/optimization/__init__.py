"""
DSPy Optimization Infrastructure for ToolOrchestra.

Provides metrics, datasets, and optimizers for tuning DSPy modules.
"""

from .metrics import routing_accuracy, orchestration_quality
from .datasets import (
    load_routing_dataset,
    load_orchestration_dataset,
    load_all_training_examples,
    load_all_routing_examples,
    get_train_dev_split,
)
from .optimizer import PromptOptimizer, optimize_all_modules
from .checkpoint import CheckpointManager

__all__ = [
    "routing_accuracy",
    "orchestration_quality",
    "load_routing_dataset",
    "load_orchestration_dataset",
    "load_all_training_examples",
    "load_all_routing_examples",
    "get_train_dev_split",
    "PromptOptimizer",
    "optimize_all_modules",
    "CheckpointManager",
]
