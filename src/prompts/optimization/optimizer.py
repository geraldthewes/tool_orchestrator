"""
DSPy Prompt Optimizer for ToolOrchestra.

Provides optimization functionality for tuning DSPy modules
using various optimization strategies.
"""

import logging
from pathlib import Path
from typing import Optional, Callable

import dspy
from dspy.teleprompt import BootstrapFewShot, MIPROv2

from .metrics import routing_accuracy, orchestration_quality
from .datasets import load_routing_dataset, load_orchestration_dataset
from ..adapters import get_orchestrator_lm

logger = logging.getLogger(__name__)


class PromptOptimizer:
    """
    Optimizer for DSPy modules using various strategies.

    Supports BootstrapFewShot and MIPROv2 optimization strategies.
    """

    def __init__(
        self,
        strategy: str = "bootstrap",
        metric: Optional[Callable] = None,
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 16,
    ):
        """
        Initialize the optimizer.

        Args:
            strategy: Optimization strategy ("bootstrap" or "mipro")
            metric: Metric function for evaluation
            max_bootstrapped_demos: Max demos for BootstrapFewShot
            max_labeled_demos: Max labeled demos for MIPROv2
        """
        self.strategy = strategy
        self.metric = metric
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos

    def optimize_router(
        self,
        module: dspy.Module,
        trainset: Optional[list[dspy.Example]] = None,
        teacher_lm: Optional[dspy.LM] = None,
    ) -> dspy.Module:
        """
        Optimize a router module.

        Args:
            module: Router module to optimize
            trainset: Training examples (loads default if None)
            teacher_lm: Teacher LM for optimization (uses orchestrator if None)

        Returns:
            Optimized module
        """
        if trainset is None:
            trainset = load_routing_dataset()

        if not trainset:
            logger.warning("No training data for router optimization")
            return module

        metric = self.metric or routing_accuracy
        teacher = teacher_lm or get_orchestrator_lm()

        return self._optimize(module, trainset, metric, teacher)

    def optimize_orchestrator(
        self,
        module: dspy.Module,
        trainset: Optional[list[dspy.Example]] = None,
        teacher_lm: Optional[dspy.LM] = None,
    ) -> dspy.Module:
        """
        Optimize an orchestrator module.

        Args:
            module: Orchestrator module to optimize
            trainset: Training examples (loads default if None)
            teacher_lm: Teacher LM for optimization

        Returns:
            Optimized module
        """
        if trainset is None:
            trainset = load_orchestration_dataset()

        if not trainset:
            logger.warning("No training data for orchestrator optimization")
            return module

        metric = self.metric or orchestration_quality
        teacher = teacher_lm or get_orchestrator_lm()

        return self._optimize(module, trainset, metric, teacher)

    def _optimize(
        self,
        module: dspy.Module,
        trainset: list[dspy.Example],
        metric: Callable,
        teacher_lm: dspy.LM,
    ) -> dspy.Module:
        """
        Run optimization with selected strategy.

        Args:
            module: Module to optimize
            trainset: Training examples
            metric: Metric function
            teacher_lm: Teacher LM

        Returns:
            Optimized module
        """
        logger.info(
            f"Starting optimization with strategy={self.strategy}, "
            f"trainset_size={len(trainset)}"
        )

        with dspy.context(lm=teacher_lm):
            if self.strategy == "bootstrap":
                optimizer = BootstrapFewShot(
                    metric=metric,
                    max_bootstrapped_demos=self.max_bootstrapped_demos,
                )
                optimized = optimizer.compile(module, trainset=trainset)

            elif self.strategy == "mipro":
                optimizer = MIPROv2(
                    metric=metric,
                    num_candidates=5,
                    init_temperature=0.7,
                )
                optimized = optimizer.compile(
                    module,
                    trainset=trainset,
                    max_labeled_demos=self.max_labeled_demos,
                )

            else:
                raise ValueError(f"Unknown optimization strategy: {self.strategy}")

        logger.info("Optimization complete")
        return optimized

    @staticmethod
    def save(module: dspy.Module, path: str) -> None:
        """
        Save an optimized module to disk.

        Args:
            module: Optimized module to save
            path: Path to save to (JSON file)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        module.save(str(path))
        logger.info(f"Saved optimized module to {path}")

    @staticmethod
    def load(module: dspy.Module, path: str) -> dspy.Module:
        """
        Load optimized weights into a module.

        Args:
            module: Module to load weights into
            path: Path to load from

        Returns:
            Module with loaded weights
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Optimized module not found: {path}")

        module.load(str(path))
        logger.info(f"Loaded optimized module from {path}")
        return module


def optimize_all_modules(
    output_dir: str = "data/optimized_prompts",
    strategy: str = "bootstrap",
) -> dict[str, Path]:
    """
    Optimize all DSPy modules and save to disk.

    Args:
        output_dir: Directory to save optimized modules
        strategy: Optimization strategy

    Returns:
        Dict mapping module names to saved paths
    """
    from ..modules import QueryRouterModule, ToolOrchestratorModule

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    optimizer = PromptOptimizer(strategy=strategy)
    saved_paths = {}

    # Optimize router
    logger.info("Optimizing router module...")
    router = QueryRouterModule()
    optimized_router = optimizer.optimize_router(router)
    router_path = output_path / "router.json"
    PromptOptimizer.save(optimized_router, str(router_path))
    saved_paths["router"] = router_path

    # Optimize orchestrator
    logger.info("Optimizing orchestrator module...")
    orchestrator = ToolOrchestratorModule()
    optimized_orchestrator = optimizer.optimize_orchestrator(orchestrator)
    orchestrator_path = output_path / "orchestrator.json"
    PromptOptimizer.save(optimized_orchestrator, str(orchestrator_path))
    saved_paths["orchestrator"] = orchestrator_path

    logger.info(f"All modules optimized and saved to {output_path}")
    return saved_paths
