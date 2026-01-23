"""
DSPy Prompt Optimizer for ToolOrchestra.

Provides optimization functionality for tuning DSPy modules
using various optimization strategies.
"""

import logging
from pathlib import Path
from typing import Optional, Callable, Literal

import dspy
from dspy.teleprompt import BootstrapFewShot, MIPROv2, GEPA

from .checkpoint import CheckpointManager
from .metrics import routing_accuracy, orchestration_quality
from .datasets import (
    load_routing_dataset,
    load_orchestration_dataset,
    load_all_training_examples,
    get_train_dev_split,
)
from ..adapters import get_teacher_lm

logger = logging.getLogger(__name__)


def _wrap_metric_for_gepa(metric: Callable) -> Callable:
    """
    Wrap a simple metric function to be compatible with GEPA's GEPAFeedbackMetric protocol.

    GEPA expects: metric(gold, pred, trace, pred_name, pred_trace) -> float | ScoreWithFeedback
    Simple metrics expect: metric(example, prediction) -> float
    """

    def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        # Call the simple metric with just gold and pred
        return metric(gold, pred)

    return gepa_metric


class PromptOptimizer:
    """
    Optimizer for DSPy modules using various strategies.

    Supports BootstrapFewShot, MIPROv2, and GEPA optimization strategies.
    GEPA (Genetic-Pareto optimization) is recommended for ReAct-style programs.
    """

    def __init__(
        self,
        strategy: str = "gepa",
        metric: Optional[Callable] = None,
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 16,
        reflection_lm: Optional[dspy.LM] = None,
        gepa_auto: Literal["light", "medium", "heavy"] = "light",
        checkpoint_dir: Optional[Path] = None,
        resume_from: Optional[Path] = None,
        teacher_base_url: Optional[str] = None,
        teacher_model: Optional[str] = None,
    ):
        """
        Initialize the optimizer.

        Args:
            strategy: Optimization strategy ("bootstrap", "mipro", or "gepa")
            metric: Metric function for evaluation
            max_bootstrapped_demos: Max demos for BootstrapFewShot
            max_labeled_demos: Max labeled demos for MIPROv2
            reflection_lm: Reflection LM for GEPA (uses teacher if None)
            gepa_auto: GEPA preset ("light", "medium", "heavy")
            checkpoint_dir: Directory to save checkpoints (enables checkpointing)
            resume_from: Path to checkpoint to resume from
            teacher_base_url: Base URL for teacher LLM (overrides TEACHER_BASE_URL env)
            teacher_model: Model name for teacher LLM (overrides TEACHER_MODEL env)
        """
        self.strategy = strategy
        self.metric = metric
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.reflection_lm = reflection_lm
        self.gepa_auto = gepa_auto
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.resume_from = Path(resume_from) if resume_from else None
        self.teacher_base_url = teacher_base_url
        self.teacher_model = teacher_model

    def optimize_router(
        self,
        module: dspy.Module,
        trainset: Optional[list[dspy.Example]] = None,
        devset: Optional[list[dspy.Example]] = None,
        teacher_lm: Optional[dspy.LM] = None,
    ) -> dspy.Module:
        """
        Optimize a router module.

        Args:
            module: Router module to optimize
            trainset: Training examples (loads default if None)
            devset: Validation examples for GEPA/MIPROv2 (auto-split if None)
            teacher_lm: Teacher LM for optimization (uses orchestrator if None)

        Returns:
            Optimized module
        """
        if trainset is None:
            from .datasets import load_all_routing_examples

            all_examples = load_all_routing_examples()
            if all_examples and self.strategy in ("gepa", "mipro"):
                trainset, devset = get_train_dev_split(all_examples, dev_ratio=0.8)
            else:
                trainset = load_routing_dataset()

        if not trainset:
            logger.warning("No training data for router optimization")
            return module

        metric = self.metric or routing_accuracy
        teacher = teacher_lm or get_teacher_lm(
            base_url=self.teacher_base_url,
            model=self.teacher_model,
        )

        return self._optimize(
            module, trainset, metric, teacher, devset=devset, module_name="router"
        )

    def optimize_orchestrator(
        self,
        module: dspy.Module,
        trainset: Optional[list[dspy.Example]] = None,
        devset: Optional[list[dspy.Example]] = None,
        teacher_lm: Optional[dspy.LM] = None,
    ) -> dspy.Module:
        """
        Optimize an orchestrator module.

        Args:
            module: Orchestrator module to optimize
            trainset: Training examples (loads all training examples if None)
            devset: Validation examples for GEPA/MIPROv2 (auto-split if None)
            teacher_lm: Teacher LM for optimization

        Returns:
            Optimized module
        """
        if trainset is None:
            # Load all training examples and split for GEPA/MIPROv2
            all_examples = load_all_training_examples()
            if all_examples:
                # DSPy recommends 20% train, 80% validation for GEPA/MIPROv2
                trainset, devset = get_train_dev_split(all_examples, dev_ratio=0.8)
                logger.info(
                    f"Auto-split {len(all_examples)} examples: "
                    f"{len(trainset)} train, {len(devset)} dev"
                )
            else:
                # Fall back to legacy dataset
                trainset = load_orchestration_dataset()

        if not trainset:
            logger.warning("No training data for orchestrator optimization")
            return module

        metric = self.metric or orchestration_quality
        teacher = teacher_lm or get_teacher_lm(
            base_url=self.teacher_base_url,
            model=self.teacher_model,
        )

        return self._optimize(
            module, trainset, metric, teacher, devset=devset, module_name="orchestrator"
        )

    def _optimize(
        self,
        module: dspy.Module,
        trainset: list[dspy.Example],
        metric: Callable,
        teacher_lm: dspy.LM,
        devset: Optional[list[dspy.Example]] = None,
        module_name: str = "module",
    ) -> dspy.Module:
        """
        Run optimization with selected strategy.

        Args:
            module: Module to optimize
            trainset: Training examples
            metric: Metric function
            teacher_lm: Teacher LM
            devset: Validation examples (used by GEPA/MIPROv2)
            module_name: Name of the module (used for checkpoint directory)

        Returns:
            Optimized module
        """
        logger.info(
            f"Starting optimization with strategy={self.strategy}, "
            f"trainset_size={len(trainset)}, devset_size={len(devset) if devset else 0}"
        )

        # Resume from checkpoint if specified
        if self.resume_from:
            logger.info(f"Resuming from checkpoint: {self.resume_from}")
            module = CheckpointManager.load_checkpoint(module, self.resume_from)

        # Set up checkpoint manager if checkpoint_dir is specified
        checkpoint_manager: Optional[CheckpointManager] = None
        if self.checkpoint_dir:
            checkpoint_manager = CheckpointManager(
                checkpoint_dir=self.checkpoint_dir,
                module_name=module_name,
                strategy=self.strategy,
            )

        try:
            with dspy.context(lm=teacher_lm):
                if self.strategy == "bootstrap":
                    # Wrap metric for checkpointing
                    wrapped_metric = metric
                    if checkpoint_manager:
                        wrapped_metric = checkpoint_manager.create_metric_wrapper(
                            metric, module
                        )

                    optimizer = BootstrapFewShot(
                        metric=wrapped_metric,
                        max_bootstrapped_demos=self.max_bootstrapped_demos,
                    )
                    optimized = optimizer.compile(module, trainset=trainset)

                elif self.strategy == "mipro":
                    # Wrap metric for checkpointing
                    wrapped_metric = metric
                    if checkpoint_manager:
                        wrapped_metric = checkpoint_manager.create_metric_wrapper(
                            metric, module
                        )

                    optimizer = MIPROv2(
                        metric=wrapped_metric,
                        num_candidates=5,
                        init_temperature=0.7,
                    )
                    optimized = optimizer.compile(
                        module,
                        trainset=trainset,
                        max_labeled_demos=self.max_labeled_demos,
                    )

                elif self.strategy == "gepa":
                    # GEPA requires a reflection LM (can be same as teacher)
                    reflection_lm = self.reflection_lm or teacher_lm
                    # Wrap metric to match GEPA's GEPAFeedbackMetric protocol
                    gepa_metric = _wrap_metric_for_gepa(metric)

                    # Wrap GEPA metric for checkpointing
                    if checkpoint_manager:
                        gepa_metric = checkpoint_manager.create_gepa_metric_wrapper(
                            gepa_metric, module
                        )

                    optimizer = GEPA(
                        metric=gepa_metric,
                        reflection_lm=reflection_lm,
                        auto=self.gepa_auto,
                    )
                    # GEPA benefits from a devset for validation
                    compile_kwargs = {"trainset": trainset}
                    if devset:
                        compile_kwargs["valset"] = devset
                    optimized = optimizer.compile(module, **compile_kwargs)

                else:
                    raise ValueError(f"Unknown optimization strategy: {self.strategy}")

        finally:
            # Finalize checkpointing even if optimization fails
            if checkpoint_manager:
                checkpoint_manager.finalize()

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
    strategy: str = "gepa",
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
