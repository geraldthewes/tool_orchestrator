"""
Checkpoint Manager for DSPy Optimization.

Manages checkpointing during optimization so that:
1. Progress is saved when a better model is found
2. Training can be resumed from the best checkpoint if interrupted
3. Checkpoint history is tracked with metadata
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import dspy

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages checkpointing during DSPy optimization.

    Checkpoints are saved when the metric improves, allowing optimization
    to be resumed from the best known state if interrupted.
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        module_name: str,
        strategy: str = "",
        valset_size: int = 0,
    ):
        """
        Initialize the checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            module_name: Name of the module being optimized (e.g., "orchestrator")
            strategy: Optimization strategy name for metadata
            valset_size: Size of validation set (for batch-based checkpointing)
        """
        self.checkpoint_dir = Path(checkpoint_dir) / module_name
        self.module_name = module_name
        self.strategy = strategy
        self.best_score: float = float("-inf")
        self.checkpoint_count: int = 0
        self.manifest: dict[str, Any] = {
            "module_name": module_name,
            "strategy": strategy,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "checkpoints": [],
            "best": None,
        }
        self._module_ref: Optional[dspy.Module] = None
        self._eval_scores: list[float] = []  # Track scores in current evaluation
        self._valset_size: int = valset_size  # Size of validation set
        self._perfect_score_reached: bool = False  # Flag for early stopping

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")

    def set_module_ref(self, module: dspy.Module) -> None:
        """
        Set the module reference for checkpointing.

        Args:
            module: The DSPy module being optimized
        """
        self._module_ref = module

    def create_metric_wrapper(
        self, base_metric: Callable, module_ref: dspy.Module
    ) -> Callable:
        """
        Wrap a metric function to checkpoint based on average score.

        Accumulates scores and checkpoints based on average across the full
        validation set, preventing single high-scoring examples from gaming
        the checkpoint system.

        Args:
            base_metric: The original metric function
            module_ref: Reference to the module being optimized

        Returns:
            Wrapped metric function that checkpoints on improvement
        """
        self._module_ref = module_ref

        def wrapped_metric(example, prediction, trace=None):
            score = base_metric(example, prediction, trace)
            self._eval_scores.append(score)

            # If we know valset size, checkpoint after each complete evaluation
            if self._valset_size > 0 and len(self._eval_scores) >= self._valset_size:
                self.evaluate_and_checkpoint()

            return score

        return wrapped_metric

    def create_gepa_metric_wrapper(
        self, base_metric: Callable, module_ref: dspy.Module
    ) -> Callable:
        """
        Wrap a GEPA-compatible metric function to checkpoint based on average score.

        GEPA expects: metric(gold, pred, trace, pred_name, pred_trace) -> float

        Accumulates scores and checkpoints based on average across the full
        validation set, preventing single high-scoring examples from gaming
        the checkpoint system.

        Args:
            base_metric: The original GEPA metric function
            module_ref: Reference to the module being optimized

        Returns:
            Wrapped metric function that checkpoints on improvement
        """
        self._module_ref = module_ref

        def wrapped_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
            score = base_metric(gold, pred, trace, pred_name, pred_trace)
            self._eval_scores.append(score)

            # If we know valset size, checkpoint after each complete evaluation
            if self._valset_size > 0 and len(self._eval_scores) >= self._valset_size:
                self.evaluate_and_checkpoint()

            return score

        return wrapped_metric

    def evaluate_and_checkpoint(self) -> float:
        """
        Call after a full evaluation pass to checkpoint if improved.

        Computes average score across accumulated evaluation scores,
        resets the accumulator, and saves checkpoint if the average
        exceeds the best score seen so far.

        Returns:
            Average score for this evaluation pass
        """
        if not self._eval_scores:
            return 0.0

        avg_score = sum(self._eval_scores) / len(self._eval_scores)
        num_examples = len(self._eval_scores)
        self._eval_scores = []  # Reset for next evaluation

        logger.info(
            f"Evaluation complete: avg_score={avg_score:.4f} "
            f"(over {num_examples} examples), best_score={self.best_score:.4f}"
        )

        if avg_score > self.best_score:
            self.best_score = avg_score
            self._save_checkpoint(avg_score)

        # Check for perfect score (early stopping signal)
        if avg_score >= 1.0:
            self._perfect_score_reached = True
            logger.info("Perfect score achieved (1.0), signaling early stop")

        return avg_score

    def should_stop_early(self) -> bool:
        """
        Check if optimization should stop early.

        Returns:
            True if perfect score was reached
        """
        return self._perfect_score_reached

    def _save_checkpoint(self, score: float) -> Optional[Path]:
        """
        Save module state and update manifest.

        Args:
            score: The score that triggered this checkpoint

        Returns:
            Path to the saved checkpoint file, or None if module not set
        """
        if self._module_ref is None:
            logger.warning("No module reference set, cannot save checkpoint")
            return None

        self.checkpoint_count += 1
        checkpoint_filename = f"checkpoint_{self.checkpoint_count:03d}.json"
        checkpoint_path = self.checkpoint_dir / checkpoint_filename

        try:
            # Save module state
            self._module_ref.save(str(checkpoint_path))

            # Create checkpoint entry
            checkpoint_entry = {
                "id": self.checkpoint_count,
                "score": score,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "path": checkpoint_filename,
            }

            # Update manifest
            self.manifest["checkpoints"].append(checkpoint_entry)
            self.manifest["best"] = checkpoint_entry
            self.save_manifest()

            logger.info(
                f"Checkpoint {self.checkpoint_count} saved: score={score:.4f}, "
                f"path={checkpoint_path}"
            )

            return checkpoint_path

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise

    def save_manifest(self) -> None:
        """Write manifest.json with checkpoint history."""
        manifest_path = self.checkpoint_dir / "manifest.json"
        self.manifest["updated_at"] = datetime.now(timezone.utc).isoformat()

        with open(manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)

        logger.debug(f"Manifest saved to {manifest_path}")

    def finalize(self) -> None:
        """
        Finalize checkpointing after optimization completes.

        Processes any remaining accumulated scores and saves final manifest
        with completion timestamp.
        """
        # Process any remaining scores that weren't checkpointed
        if self._eval_scores:
            logger.info(
                f"Processing {len(self._eval_scores)} remaining scores at finalization"
            )
            self.evaluate_and_checkpoint()

        self.manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        self.manifest["final_best_score"] = self.best_score
        self.save_manifest()
        logger.info(
            f"Checkpointing finalized. {self.checkpoint_count} checkpoints saved. "
            f"Best score: {self.best_score:.4f}"
        )

    @classmethod
    def get_best_checkpoint(cls, checkpoint_dir: Path) -> Optional[Path]:
        """
        Return path to best checkpoint for resuming.

        Args:
            checkpoint_dir: Directory containing checkpoints

        Returns:
            Path to the best checkpoint file, or None if not found
        """
        manifest_path = checkpoint_dir / "manifest.json"

        if not manifest_path.exists():
            logger.debug(f"No manifest found at {manifest_path}")
            return None

        try:
            with open(manifest_path) as f:
                manifest = json.load(f)

            best = manifest.get("best")
            if best is None:
                logger.debug("No best checkpoint in manifest")
                return None

            checkpoint_path = checkpoint_dir / best["path"]
            if not checkpoint_path.exists():
                logger.warning(f"Best checkpoint file not found: {checkpoint_path}")
                return None

            logger.info(
                f"Found best checkpoint: {checkpoint_path} (score: {best['score']:.4f})"
            )
            return checkpoint_path

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to read manifest: {e}")
            return None

    @classmethod
    def load_checkpoint(cls, module: dspy.Module, checkpoint_path: Path) -> dspy.Module:
        """
        Load module state from checkpoint.

        Args:
            module: Module to load state into
            checkpoint_path: Path to checkpoint file

        Returns:
            Module with loaded state
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        module.load(str(checkpoint_path))
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return module

    @classmethod
    def get_manifest(cls, checkpoint_dir: Path) -> Optional[dict[str, Any]]:
        """
        Load manifest from checkpoint directory.

        Args:
            checkpoint_dir: Directory containing checkpoints

        Returns:
            Manifest dict or None if not found
        """
        manifest_path = checkpoint_dir / "manifest.json"

        if not manifest_path.exists():
            return None

        try:
            with open(manifest_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to read manifest: {e}")
            return None

    @classmethod
    def get_best_score(cls, checkpoint_dir: Path) -> Optional[float]:
        """
        Get the best score from checkpoint manifest.

        Args:
            checkpoint_dir: Directory containing checkpoints

        Returns:
            Best score or None if not found
        """
        manifest = cls.get_manifest(checkpoint_dir)
        if manifest is None:
            return None

        best = manifest.get("best")
        if best is None:
            return None

        return best.get("score")
