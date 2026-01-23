#!/usr/bin/env python
"""
Run DSPy prompt optimization with GEPA.

This script optimizes the DSPy modules used by ToolOrchestra using
the configured training examples and selected optimization strategy.

Supports checkpointing to save progress when improvements are found,
allowing optimization to be resumed from the best checkpoint if interrupted.

Usage:
    python scripts/optimize_prompts.py --strategy gepa
    python scripts/optimize_prompts.py --strategy bootstrap --output-dir data/optimized
    python scripts/optimize_prompts.py --module orchestrator --dry-run

    # With checkpointing (enabled by default)
    python scripts/optimize_prompts.py --checkpoint-dir data/checkpoints

    # Resume from best checkpoint
    python scripts/optimize_prompts.py --resume --checkpoint-dir data/checkpoints

    # Disable checkpointing
    python scripts/optimize_prompts.py --no-checkpoint
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prompts.optimization import (
    PromptOptimizer,
    CheckpointManager,
    load_all_training_examples,
    load_all_routing_examples,
    get_train_dev_split,
)
from src.prompts.modules import QueryRouterModule, ToolOrchestratorModule

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run DSPy prompt optimization for ToolOrchestra",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--strategy",
        default="gepa",
        choices=["gepa", "mipro", "bootstrap"],
        help="Optimization strategy (default: gepa)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/optimized_prompts",
        help="Directory to save optimized modules (default: data/optimized_prompts)",
    )
    parser.add_argument(
        "--module",
        choices=["all", "orchestrator", "router"],
        default="all",
        help="Which module(s) to optimize (default: all)",
    )
    parser.add_argument(
        "--gepa-auto",
        choices=["light", "medium", "heavy"],
        default="light",
        help="GEPA preset: light (fast), medium, heavy (thorough) (default: light)",
    )
    parser.add_argument(
        "--dev-ratio",
        type=float,
        default=0.8,
        help="Ratio of examples for validation set (default: 0.8)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without running optimization",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="data/optimized_prompts/checkpoints",
        help="Directory to save checkpoints (default: data/optimized_prompts/checkpoints)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from best checkpoint if available",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable checkpointing",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--teacher-base-url",
        help="Base URL for teacher LLM (overrides TEACHER_BASE_URL env var)",
    )
    parser.add_argument(
        "--teacher-model",
        help="Model name for teacher LLM (overrides TEACHER_MODEL env var)",
    )
    parser.add_argument(
        "--teacher-max-tokens",
        type=int,
        help="Max tokens for teacher LLM responses (overrides config, default: 4096)",
    )
    return parser.parse_args()


def optimize_orchestrator_module(
    optimizer: PromptOptimizer,
    output_path: Path,
    dev_ratio: float,
    dry_run: bool,
    checkpoint_dir: Path = None,
    resume: bool = False,
) -> Path:
    """Optimize the orchestrator module."""
    logger.info("Loading orchestration training examples...")
    examples = load_all_training_examples()
    logger.info(f"Loaded {len(examples)} orchestration examples")

    if not examples:
        logger.error("No training examples found for orchestrator")
        return None

    trainset, devset = get_train_dev_split(examples, dev_ratio=dev_ratio)
    logger.info(f"Split into {len(trainset)} train, {len(devset)} dev examples")

    # Check for checkpoint to resume from
    resume_from = None
    if resume and checkpoint_dir:
        module_checkpoint_dir = checkpoint_dir / "orchestrator"
        resume_from = CheckpointManager.get_best_checkpoint(module_checkpoint_dir)
        if resume_from:
            logger.info(f"Will resume from checkpoint: {resume_from}")
        else:
            logger.info("No checkpoint found to resume from, starting fresh")

    if dry_run:
        logger.info(f"[DRY RUN] Would optimize orchestrator module with {len(examples)} examples")
        logger.info(f"[DRY RUN] Would save to {output_path / 'orchestrator.json'}")
        if checkpoint_dir:
            logger.info(f"[DRY RUN] Checkpoints would be saved to {checkpoint_dir / 'orchestrator'}")
        if resume_from:
            logger.info(f"[DRY RUN] Would resume from {resume_from}")
        return output_path / "orchestrator.json"

    # Update optimizer with checkpoint settings
    optimizer.checkpoint_dir = checkpoint_dir
    optimizer.resume_from = resume_from

    logger.info("Optimizing orchestrator module...")
    module = ToolOrchestratorModule()
    optimized = optimizer.optimize_orchestrator(
        module, trainset=trainset, devset=devset
    )

    save_path = output_path / "orchestrator.json"
    PromptOptimizer.save(optimized, str(save_path))
    logger.info(f"Saved optimized orchestrator to {save_path}")

    return save_path


def optimize_router_module(
    optimizer: PromptOptimizer,
    output_path: Path,
    dev_ratio: float,
    dry_run: bool,
    checkpoint_dir: Path = None,
    resume: bool = False,
) -> Path:
    """Optimize the router module."""
    logger.info("Loading routing training examples...")
    examples = load_all_routing_examples()
    logger.info(f"Loaded {len(examples)} routing examples")

    if not examples:
        logger.warning("No training examples found for router, skipping")
        return None

    trainset, devset = get_train_dev_split(examples, dev_ratio=dev_ratio)
    logger.info(f"Split into {len(trainset)} train, {len(devset)} dev examples")

    # Check for checkpoint to resume from
    resume_from = None
    if resume and checkpoint_dir:
        module_checkpoint_dir = checkpoint_dir / "router"
        resume_from = CheckpointManager.get_best_checkpoint(module_checkpoint_dir)
        if resume_from:
            logger.info(f"Will resume from checkpoint: {resume_from}")
        else:
            logger.info("No checkpoint found to resume from, starting fresh")

    if dry_run:
        logger.info(f"[DRY RUN] Would optimize router module with {len(examples)} examples")
        logger.info(f"[DRY RUN] Would save to {output_path / 'router.json'}")
        if checkpoint_dir:
            logger.info(f"[DRY RUN] Checkpoints would be saved to {checkpoint_dir / 'router'}")
        if resume_from:
            logger.info(f"[DRY RUN] Would resume from {resume_from}")
        return output_path / "router.json"

    # Update optimizer with checkpoint settings
    optimizer.checkpoint_dir = checkpoint_dir
    optimizer.resume_from = resume_from

    logger.info("Optimizing router module...")
    module = QueryRouterModule()
    optimized = optimizer.optimize_router(module, trainset=trainset, devset=devset)

    save_path = output_path / "router.json"
    PromptOptimizer.save(optimized, str(save_path))
    logger.info(f"Saved optimized router to {save_path}")

    return save_path


def main():
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"Starting prompt optimization with strategy: {args.strategy}")

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set up checkpointing
    checkpoint_dir = None if args.no_checkpoint else Path(args.checkpoint_dir)
    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Checkpointing enabled: {checkpoint_dir}")
    else:
        logger.info("Checkpointing disabled")

    optimizer = PromptOptimizer(
        strategy=args.strategy,
        gepa_auto=args.gepa_auto,
        teacher_base_url=args.teacher_base_url,
        teacher_model=args.teacher_model,
        teacher_max_tokens=args.teacher_max_tokens,
    )

    saved_paths = {}

    if args.module in ("all", "orchestrator"):
        path = optimize_orchestrator_module(
            optimizer,
            output_path,
            args.dev_ratio,
            args.dry_run,
            checkpoint_dir=checkpoint_dir,
            resume=args.resume,
        )
        if path:
            saved_paths["orchestrator"] = path

    if args.module in ("all", "router"):
        path = optimize_router_module(
            optimizer,
            output_path,
            args.dev_ratio,
            args.dry_run,
            checkpoint_dir=checkpoint_dir,
            resume=args.resume,
        )
        if path:
            saved_paths["router"] = path

    if args.dry_run:
        logger.info("[DRY RUN] No optimization performed")
    else:
        logger.info("Optimization complete!")
        for name, path in saved_paths.items():
            logger.info(f"  {name}: {path}")


if __name__ == "__main__":
    main()
