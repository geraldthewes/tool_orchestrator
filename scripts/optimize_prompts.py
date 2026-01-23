#!/usr/bin/env python
"""
Run DSPy prompt optimization with GEPA.

This script optimizes the DSPy modules used by ToolOrchestra using
the configured training examples and selected optimization strategy.

Usage:
    python scripts/optimize_prompts.py --strategy gepa
    python scripts/optimize_prompts.py --strategy bootstrap --output-dir data/optimized
    python scripts/optimize_prompts.py --module orchestrator --dry-run
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prompts.optimization import (
    PromptOptimizer,
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
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def optimize_orchestrator_module(
    optimizer: PromptOptimizer,
    output_path: Path,
    dev_ratio: float,
    dry_run: bool,
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

    if dry_run:
        logger.info(f"[DRY RUN] Would optimize orchestrator module with {len(examples)} examples")
        logger.info(f"[DRY RUN] Would save to {output_path / 'orchestrator.json'}")
        return output_path / "orchestrator.json"

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

    if dry_run:
        logger.info(f"[DRY RUN] Would optimize router module with {len(examples)} examples")
        logger.info(f"[DRY RUN] Would save to {output_path / 'router.json'}")
        return output_path / "router.json"

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

    optimizer = PromptOptimizer(
        strategy=args.strategy,
        gepa_auto=args.gepa_auto,
    )

    saved_paths = {}

    if args.module in ("all", "orchestrator"):
        path = optimize_orchestrator_module(
            optimizer, output_path, args.dev_ratio, args.dry_run
        )
        if path:
            saved_paths["orchestrator"] = path

    if args.module in ("all", "router"):
        path = optimize_router_module(
            optimizer, output_path, args.dev_ratio, args.dry_run
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
