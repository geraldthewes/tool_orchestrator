"""
DSPy Dataset Loading for ToolOrchestra.

Provides functions for loading training and evaluation datasets
for DSPy optimization.
"""

import json
import logging
import random
from pathlib import Path
from typing import Optional, Union

import dspy

logger = logging.getLogger(__name__)

# Default paths relative to project root
DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"
DEFAULT_EXAMPLES_DIR = DEFAULT_DATA_DIR / "examples"


def load_routing_dataset(
    path: Optional[Union[str, Path]] = None,
    split: str = "train",
) -> list[dspy.Example]:
    """
    Load query routing examples for optimization.

    Expected JSONL format:
    {"query": "...", "needs_tools": true/false, "reasoning": "...", "direct_answer": "..."}

    Args:
        path: Path to JSONL file. Defaults to data/routing_examples.jsonl
        split: Dataset split to load ("train", "dev", "test")

    Returns:
        List of DSPy Examples
    """
    resolved_path: Path
    if path is None:
        resolved_path = DEFAULT_DATA_DIR / "routing_examples.jsonl"
    else:
        resolved_path = Path(path)

    if not resolved_path.exists():
        logger.warning(f"Routing dataset not found at {resolved_path}")
        return []

    examples = []
    with open(resolved_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                example = dspy.Example(
                    query=data["query"],
                    available_tools=data.get("available_tools", ""),
                    needs_tools=data.get("needs_tools", True),
                    reasoning=data.get("reasoning", ""),
                    direct_answer=data.get("direct_answer", ""),
                ).with_inputs("query", "available_tools")
                examples.append(example)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse routing example: {e}")
                continue

    logger.info(f"Loaded {len(examples)} routing examples from {resolved_path}")
    return examples


def load_orchestration_dataset(
    path: Optional[Union[str, Path]] = None,
    split: str = "train",
) -> list[dspy.Example]:
    """
    Load orchestration examples for optimization.

    Expected JSONL format:
    {"question": "...", "answer": "...", "expected_keywords": [...]}

    Args:
        path: Path to JSONL file. Defaults to data/orchestration_examples.jsonl
        split: Dataset split to load ("train", "dev", "test")

    Returns:
        List of DSPy Examples
    """
    resolved_path: Path
    if path is None:
        resolved_path = DEFAULT_DATA_DIR / "orchestration_examples.jsonl"
    else:
        resolved_path = Path(path)

    if not resolved_path.exists():
        logger.warning(f"Orchestration dataset not found at {resolved_path}")
        return []

    examples = []
    with open(resolved_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                example = dspy.Example(
                    question=data["question"],
                    answer=data.get("answer", ""),
                    expected_keywords=data.get("expected_keywords", []),
                ).with_inputs("question")
                examples.append(example)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse orchestration example: {e}")
                continue

    logger.info(f"Loaded {len(examples)} orchestration examples from {resolved_path}")
    return examples


def create_routing_example(
    query: str,
    needs_tools: bool,
    reasoning: str = "",
    direct_answer: str = "",
    available_tools: str = "",
) -> dspy.Example:
    """
    Create a single routing example.

    Args:
        query: The user query
        needs_tools: Whether tools are needed
        reasoning: Explanation for the decision
        direct_answer: Direct answer if no tools needed
        available_tools: Tool descriptions

    Returns:
        DSPy Example
    """
    return dspy.Example(
        query=query,
        available_tools=available_tools,
        needs_tools=needs_tools,
        reasoning=reasoning,
        direct_answer=direct_answer,
    ).with_inputs("query", "available_tools")


def create_orchestration_example(
    question: str,
    answer: str = "",
    expected_keywords: Optional[list[str]] = None,
    tool: Optional[str] = None,
    category: Optional[str] = None,
    **kwargs,
) -> dspy.Example:
    """
    Create a single orchestration example.

    Args:
        question: The user question
        answer: Expected answer
        expected_keywords: Keywords expected in answer
        tool: The tool used (optional)
        category: Example category (optional)

    Returns:
        DSPy Example
    """
    example_data = {
        "question": question,
        "answer": answer,
        "expected_keywords": expected_keywords or [],
    }
    if tool:
        example_data["tool"] = tool
    if category:
        example_data["category"] = category
    return dspy.Example(**example_data).with_inputs("question")


def load_examples_from_directory(
    directory: Optional[Union[str, Path]] = None,
    pattern: str = "*.jsonl",
    example_type: str = "orchestration",
) -> list[dspy.Example]:
    """
    Load all examples from a directory of JSONL files.

    Args:
        directory: Path to examples directory. Defaults to data/examples
        pattern: Glob pattern for files to load
        example_type: Type of examples ("orchestration" or "routing")

    Returns:
        List of DSPy Examples
    """
    resolved_dir: Path
    if directory is None:
        resolved_dir = DEFAULT_EXAMPLES_DIR
    else:
        resolved_dir = Path(directory)

    if not resolved_dir.exists():
        logger.warning(f"Examples directory not found: {resolved_dir}")
        return []

    examples = []
    for filepath in sorted(resolved_dir.glob(pattern)):
        # Determine type from filename if not specified
        is_routing = "routing" in filepath.name
        file_type = "routing" if is_routing else "orchestration"

        # Skip files that don't match requested type
        if example_type != "all" and file_type != example_type:
            continue

        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if file_type == "routing":
                        example = create_routing_example(
                            query=data["query"],
                            needs_tools=data.get("needs_tools", True),
                            reasoning=data.get("reasoning", ""),
                            direct_answer=data.get("direct_answer", ""),
                            available_tools=data.get("available_tools", ""),
                        )
                    else:
                        example = create_orchestration_example(
                            question=data["question"],
                            answer=data.get("answer", ""),
                            expected_keywords=data.get("expected_keywords", []),
                            tool=data.get("tool"),
                            category=data.get("category"),
                        )
                    examples.append(example)
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Failed to parse example from {filepath}: {e}")
                    continue

    logger.info(f"Loaded {len(examples)} {example_type} examples from {directory}")
    return examples


def load_all_training_examples(
    examples_dir: Optional[str] = None,
    include_legacy: bool = True,
) -> list[dspy.Example]:
    """
    Load all training examples from both new and legacy locations.

    Args:
        examples_dir: Path to examples directory
        include_legacy: Whether to include legacy orchestration_examples.jsonl

    Returns:
        List of DSPy Examples
    """
    examples = []

    # Load from new examples directory
    examples.extend(
        load_examples_from_directory(examples_dir, example_type="orchestration")
    )

    # Optionally load legacy examples
    if include_legacy:
        legacy_path = DEFAULT_DATA_DIR / "orchestration_examples.jsonl"
        if legacy_path.exists():
            legacy = load_orchestration_dataset(str(legacy_path))
            examples.extend(legacy)
            logger.info(f"Added {len(legacy)} legacy orchestration examples")

    return examples


def load_all_routing_examples(
    examples_dir: Optional[str] = None,
    include_legacy: bool = True,
) -> list[dspy.Example]:
    """
    Load all routing examples from both new and legacy locations.

    Args:
        examples_dir: Path to examples directory
        include_legacy: Whether to include legacy routing_examples.jsonl

    Returns:
        List of DSPy Examples
    """
    examples = []

    # Load from new examples directory
    examples.extend(load_examples_from_directory(examples_dir, example_type="routing"))

    # Optionally load legacy examples
    if include_legacy:
        legacy_path = DEFAULT_DATA_DIR / "routing_examples.jsonl"
        if legacy_path.exists():
            legacy = load_routing_dataset(str(legacy_path))
            examples.extend(legacy)
            logger.info(f"Added {len(legacy)} legacy routing examples")

    return examples


def get_train_dev_split(
    examples: list[dspy.Example],
    dev_ratio: float = 0.2,
    seed: Optional[int] = 42,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """
    Split examples into training and development sets.

    Args:
        examples: List of examples to split
        dev_ratio: Fraction of examples for dev set (0.0-1.0)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_examples, dev_examples)
    """
    if seed is not None:
        random.seed(seed)

    shuffled = examples.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * (1 - dev_ratio))
    train = shuffled[:split_idx]
    dev = shuffled[split_idx:]

    logger.info(f"Split {len(examples)} examples: {len(train)} train, {len(dev)} dev")
    return train, dev


def get_examples_by_tool(
    examples: list[dspy.Example],
    tool: str,
) -> list[dspy.Example]:
    """
    Filter examples by tool type.

    Args:
        examples: List of examples to filter
        tool: Tool name to filter by

    Returns:
        Filtered list of examples
    """
    return [ex for ex in examples if getattr(ex, "tool", None) == tool]


def get_examples_by_category(
    examples: list[dspy.Example],
    category: str,
) -> list[dspy.Example]:
    """
    Filter examples by category.

    Args:
        examples: List of examples to filter
        category: Category to filter by

    Returns:
        Filtered list of examples
    """
    return [ex for ex in examples if getattr(ex, "category", None) == category]
