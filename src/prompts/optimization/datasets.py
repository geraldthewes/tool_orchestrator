"""
DSPy Dataset Loading for ToolOrchestra.

Provides functions for loading training and evaluation datasets
for DSPy optimization.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import dspy

logger = logging.getLogger(__name__)

# Default paths relative to project root
DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"


def load_routing_dataset(
    path: Optional[str] = None,
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
    if path is None:
        path = DEFAULT_DATA_DIR / "routing_examples.jsonl"
    else:
        path = Path(path)

    if not path.exists():
        logger.warning(f"Routing dataset not found at {path}")
        return []

    examples = []
    with open(path, "r") as f:
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

    logger.info(f"Loaded {len(examples)} routing examples from {path}")
    return examples


def load_orchestration_dataset(
    path: Optional[str] = None,
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
    if path is None:
        path = DEFAULT_DATA_DIR / "orchestration_examples.jsonl"
    else:
        path = Path(path)

    if not path.exists():
        logger.warning(f"Orchestration dataset not found at {path}")
        return []

    examples = []
    with open(path, "r") as f:
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

    logger.info(f"Loaded {len(examples)} orchestration examples from {path}")
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
) -> dspy.Example:
    """
    Create a single orchestration example.

    Args:
        question: The user question
        answer: Expected answer
        expected_keywords: Keywords expected in answer

    Returns:
        DSPy Example
    """
    return dspy.Example(
        question=question,
        answer=answer,
        expected_keywords=expected_keywords or [],
    ).with_inputs("question")
