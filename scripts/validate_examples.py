#!/usr/bin/env python3
"""
Validate DSPy training examples for quality and consistency.

Run with: python -m scripts.validate_examples
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExampleValidator:
    """Validate DSPy training examples."""

    VALID_TOOLS = {
        "calculate",
        "python_execute",
        "web_search",
        "ask_reasoner",
        "ask_coder",
        "ask_fast",
    }

    VALID_CATEGORIES = {
        "single_tool",
        "multi_tool",
        "delegate",
        "greeting",
        "basic_knowledge",
        "definition",
        "simple_math",
        "advice",
    }

    def __init__(self, examples_dir: str = "data/examples"):
        self.examples_dir = Path(examples_dir)
        self.errors: list[dict[str, Any]] = []
        self.warnings: list[dict[str, Any]] = []
        self.stats: dict[str, Any] = defaultdict(int)

    def validate_orchestration_example(
        self, example: dict[str, Any], filename: str, line_num: int
    ) -> tuple[bool, list[str]]:
        """Validate a single orchestration example."""
        errors = []

        # Required fields
        if "question" not in example:
            errors.append("Missing 'question' field")
        elif not example["question"].strip():
            errors.append("Empty 'question' field")
        elif len(example["question"]) < 5:
            errors.append("Question too short (< 5 chars)")

        if "answer" not in example:
            errors.append("Missing 'answer' field")
        elif not example["answer"].strip():
            errors.append("Empty 'answer' field")

        # Optional but recommended fields
        if "expected_keywords" in example:
            keywords = example["expected_keywords"]
            if not isinstance(keywords, list):
                errors.append("'expected_keywords' must be a list")
            elif len(keywords) == 0:
                self.warnings.append(
                    {
                        "file": filename,
                        "line": line_num,
                        "message": "Empty expected_keywords list",
                    }
                )

        # Tool validation
        if "tool" in example:
            tool = example["tool"]
            if tool not in self.VALID_TOOLS:
                errors.append(f"Invalid tool: {tool}")
            self.stats[f"tool_{tool}"] += 1

        # Category validation
        if "category" in example:
            category = example["category"]
            if category not in self.VALID_CATEGORIES:
                self.warnings.append(
                    {
                        "file": filename,
                        "line": line_num,
                        "message": f"Unrecognized category: {category}",
                    }
                )
            self.stats[f"category_{category}"] += 1

        return len(errors) == 0, errors

    def validate_routing_example(
        self, example: dict[str, Any], filename: str, line_num: int
    ) -> tuple[bool, list[str]]:
        """Validate a single routing example."""
        errors = []

        # Required fields
        if "query" not in example:
            errors.append("Missing 'query' field")
        elif not example["query"].strip():
            errors.append("Empty 'query' field")

        if "needs_tools" not in example:
            errors.append("Missing 'needs_tools' field")
        elif not isinstance(example["needs_tools"], bool):
            errors.append("'needs_tools' must be boolean")

        if "reasoning" not in example:
            errors.append("Missing 'reasoning' field")

        # If doesn't need tools, should have direct_answer
        if example.get("needs_tools") is False:
            if "direct_answer" not in example or not example["direct_answer"].strip():
                self.warnings.append(
                    {
                        "file": filename,
                        "line": line_num,
                        "message": "No direct_answer for needs_tools=false",
                    }
                )

        # Track stats
        if example.get("needs_tools"):
            self.stats["routing_needs_tools"] += 1
        else:
            self.stats["routing_no_tools"] += 1

        return len(errors) == 0, errors

    def validate_file(self, filepath: Path) -> tuple[int, int, int]:
        """Validate all examples in a file."""
        valid_count = 0
        invalid_count = 0
        total_count = 0

        # Determine example type from filename
        is_routing = "routing" in filepath.name

        with open(filepath, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                total_count += 1

                try:
                    example = json.loads(line)
                except json.JSONDecodeError as e:
                    self.errors.append(
                        {
                            "file": filepath.name,
                            "line": line_num,
                            "errors": [f"JSON parse error: {e}"],
                        }
                    )
                    invalid_count += 1
                    continue

                if is_routing:
                    is_valid, errors = self.validate_routing_example(
                        example, filepath.name, line_num
                    )
                else:
                    is_valid, errors = self.validate_orchestration_example(
                        example, filepath.name, line_num
                    )

                if is_valid:
                    valid_count += 1
                else:
                    invalid_count += 1
                    self.errors.append(
                        {
                            "file": filepath.name,
                            "line": line_num,
                            "errors": errors,
                        }
                    )

        return total_count, valid_count, invalid_count

    def check_duplicates(self) -> list[dict[str, Any]]:
        """Check for duplicate questions/queries across all files."""
        seen: dict[str, list[str]] = defaultdict(list)
        duplicates = []

        for filepath in self.examples_dir.glob("*.jsonl"):
            is_routing = "routing" in filepath.name
            key_field = "query" if is_routing else "question"

            with open(filepath, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        example = json.loads(line)
                        key = example.get(key_field, "").lower().strip()
                        if key:
                            seen[key].append(filepath.name)
                    except json.JSONDecodeError:
                        continue

        for key, files in seen.items():
            if len(files) > 1:
                duplicates.append(
                    {
                        "text": key[:50] + "..." if len(key) > 50 else key,
                        "files": files,
                        "count": len(files),
                    }
                )

        return duplicates

    def check_balance(self) -> dict[str, Any]:
        """Check distribution balance across categories."""
        balance = {
            "tools": {},
            "categories": {},
            "routing": {
                "needs_tools": self.stats.get("routing_needs_tools", 0),
                "no_tools": self.stats.get("routing_no_tools", 0),
            },
        }

        for key, value in self.stats.items():
            if key.startswith("tool_"):
                tool_name = key[5:]
                balance["tools"][tool_name] = value
            elif key.startswith("category_"):
                cat_name = key[9:]
                balance["categories"][cat_name] = value

        return balance

    def validate_all(self) -> dict[str, Any]:
        """Validate all example files."""
        results = {
            "files": {},
            "total": {"count": 0, "valid": 0, "invalid": 0},
            "errors": [],
            "warnings": [],
            "duplicates": [],
            "balance": {},
        }

        if not self.examples_dir.exists():
            logger.error(f"Examples directory not found: {self.examples_dir}")
            return results

        # Validate each file
        for filepath in sorted(self.examples_dir.glob("*.jsonl")):
            total, valid, invalid = self.validate_file(filepath)
            results["files"][filepath.name] = {
                "total": total,
                "valid": valid,
                "invalid": invalid,
            }
            results["total"]["count"] += total
            results["total"]["valid"] += valid
            results["total"]["invalid"] += invalid

            logger.info(f"{filepath.name}: {valid}/{total} valid ({invalid} invalid)")

        # Check for duplicates
        results["duplicates"] = self.check_duplicates()

        # Check balance
        results["balance"] = self.check_balance()

        # Add errors and warnings
        results["errors"] = self.errors
        results["warnings"] = self.warnings

        return results

    def print_report(self, results: dict[str, Any]) -> None:
        """Print validation report."""
        print("\n" + "=" * 60)
        print("VALIDATION REPORT")
        print("=" * 60)

        # Summary
        print(f"\nTotal examples: {results['total']['count']}")
        print(f"Valid: {results['total']['valid']}")
        print(f"Invalid: {results['total']['invalid']}")

        # Per-file breakdown
        print("\nPer-file breakdown:")
        for filename, stats in results["files"].items():
            status = "✓" if stats["invalid"] == 0 else "✗"
            print(f"  {status} {filename}: {stats['valid']}/{stats['total']}")

        # Balance report
        print("\nTool distribution:")
        for tool, count in sorted(results["balance"].get("tools", {}).items()):
            print(f"  {tool}: {count}")

        print("\nCategory distribution:")
        for cat, count in sorted(results["balance"].get("categories", {}).items()):
            print(f"  {cat}: {count}")

        routing = results["balance"].get("routing", {})
        print("\nRouting distribution:")
        print(f"  Needs tools: {routing.get('needs_tools', 0)}")
        print(f"  No tools: {routing.get('no_tools', 0)}")

        # Duplicates
        if results["duplicates"]:
            print(f"\nDuplicates found: {len(results['duplicates'])}")
            for dup in results["duplicates"][:5]:
                print(f"  '{dup['text']}' in {dup['files']}")
            if len(results["duplicates"]) > 5:
                print(f"  ... and {len(results['duplicates']) - 5} more")

        # Errors
        if results["errors"]:
            print(f"\nErrors ({len(results['errors'])}):")
            for err in results["errors"][:10]:
                print(f"  {err['file']}:{err['line']}: {err['errors']}")
            if len(results["errors"]) > 10:
                print(f"  ... and {len(results['errors']) - 10} more")

        # Warnings
        if results["warnings"]:
            print(f"\nWarnings ({len(results['warnings'])}):")
            for warn in results["warnings"][:5]:
                print(f"  {warn['file']}:{warn['line']}: {warn['message']}")
            if len(results["warnings"]) > 5:
                print(f"  ... and {len(results['warnings']) - 5} more")

        print("\n" + "=" * 60)


def main():
    """Main entry point."""
    validator = ExampleValidator()
    results = validator.validate_all()
    validator.print_report(results)

    # Exit with error code if validation failed
    if results["total"]["invalid"] > 0:
        exit(1)


if __name__ == "__main__":
    main()
