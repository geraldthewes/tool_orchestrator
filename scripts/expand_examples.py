#!/usr/bin/env python3
"""
Expand seed examples using LLM-assisted generation.

This script uses an LLM to generate variations of seed examples.
Run with: python -m scripts.expand_examples
"""

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExampleExpander:
    """Expand seed examples using LLM generation."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        examples_dir: str = "data/examples",
    ):
        self.base_url = base_url or os.getenv(
            "TEACHER_BASE_URL", "http://localhost:11434"
        )
        # Normalize base_url: remove trailing slash and /v1 suffix for consistent URL construction
        self.base_url = self.base_url.rstrip("/")
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]
        self.model = model or os.getenv("TEACHER_MODEL", "llama3")
        self.examples_dir = Path(examples_dir)
        self.client = httpx.AsyncClient(timeout=120.0)

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def generate_variations(
        self,
        seed_example: dict[str, Any],
        count: int = 3,
        example_type: str = "orchestration",
    ) -> list[dict[str, Any]]:
        """Generate variations of a seed example using LLM."""
        if example_type == "routing":
            prompt = self._create_routing_prompt(seed_example, count)
        else:
            prompt = self._create_orchestration_prompt(seed_example, count)

        try:
            response = await self.client.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.8,
                },
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return self._parse_json_response(content, example_type)
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error generating variations: {e}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            return []
        except Exception as e:
            logger.error(f"Error generating variations: {e}")
            return []

    def _create_orchestration_prompt(
        self, seed_example: dict[str, Any], count: int
    ) -> str:
        """Create prompt for generating orchestration example variations."""
        tool = seed_example.get("tool", "calculate")
        category = seed_example.get("category", "single_tool")

        return f"""Generate {count} variations of this tool orchestration training example.

Original example:
- Question: {seed_example["question"]}
- Answer: {seed_example["answer"]}
- Expected keywords: {seed_example.get("expected_keywords", [])}
- Tool: {tool}
- Category: {category}

Requirements:
1. Generate {count} NEW examples that would use the SAME tool ({tool})
2. Vary the complexity, numbers, and context
3. Keep answers accurate and factual
4. Include relevant expected_keywords that should appear in the answer

Respond with ONLY a valid JSON array of objects. Each object must have:
- "question": the user's question
- "answer": the expected answer
- "expected_keywords": list of keywords expected in the answer
- "tool": "{tool}"
- "category": "{category}"

Example format:
[
  {{"question": "...", "answer": "...", "expected_keywords": ["..."], "tool": "{tool}", "category": "{category}"}}
]

Generate exactly {count} examples as a JSON array:"""

    def _create_routing_prompt(self, seed_example: dict[str, Any], count: int) -> str:
        """Create prompt for generating routing example variations."""
        needs_tools = seed_example.get("needs_tools", False)
        category = seed_example.get("category", "greeting")

        return f"""Generate {count} variations of this query routing training example.

Original example:
- Query: {seed_example["query"]}
- Needs tools: {needs_tools}
- Reasoning: {seed_example.get("reasoning", "")}
- Direct answer: {seed_example.get("direct_answer", "")}
- Category: {category}

Requirements:
1. Generate {count} NEW examples with the SAME needs_tools value ({needs_tools})
2. Vary the phrasing and context
3. Keep the same general category ({category})
4. If needs_tools is false, provide a direct_answer

Respond with ONLY a valid JSON array of objects. Each object must have:
- "query": the user's query
- "needs_tools": {str(needs_tools).lower()}
- "reasoning": explanation for the decision
- "direct_answer": answer if no tools needed (empty string if needs_tools is true)
- "category": "{category}"

Generate exactly {count} examples as a JSON array:"""

    def _parse_json_response(
        self, content: str, example_type: str
    ) -> list[dict[str, Any]]:
        """Parse JSON array from LLM response."""
        # Try to extract JSON array from response
        content = content.strip()

        # Find JSON array in response
        json_match = re.search(r"\[[\s\S]*\]", content)
        if not json_match:
            logger.warning("No JSON array found in response")
            return []

        try:
            examples = json.loads(json_match.group())
            if not isinstance(examples, list):
                return []

            # Validate each example
            valid_examples = []
            for ex in examples:
                if self._validate_example(ex, example_type):
                    valid_examples.append(ex)
                else:
                    logger.warning(f"Invalid example skipped: {ex}")

            return valid_examples
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON: {e}")
            return []

    def _validate_example(self, example: dict[str, Any], example_type: str) -> bool:
        """Validate that an example has required fields."""
        if example_type == "routing":
            required = ["query", "needs_tools", "reasoning"]
            return all(k in example for k in required)
        else:
            required = ["question", "answer"]
            return all(k in example for k in required)

    async def expand_file(
        self,
        input_file: str,
        output_file: str,
        target_count: int,
        example_type: str = "orchestration",
    ) -> int:
        """Expand examples in a file to reach target count."""
        input_path = self.examples_dir / input_file
        output_path = self.examples_dir / output_file

        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return 0

        # Load existing examples
        examples = []
        with open(input_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))

        original_count = len(examples)
        logger.info(f"Loaded {original_count} examples from {input_file}")

        if original_count >= target_count:
            logger.info(f"Already have {original_count} >= {target_count} examples")
            return original_count

        # Calculate how many more we need
        needed = target_count - original_count
        variations_per_seed = max(1, needed // original_count + 1)

        logger.info(f"Generating ~{variations_per_seed} variations per seed example")

        new_examples = []
        for seed in examples:
            if len(examples) + len(new_examples) >= target_count:
                break

            variations = await self.generate_variations(
                seed, count=variations_per_seed, example_type=example_type
            )
            new_examples.extend(variations)
            logger.info(f"Generated {len(variations)} variations")

            # Small delay to avoid rate limiting
            await asyncio.sleep(0.5)

        # Combine and deduplicate
        all_examples = examples + new_examples

        # Simple deduplication by question/query
        seen = set()
        unique_examples = []
        key_field = "query" if example_type == "routing" else "question"

        for ex in all_examples:
            key = ex.get(key_field, "")
            if key and key not in seen:
                seen.add(key)
                unique_examples.append(ex)

        # Truncate to target
        unique_examples = unique_examples[:target_count]

        # Save expanded file
        with open(output_path, "w") as f:
            for ex in unique_examples:
                f.write(json.dumps(ex) + "\n")

        logger.info(f"Saved {len(unique_examples)} examples to {output_file}")
        return len(unique_examples)


async def main():
    """Main entry point."""
    expander = ExampleExpander()

    try:
        # Define expansion targets (file, output, target_count, type)
        expansions = [
            (
                "calculate_examples.jsonl",
                "calculate_examples.jsonl",
                45,
                "orchestration",
            ),
            ("python_examples.jsonl", "python_examples.jsonl", 45, "orchestration"),
            ("search_examples.jsonl", "search_examples.jsonl", 45, "orchestration"),
            ("delegate_reasoner.jsonl", "delegate_reasoner.jsonl", 30, "orchestration"),
            ("delegate_coder.jsonl", "delegate_coder.jsonl", 20, "orchestration"),
            ("delegate_fast.jsonl", "delegate_fast.jsonl", 10, "orchestration"),
            (
                "multi_tool_examples.jsonl",
                "multi_tool_examples.jsonl",
                60,
                "orchestration",
            ),
            ("routing_no_tools.jsonl", "routing_no_tools.jsonl", 45, "routing"),
        ]

        total = 0
        for input_file, output_file, target, example_type in expansions:
            count = await expander.expand_file(
                input_file, output_file, target, example_type
            )
            total += count

        print(f"\nTotal examples after expansion: {total}")

    finally:
        await expander.close()


if __name__ == "__main__":
    asyncio.run(main())
