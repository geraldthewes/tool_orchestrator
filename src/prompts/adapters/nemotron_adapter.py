"""
Custom JSON Adapter for Nemotron-Orchestrator models.

Handles the "final" wrapper that Nemotron-Orchestrator-8B produces
when generating final answers in ReAct-style orchestration.
"""

import logging
import re
from typing import Any

import dspy
from dspy.adapters import JSONAdapter

logger = logging.getLogger(__name__)


class NemotronJSONAdapter(JSONAdapter):
    """
    JSONAdapter subclass that handles Nemotron's "final" wrapper format.

    Nemotron-Orchestrator-8B wraps final answers as:
        {"final": {"reasoning": "...", "answer": "..."}}

    But DSPy expects fields at the root level:
        {"reasoning": "...", "answer": "..."}

    This adapter unwraps the "final" object when present.
    """

    def parse(
        self,
        signature: dspy.Signature,
        completion: str,
        _parse_field: Any = None,
    ) -> dict[str, Any]:
        """
        Parse LM completion, handling Nemotron's "final" wrapper.

        Args:
            signature: The DSPy signature defining expected output fields
            completion: Raw LM response string
            _parse_field: Optional field parser (passed to parent)

        Returns:
            Dictionary of parsed output fields
        """
        # First try standard parsing
        try:
            fields = super().parse(signature, completion, _parse_field)
            if fields:
                return fields
        except Exception as e:
            logger.debug(f"Standard parsing failed: {e}")

        # If standard parsing returned empty or failed, check for "final" wrapper
        fields = self._parse_with_final_unwrap(signature, completion)
        if fields:
            logger.debug("Successfully parsed after unwrapping 'final' object")
            return fields

        # Last resort: return empty dict (will trigger DSPy's error handling)
        logger.warning(
            f"Failed to parse Nemotron response. "
            f"Expected fields: {list(signature.output_fields.keys())}"
        )
        return {}

    def _parse_with_final_unwrap(
        self,
        signature: dspy.Signature,
        completion: str,
    ) -> dict[str, Any]:
        """
        Attempt to parse by unwrapping "final" object.

        Args:
            signature: The DSPy signature defining expected output fields
            completion: Raw LM response string

        Returns:
            Dictionary of parsed fields, or empty dict if parsing fails
        """
        try:
            import json_repair

            # Extract JSON from completion
            json_str = self._extract_json(completion)
            if not json_str:
                return {}

            data = json_repair.loads(json_str)

            if not isinstance(data, dict):
                return {}

            # Check for "final" wrapper
            if "final" in data and isinstance(data["final"], dict):
                unwrapped = data["final"]
                logger.debug(
                    f"Unwrapped 'final' object with keys: {list(unwrapped.keys())}"
                )

                # Filter to expected output fields
                output_field_names = set(signature.output_fields.keys())
                fields = {k: v for k, v in unwrapped.items() if k in output_field_names}

                if fields:
                    return fields

            # Also check if fields are directly in data (no wrapper)
            output_field_names = set(signature.output_fields.keys())
            fields = {k: v for k, v in data.items() if k in output_field_names}

            return fields

        except Exception as e:
            logger.debug(f"Failed to parse with final unwrap: {e}")
            return {}

    def _extract_json(self, text: str) -> str | None:
        """
        Extract JSON object from text.

        Handles cases where JSON is embedded in markdown code blocks
        or surrounded by other text.

        Args:
            text: Raw text that may contain JSON

        Returns:
            Extracted JSON string, or None if not found
        """
        # Try to find JSON in code blocks first
        code_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
        matches = re.findall(code_block_pattern, text)
        for match in matches:
            if "{" in match:
                return match.strip()

        # Try to find raw JSON object
        # Find first { and last }
        start = text.find("{")
        if start == -1:
            return None

        # Find matching closing brace
        depth = 0
        end = -1
        for i, char in enumerate(text[start:], start):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break

        if end == -1:
            return None

        return text[start : end + 1]
