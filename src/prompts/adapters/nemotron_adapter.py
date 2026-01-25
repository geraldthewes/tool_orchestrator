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

    def _get_context_info(self, signature: dspy.Signature) -> str:
        """Get context string for logging (model name, signature name)."""
        parts = []
        if dspy.settings.lm:
            model = getattr(dspy.settings.lm, "model", None)
            if model:
                parts.append(f"model={model}")
        sig_name = getattr(signature, "__name__", type(signature).__name__)
        parts.append(f"signature={sig_name}")
        return f" ({', '.join(parts)})" if parts else ""

    def parse(
        self,
        signature: dspy.Signature,
        completion: str,
    ) -> dict[str, Any]:
        """
        Parse LM completion, handling Nemotron's "final" wrapper.

        Args:
            signature: The DSPy signature defining expected output fields
            completion: Raw LM response string

        Returns:
            Dictionary of parsed output fields
        """
        logger.debug(
            f"Parsing completion for fields {list(signature.output_fields.keys())}: "
            f"{completion[:500]}{'...' if len(completion) > 500 else ''}"
        )

        # First try standard parsing
        try:
            fields = super().parse(signature, completion)
            if fields:
                logger.debug(f"Standard parsing succeeded: {list(fields.keys())}")
                return fields
        except Exception as e:
            logger.info(
                f"Standard JSONAdapter parsing failed, trying fallback: {type(e).__name__}: {e}"
            )

        # If standard parsing returned empty or failed, check for "final" wrapper
        fields = self._parse_with_final_unwrap(signature, completion)
        if fields:
            logger.debug("Successfully parsed after unwrapping 'final' object")
            return fields

        # Try raw tool call format ({"id": "...", "args": {...}})
        fields = self._parse_raw_tool_call(signature, completion)
        if fields:
            logger.debug("Successfully parsed raw tool call format")
            return fields

        # Last resort: return empty dict (will trigger DSPy's error handling)
        context = self._get_context_info(signature)
        logger.info(
            f"Failed to parse Nemotron response{context}. "
            f"Expected fields: {list(signature.output_fields.keys())}. "
            f"Completion was: {completion[:1000]}{'...' if len(completion) > 1000 else ''}"
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
            if "final" in data:
                final_value = data["final"]

                # Handle case where "final" is a stringified JSON (double-encoded)
                if isinstance(final_value, str):
                    try:
                        final_value = json_repair.loads(final_value)
                        logger.debug("Parsed stringified 'final' value as JSON")
                    except Exception:
                        logger.debug("Could not parse 'final' string as JSON")
                        final_value = None

                if isinstance(final_value, dict):
                    logger.debug(
                        f"Unwrapped 'final' object with keys: {list(final_value.keys())}"
                    )

                    # Filter to expected output fields
                    output_field_names = set(signature.output_fields.keys())
                    fields = {
                        k: v for k, v in final_value.items() if k in output_field_names
                    }

                    if fields:
                        return fields

            # Also check if fields are directly in data (no wrapper)
            output_field_names = set(signature.output_fields.keys())
            fields = {k: v for k, v in data.items() if k in output_field_names}

            return fields

        except Exception as e:
            context = self._get_context_info(signature)
            logger.warning(
                f"Failed to parse with final unwrap: {type(e).__name__}: {e}{context}"
            )
            return {}

    def _parse_raw_tool_call(
        self,
        signature: dspy.Signature,
        completion: str,
    ) -> dict[str, Any]:
        """
        Attempt to parse raw tool call format.

        Handles model output like:
            {"id": "web_search", "args": {"query": "..."}, "timeout": 10000}

        Maps to DSPy ReAct fields:
            {"next_tool_name": "web_search", "next_tool_args": '{"query": "..."}'}

        Args:
            signature: The DSPy signature defining expected output fields
            completion: Raw LM response string

        Returns:
            Dictionary of parsed fields, or empty dict if parsing fails
        """
        try:
            import json

            import json_repair

            json_str = self._extract_json(completion)
            if not json_str:
                return {}

            data = json_repair.loads(json_str)
            if not isinstance(data, dict):
                return {}

            # Check for raw tool call format (has "id" and "args" keys)
            if "id" not in data or "args" not in data:
                return {}

            output_field_names = set(signature.output_fields.keys())

            # Only apply mapping if signature expects ReAct action fields
            if "next_tool_name" not in output_field_names:
                return {}

            fields = {}

            # Map id -> next_tool_name
            if "next_tool_name" in output_field_names:
                fields["next_tool_name"] = data["id"]

            # Map args -> next_tool_args (stringify if dict)
            if "next_tool_args" in output_field_names:
                args = data["args"]
                if isinstance(args, dict):
                    fields["next_tool_args"] = json.dumps(args)
                else:
                    fields["next_tool_args"] = str(args)

            # Generate a default thought if expected
            if "next_thought" in output_field_names:
                fields["next_thought"] = f"Using {data['id']} tool"

            logger.debug(f"Parsed raw tool call format: {list(fields.keys())}")
            return fields

        except Exception as e:
            logger.debug(f"Raw tool call parsing failed: {type(e).__name__}: {e}")
            return {}

    def _extract_json(self, text: str) -> str | None:
        """
        Extract JSON object from text.

        Handles cases where JSON is embedded in markdown code blocks,
        surrounded by other text, or follows a <think>...</think> block
        (common with reasoning models like Nemotron-Orchestrator).

        Args:
            text: Raw text that may contain JSON

        Returns:
            Extracted JSON string, or None if not found
        """
        # Handle <think>...</think> blocks - extract content after </think>
        think_end = text.find("</think>")
        if think_end != -1:
            text = text[think_end + len("</think>") :].strip()
            logger.debug("Extracted content after </think> block")

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
            logger.debug(f"Could not find matching closing brace in: {text[:200]}...")
            return None

        extracted = text[start : end + 1]
        logger.debug(f"Extracted JSON ({len(extracted)} chars)")
        return extracted
