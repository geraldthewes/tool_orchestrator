"""
Content summarizer for context externalization.

Extracts or generates summaries of delegate responses to include in the
observation buffer when full content is externalized.
"""

import logging
import re
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Target summary length in characters (roughly 200-400 tokens)
TARGET_SUMMARY_CHARS = 1200
MAX_SUMMARY_CHARS = 1600

# Marker that delegates can use to embed their own summary
SUMMARY_MARKER = "## Summary"


class Summarizer:
    """
    Extracts or generates summaries for delegate responses.

    The summarizer first attempts to extract an existing summary if the
    delegate has embedded one using the ## Summary marker. If no embedded
    summary is found, it calls a fast delegate to generate one.

    This allows delegates to provide their own high-quality summaries while
    still having a fallback for delegates that don't.
    """

    def __init__(
        self,
        fast_delegate_caller: Optional[Callable[[str], Optional[str]]] = None,
        target_chars: int = TARGET_SUMMARY_CHARS,
        max_chars: int = MAX_SUMMARY_CHARS,
    ) -> None:
        """
        Initialize the summarizer.

        Args:
            fast_delegate_caller: Optional callable that takes (prompt: str)
                and returns a response string. Used for generating summaries
                when none is embedded. If None, falls back to truncation.
            target_chars: Target summary length in characters.
            max_chars: Maximum summary length in characters.
        """
        self._fast_delegate = fast_delegate_caller
        self._target_chars = target_chars
        self._max_chars = max_chars

    def summarize(self, content: str, tool_name: str) -> str:
        """
        Generate or extract a summary of the content.

        Args:
            content: The full content to summarize.
            tool_name: Name of the tool that produced the content.

        Returns:
            A summary suitable for the observation buffer.
        """
        # First try to extract an embedded summary
        extracted = self._extract_embedded_summary(content)
        if extracted:
            logger.debug(
                "Extracted embedded summary (%d chars) from %s",
                len(extracted),
                tool_name,
            )
            return self._ensure_length(extracted)

        # Try to generate a summary via fast delegate
        if self._fast_delegate:
            generated = self._generate_summary(content, tool_name)
            if generated:
                logger.debug(
                    "Generated summary (%d chars) for %s",
                    len(generated),
                    tool_name,
                )
                return self._ensure_length(generated)

        # Fallback: extract key points or truncate
        logger.debug("Using fallback summary extraction for %s", tool_name)
        return self._fallback_summary(content)

    def _extract_embedded_summary(self, content: str) -> Optional[str]:
        """
        Extract a summary embedded by the delegate.

        Looks for content after a ## Summary header, up to the next
        heading or the end of content.

        Args:
            content: The full delegate response.

        Returns:
            Extracted summary text, or None if not found.
        """
        # Case-insensitive search for ## Summary heading
        pattern = r"(?:^|\n)##\s*Summary\s*\n(.*?)(?=\n##\s|\Z)"
        match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)

        if match:
            summary = match.group(1).strip()
            if summary:
                return summary

        return None

    def _generate_summary(self, content: str, tool_name: str) -> Optional[str]:
        """
        Generate a summary using the fast delegate.

        Args:
            content: The full content to summarize.
            tool_name: Name of the tool (for context in the prompt).

        Returns:
            Generated summary, or None on failure.
        """
        if not self._fast_delegate:
            return None

        # Truncate content if too long for the summarization request
        max_content_for_summary = 16000
        if len(content) > max_content_for_summary:
            content = content[:max_content_for_summary] + "\n\n[Content truncated...]"

        prompt = f"""Summarize the following {tool_name} response in 2-4 sentences.
Focus on the key findings, conclusions, or actionable information.
Be concise but preserve important details.

---
{content}
---

Summary:"""

        try:
            response = self._fast_delegate(prompt)
            if response and len(response.strip()) > 20:
                return response.strip()
        except Exception as e:
            logger.warning("Summary generation failed: %s", e)

        return None

    def _fallback_summary(self, content: str) -> str:
        """
        Create a fallback summary by extracting key sentences.

        Uses heuristics to find the most informative parts of the content:
        1. First non-empty paragraph (often contains the main point)
        2. Sentences containing conclusion/result keywords
        3. Truncated beginning if nothing else works

        Args:
            content: The full content to summarize.

        Returns:
            A fallback summary.
        """
        lines = content.strip().split("\n")

        # Try to find a good opening paragraph
        opening_lines: list[str] = []
        char_count = 0

        for line in lines:
            stripped = line.strip()
            # Skip headers, empty lines, and code blocks
            if not stripped or stripped.startswith("#") or stripped.startswith("```"):
                if opening_lines:
                    break  # End of first paragraph
                continue

            opening_lines.append(stripped)
            char_count += len(stripped)

            if char_count >= self._target_chars // 2:
                break

        if opening_lines:
            opening = " ".join(opening_lines)
            if len(opening) >= 100:
                return self._ensure_length(opening)

        # Fallback to simple truncation with ellipsis
        clean_content = " ".join(line.strip() for line in lines if line.strip())
        if len(clean_content) <= self._max_chars:
            return clean_content

        # Truncate at word boundary
        truncated = clean_content[: self._max_chars]
        last_space = truncated.rfind(" ")
        if last_space > self._max_chars // 2:
            truncated = truncated[:last_space]

        return truncated + "..."

    def _ensure_length(self, summary: str) -> str:
        """Ensure summary is within length limits."""
        if len(summary) <= self._max_chars:
            return summary

        # Truncate at word boundary
        truncated = summary[: self._max_chars]
        last_space = truncated.rfind(" ")
        if last_space > self._max_chars // 2:
            truncated = truncated[:last_space]

        return truncated + "..."
