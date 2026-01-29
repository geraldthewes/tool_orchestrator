"""
External content storage for context externalization.

Implements RLM-inspired approach where large delegate outputs are stored
externally rather than truncated, allowing the orchestrator to selectively
retrieve content when needed via the retrieve_context tool.
"""

import logging
import uuid
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class StoredContent:
    """A single piece of externally stored content."""

    content_id: str
    full_text: str
    tool_name: str
    step_number: int
    summary: str
    char_count: int = field(init=False)

    def __post_init__(self) -> None:
        """Compute character count after initialization."""
        self.char_count = len(self.full_text)


class ContentStore:
    """
    In-memory store for externalized delegate content.

    When delegate responses exceed the externalization threshold, the full
    content is stored here and a summary + reference ID is returned to the
    observation buffer. The orchestrator can then call retrieve_context()
    to access the full content when needed.

    This approach preserves expert reasoning that would otherwise be lost
    to truncation, following the RLM paper's insight that long prompts
    should be externalized to the environment.
    """

    def __init__(self) -> None:
        """Initialize an empty content store."""
        self._storage: dict[str, StoredContent] = {}

    def store(
        self,
        content: str,
        tool_name: str,
        step_number: int,
        summary: str,
    ) -> str:
        """
        Store content externally and return a reference ID.

        Args:
            content: The full text content to store.
            tool_name: Name of the tool that produced this content.
            step_number: The orchestration step number.
            summary: A summary of the content for the observation buffer.

        Returns:
            A unique content ID (ctx_xxx format) for retrieval.
        """
        content_id = f"ctx_{uuid.uuid4().hex[:12]}"

        stored = StoredContent(
            content_id=content_id,
            full_text=content,
            tool_name=tool_name,
            step_number=step_number,
            summary=summary,
        )
        self._storage[content_id] = stored

        logger.debug(
            "Stored %d chars from %s (step %d) as %s",
            stored.char_count,
            tool_name,
            step_number,
            content_id,
        )

        return content_id

    def retrieve(
        self,
        content_id: str,
        offset: int = 0,
        limit: int = 4000,
    ) -> str | None:
        """
        Retrieve stored content by ID with pagination support.

        Args:
            content_id: The content ID returned from store().
            offset: Character offset to start from (default 0).
            limit: Maximum characters to return (default 4000).

        Returns:
            The requested content slice, or None if content_id not found.
        """
        stored = self._storage.get(content_id)
        if stored is None:
            logger.warning("Content ID not found: %s", content_id)
            return None

        full_text = stored.full_text
        total_len = len(full_text)

        # Clamp offset to valid range
        if offset < 0:
            offset = 0
        if offset >= total_len:
            logger.debug(
                "Offset %d beyond content length %d for %s",
                offset,
                total_len,
                content_id,
            )
            return ""

        # Extract the slice
        end = min(offset + limit, total_len)
        content_slice = full_text[offset:end]

        # Add metadata about pagination
        if offset > 0 or end < total_len:
            remaining = total_len - end
            header = f"[Showing chars {offset}-{end} of {total_len}]"
            if remaining > 0:
                header += f" [{remaining} chars remaining]"
            content_slice = f"{header}\n\n{content_slice}"

        logger.debug(
            "Retrieved %d chars from %s (offset=%d, limit=%d)",
            len(content_slice),
            content_id,
            offset,
            limit,
        )

        return content_slice

    def get_metadata(self, content_id: str) -> dict | None:
        """
        Get metadata about stored content without retrieving it.

        Args:
            content_id: The content ID to look up.

        Returns:
            Dictionary with metadata, or None if not found.
        """
        stored = self._storage.get(content_id)
        if stored is None:
            return None

        return {
            "content_id": stored.content_id,
            "tool_name": stored.tool_name,
            "step_number": stored.step_number,
            "char_count": stored.char_count,
            "summary": stored.summary,
        }

    def list_content_ids(self) -> list[str]:
        """Return all stored content IDs."""
        return list(self._storage.keys())

    def clear(self) -> None:
        """Clear all stored content."""
        count = len(self._storage)
        self._storage.clear()
        logger.debug("Cleared %d items from content store", count)

    def __len__(self) -> int:
        """Return the number of stored items."""
        return len(self._storage)

    def __contains__(self, content_id: str) -> bool:
        """Check if a content ID exists in the store."""
        return content_id in self._storage
