"""
Observation buffers with hierarchical token budgeting.

Implements structured observation storage with per-category token budgets
following NVIDIA's ToolOrchestra reference architecture. Observations are
categorised by tool type and truncated with recency bias (oldest first).

Supports context externalization for delegate responses: large outputs are
stored externally and replaced with summaries + reference IDs, allowing the
orchestrator to selectively retrieve full content when needed.
"""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .content_store import ContentStore
    from .summarizer import Summarizer

logger = logging.getLogger(__name__)

# Rough approximation: 1 token ~ 4 characters.
CHARS_PER_TOKEN = 4


@dataclass
class TokenBudgets:
    """Configurable token budgets for each observation category."""

    attempts: int = 8000
    code: int = 12000
    delegates: int = 8000
    # Documents (search results) fill the remainder up to total.
    total: int = 27000


@dataclass
class _BufferEntry:
    """A single observation stored in a buffer."""

    step: int
    tool_name: str
    content: str
    is_externalized: bool = False
    content_id: Optional[str] = None


@dataclass
class ObservationBuffers:
    """
    Categorised observation buffers with token-aware truncation.

    Routes observations to the appropriate buffer based on tool name
    and supports hierarchical truncation with recency bias.

    For delegate responses, supports context externalization: when content
    exceeds the threshold, it is stored externally and replaced with a
    summary + reference ID for selective retrieval.
    """

    budgets: TokenBudgets = field(default_factory=TokenBudgets)
    max_observation_chars: int = 2048 * CHARS_PER_TOKEN  # Per-observation cap

    # Context externalization settings
    externalize_threshold: int = 4000  # Chars above which to externalize
    keep_recent_delegate_full: int = 1  # Keep last N delegate calls in full

    # Internal buffers
    _doc_list: list[_BufferEntry] = field(default_factory=list)
    _code_list: list[_BufferEntry] = field(default_factory=list)
    _delegate_list: list[_BufferEntry] = field(default_factory=list)
    _attempt_list: list[_BufferEntry] = field(default_factory=list)

    # External dependencies (set after init via set_externalization_deps)
    _content_store: Optional["ContentStore"] = field(default=None, repr=False)
    _summarizer: Optional["Summarizer"] = field(default=None, repr=False)

    # Tool name -> buffer routing
    _CODE_TOOLS = frozenset({"python_execute", "calculate"})
    _DOC_TOOLS = frozenset({"web_search"})

    def set_externalization_deps(
        self,
        content_store: "ContentStore",
        summarizer: "Summarizer",
    ) -> None:
        """
        Set dependencies for context externalization.

        Args:
            content_store: The ContentStore instance for external storage.
            summarizer: The Summarizer instance for generating summaries.
        """
        self._content_store = content_store
        self._summarizer = summarizer

    def add_observation(self, tool_name: str, content: str, step_number: int) -> None:
        """
        Add an observation to the appropriate buffer.

        Args:
            tool_name: The tool that produced this observation.
            content: The observation text.
            step_number: The orchestration step number.
        """
        # Route delegate observations through specialized handler
        if tool_name.startswith("ask_"):
            self.add_delegate_observation(tool_name, content, step_number)
            return

        # Cap per-observation size for non-delegate tools
        if len(content) > self.max_observation_chars:
            content = content[: self.max_observation_chars] + "\n[...truncated]"

        entry = _BufferEntry(step=step_number, tool_name=tool_name, content=content)

        if tool_name in self._DOC_TOOLS:
            self._doc_list.append(entry)
        elif tool_name in self._CODE_TOOLS:
            self._code_list.append(entry)
        else:
            # Unknown tools go to attempts
            self._attempt_list.append(entry)

    def add_delegate_observation(
        self,
        tool_name: str,
        content: str,
        step_number: int,
    ) -> Optional[str]:
        """
        Add a delegate observation with context externalization support.

        When externalization is enabled and content exceeds the threshold,
        the full content is stored externally and replaced with a summary
        + reference ID. The most recent N delegate calls (configured by
        keep_recent_delegate_full) are kept in full.

        Args:
            tool_name: The delegate tool name (e.g., "ask_reasoner").
            content: The full delegate response.
            step_number: The orchestration step number.

        Returns:
            The content_id if externalized, None otherwise.
        """
        content_len = len(content)
        should_externalize = (
            self._content_store is not None
            and self._summarizer is not None
            and content_len > self.externalize_threshold
        )

        if should_externalize:
            # Externalize older delegate calls if we exceed keep_recent
            self._externalize_older_delegates()

            # Generate summary for this content
            summary = self._summarizer.summarize(content, tool_name)

            # Store full content externally
            content_id = self._content_store.store(
                content=content,
                tool_name=tool_name,
                step_number=step_number,
                summary=summary,
            )

            # Create observation with summary + reference
            externalized_content = (
                f"{summary}\n\n"
                f"[Full response ({content_len} chars) stored as {content_id}. "
                f"Use retrieve_context(\"{content_id}\") to access.]"
            )

            entry = _BufferEntry(
                step=step_number,
                tool_name=tool_name,
                content=externalized_content,
                is_externalized=True,
                content_id=content_id,
            )
            self._delegate_list.append(entry)

            logger.debug(
                "Externalized %s response (%d chars) as %s",
                tool_name,
                content_len,
                content_id,
            )
            return content_id

        # Not externalizing: add with standard truncation
        if content_len > self.max_observation_chars:
            content = content[: self.max_observation_chars] + "\n[...truncated]"

        entry = _BufferEntry(
            step=step_number,
            tool_name=tool_name,
            content=content,
            is_externalized=False,
        )
        self._delegate_list.append(entry)
        return None

    def _externalize_older_delegates(self) -> None:
        """
        Externalize older non-externalized delegate entries.

        Keeps the most recent keep_recent_delegate_full entries in full,
        and externalizes any older full entries.
        """
        if self._content_store is None or self._summarizer is None:
            return

        # Find non-externalized entries
        full_entries = [e for e in self._delegate_list if not e.is_externalized]
        num_to_externalize = len(full_entries) - self.keep_recent_delegate_full

        if num_to_externalize <= 0:
            return

        # Externalize the oldest entries
        for entry in full_entries[:num_to_externalize]:
            summary = self._summarizer.summarize(entry.content, entry.tool_name)
            content_len = len(entry.content)

            content_id = self._content_store.store(
                content=entry.content,
                tool_name=entry.tool_name,
                step_number=entry.step,
                summary=summary,
            )

            # Update entry in place
            entry.content = (
                f"{summary}\n\n"
                f"[Full response ({content_len} chars) stored as {content_id}. "
                f"Use retrieve_context(\"{content_id}\") to access.]"
            )
            entry.is_externalized = True
            entry.content_id = content_id

            logger.debug(
                "Retroactively externalized %s (step %d) as %s",
                entry.tool_name,
                entry.step,
                content_id,
            )

    def build_context_string(self) -> str:
        """
        Build a context string from all buffers with hierarchical truncation.

        Applies per-category token budgets, removing oldest entries first
        when a category exceeds its budget. Documents fill the remaining
        budget after other categories.

        Returns:
            Formatted context string for inclusion in the user message.
        """
        sections: list[str] = []

        # Attempts
        attempt_text = self._format_entries(self._attempt_list, self.budgets.attempts)
        if attempt_text:
            sections.append(f"## Previous Attempts\n{attempt_text}")

        # Code outputs
        code_text = self._format_entries(self._code_list, self.budgets.code)
        if code_text:
            sections.append(f"## Code & Calculation Results\n{code_text}")

        # Delegate responses
        delegate_text = self._format_entries(
            self._delegate_list, self.budgets.delegates
        )
        if delegate_text:
            sections.append(f"## Delegate Responses\n{delegate_text}")

        # Documents fill remaining budget
        used_tokens = sum(self._estimate_tokens(s) for s in sections)
        remaining = max(0, self.budgets.total - used_tokens)
        doc_text = self._format_entries(self._doc_list, remaining)
        if doc_text:
            sections.append(f"## Search Results\n{doc_text}")

        return "\n\n".join(sections)

    @property
    def has_observations(self) -> bool:
        """Return True if any buffer has entries."""
        return bool(
            self._doc_list
            or self._code_list
            or self._delegate_list
            or self._attempt_list
        )

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token count estimate."""
        return len(text) // CHARS_PER_TOKEN

    @staticmethod
    def cut_seq(entries: list[_BufferEntry], max_tokens: int) -> list[_BufferEntry]:
        """
        Truncate a list of entries to fit within a token budget.

        Removes oldest entries first (recency bias) until the total
        estimated token count is within the budget.

        Args:
            entries: List of buffer entries (oldest first).
            max_tokens: Maximum token budget for this category.

        Returns:
            Truncated list of entries (most recent retained).
        """
        if not entries:
            return []

        max_chars = max_tokens * CHARS_PER_TOKEN
        total_chars = sum(len(e.content) for e in entries)

        if total_chars <= max_chars:
            return list(entries)

        # Remove from the front (oldest) until we fit
        result = list(entries)
        while result and total_chars > max_chars:
            removed = result.pop(0)
            total_chars -= len(removed.content)

        return result

    def _format_entries(self, entries: list[_BufferEntry], max_tokens: int) -> str:
        """Format buffer entries within a token budget."""
        truncated = self.cut_seq(entries, max_tokens)
        if not truncated:
            return ""

        lines: list[str] = []
        for entry in truncated:
            lines.append(f"[Step {entry.step}] {entry.tool_name}:\n{entry.content}")
        return "\n\n".join(lines)
