"""
Observation buffers with hierarchical token budgeting.

Implements structured observation storage with per-category token budgets
following NVIDIA's ToolOrchestra reference architecture. Observations are
categorised by tool type and truncated with recency bias (oldest first).
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Rough approximation: 1 token ~ 4 characters.
CHARS_PER_TOKEN = 4


@dataclass
class TokenBudgets:
    """Configurable token budgets for each observation category."""

    attempts: int = 8000
    code: int = 12000
    delegates: int = 12000
    # Documents (search results) fill the remainder up to total.
    total: int = 27000


@dataclass
class _BufferEntry:
    """A single observation stored in a buffer."""

    step: int
    tool_name: str
    content: str


@dataclass
class ObservationBuffers:
    """
    Categorised observation buffers with token-aware truncation.

    Routes observations to the appropriate buffer based on tool name
    and supports hierarchical truncation with recency bias.
    """

    budgets: TokenBudgets = field(default_factory=TokenBudgets)
    max_observation_chars: int = 2048 * CHARS_PER_TOKEN  # Per-observation cap

    # Internal buffers
    _doc_list: list[_BufferEntry] = field(default_factory=list)
    _code_list: list[_BufferEntry] = field(default_factory=list)
    _delegate_list: list[_BufferEntry] = field(default_factory=list)
    _attempt_list: list[_BufferEntry] = field(default_factory=list)

    # Tool name -> buffer routing
    _CODE_TOOLS = frozenset({"python_execute", "calculate"})
    _DOC_TOOLS = frozenset({"web_search"})

    def add_observation(self, tool_name: str, content: str, step_number: int) -> None:
        """
        Add an observation to the appropriate buffer.

        Args:
            tool_name: The tool that produced this observation.
            content: The observation text.
            step_number: The orchestration step number.
        """
        # Cap per-observation size
        if len(content) > self.max_observation_chars:
            content = content[: self.max_observation_chars] + "\n[...truncated]"

        entry = _BufferEntry(step=step_number, tool_name=tool_name, content=content)

        if tool_name in self._DOC_TOOLS:
            self._doc_list.append(entry)
        elif tool_name in self._CODE_TOOLS:
            self._code_list.append(entry)
        elif tool_name.startswith("ask_"):
            self._delegate_list.append(entry)
        else:
            # Unknown tools go to attempts
            self._attempt_list.append(entry)

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
