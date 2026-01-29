"""Tests for observation buffers and token budgeting."""

from src.orchestration.buffers import (
    ObservationBuffers,
    TokenBudgets,
    _BufferEntry,
    CHARS_PER_TOKEN,
)


class TestTokenBudgets:
    """Tests for TokenBudgets dataclass."""

    def test_default_budgets(self):
        """Default budgets should have sensible values."""
        budgets = TokenBudgets()
        assert budgets.attempts == 8000
        assert budgets.code == 12000
        assert budgets.delegates == 8000  # Lower budget, externalization handles overflow
        assert budgets.total == 27000

    def test_custom_budgets(self):
        """Custom budgets should override defaults."""
        budgets = TokenBudgets(attempts=1000, code=2000, total=5000)
        assert budgets.attempts == 1000
        assert budgets.code == 2000
        assert budgets.total == 5000


class TestObservationBuffers:
    """Tests for ObservationBuffers."""

    def test_empty_buffers(self):
        """Empty buffers should report no observations."""
        buffers = ObservationBuffers()
        assert not buffers.has_observations
        assert buffers.build_context_string() == ""

    def test_add_search_observation(self):
        """Search observations should go to doc buffer."""
        buffers = ObservationBuffers()
        buffers.add_observation("web_search", "Search results here", 1)
        assert buffers.has_observations
        assert len(buffers._doc_list) == 1
        assert buffers._doc_list[0].tool_name == "web_search"

    def test_add_code_observation(self):
        """Code/calculate observations should go to code buffer."""
        buffers = ObservationBuffers()
        buffers.add_observation("python_execute", "Output: 42", 1)
        buffers.add_observation("calculate", "2 + 2 = 4", 2)
        assert len(buffers._code_list) == 2

    def test_add_delegate_observation(self):
        """Delegate observations should go to delegate buffer."""
        buffers = ObservationBuffers()
        buffers.add_observation("ask_reasoner", "Reasoner response", 1)
        assert len(buffers._delegate_list) == 1

    def test_add_unknown_tool_observation(self):
        """Unknown tools should go to attempt buffer."""
        buffers = ObservationBuffers()
        buffers.add_observation("unknown_tool", "Some result", 1)
        assert len(buffers._attempt_list) == 1

    def test_observation_truncation(self):
        """Long observations should be truncated to max_observation_chars."""
        buffers = ObservationBuffers(max_observation_chars=100)
        long_content = "x" * 200
        buffers.add_observation("calculate", long_content, 1)

        entry = buffers._code_list[0]
        assert len(entry.content) < 200
        assert entry.content.endswith("[...truncated]")

    def test_build_context_string_has_sections(self):
        """Context string should include categorized sections."""
        buffers = ObservationBuffers()
        buffers.add_observation("web_search", "Search: python docs", 1)
        buffers.add_observation("calculate", "2 + 2 = 4", 2)
        buffers.add_observation("ask_reasoner", "Reason: yes", 3)
        buffers.add_observation("unknown_tool", "Attempt 1", 4)

        context = buffers.build_context_string()
        assert "## Search Results" in context
        assert "## Code & Calculation Results" in context
        assert "## Delegate Responses" in context
        assert "## Previous Attempts" in context

    def test_build_context_includes_step_numbers(self):
        """Context string should include step numbers."""
        buffers = ObservationBuffers()
        buffers.add_observation("calculate", "Result: 42", 3)

        context = buffers.build_context_string()
        assert "[Step 3]" in context
        assert "calculate:" in context


class TestCutSeq:
    """Tests for the cut_seq truncation function."""

    def test_empty_list(self):
        """Empty list should return empty."""
        result = ObservationBuffers.cut_seq([], 1000)
        assert result == []

    def test_within_budget(self):
        """Entries within budget should be returned unchanged."""
        entries = [
            _BufferEntry(step=1, tool_name="test", content="short"),
            _BufferEntry(step=2, tool_name="test", content="also short"),
        ]
        result = ObservationBuffers.cut_seq(entries, 1000)
        assert len(result) == 2

    def test_exceeds_budget_removes_oldest(self):
        """When exceeding budget, oldest entries should be removed first."""
        entries = [
            _BufferEntry(step=1, tool_name="test", content="a" * 100),
            _BufferEntry(step=2, tool_name="test", content="b" * 100),
            _BufferEntry(step=3, tool_name="test", content="c" * 100),
        ]
        # Budget for ~1 entry (~25 tokens = 100 chars)
        result = ObservationBuffers.cut_seq(entries, 30)
        assert len(result) < 3
        # Most recent entry should be retained
        assert result[-1].step == 3

    def test_zero_budget_returns_empty(self):
        """Zero budget should return empty or minimal list."""
        entries = [
            _BufferEntry(step=1, tool_name="test", content="a" * 100),
        ]
        result = ObservationBuffers.cut_seq(entries, 0)
        assert len(result) == 0

    def test_preserves_order(self):
        """Remaining entries should maintain chronological order."""
        entries = [
            _BufferEntry(step=i, tool_name="test", content=f"entry_{i}")
            for i in range(5)
        ]
        result = ObservationBuffers.cut_seq(entries, 1000)
        steps = [e.step for e in result]
        assert steps == sorted(steps)


class TestTokenEstimation:
    """Tests for token estimation."""

    def test_estimate_tokens(self):
        """Token estimate should use CHARS_PER_TOKEN ratio."""
        text = "a" * 400
        tokens = ObservationBuffers._estimate_tokens(text)
        assert tokens == 400 // CHARS_PER_TOKEN

    def test_estimate_empty_string(self):
        """Empty string should estimate 0 tokens."""
        assert ObservationBuffers._estimate_tokens("") == 0
