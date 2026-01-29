"""Tests for the Summarizer module."""

from src.orchestration.summarizer import (
    Summarizer,
    SUMMARY_MARKER,
    TARGET_SUMMARY_CHARS,
    MAX_SUMMARY_CHARS,
)


class TestSummarizerWithEmbeddedSummary:
    """Tests for extracting embedded summaries."""

    def test_extract_simple_summary(self):
        """Should extract summary after ## Summary header."""
        content = """# Analysis

This is a detailed analysis of the problem.

## Summary

The key finding is that X leads to Y.

## Details

More details here...
"""
        summarizer = Summarizer()
        summary = summarizer.summarize(content, "ask_reasoner")
        assert "The key finding is that X leads to Y." in summary

    def test_extract_summary_case_insensitive(self):
        """Summary extraction should be case insensitive."""
        content = """## SUMMARY

This is the summary content.
"""
        summarizer = Summarizer()
        summary = summarizer.summarize(content, "ask_fast")
        assert "This is the summary content." in summary

    def test_extract_summary_at_end(self):
        """Should extract summary at end of content."""
        content = """# Report

Detailed report content here.

## Summary

Final conclusions are important.
"""
        summarizer = Summarizer()
        summary = summarizer.summarize(content, "ask_reasoner")
        assert "Final conclusions are important." in summary

    def test_no_summary_marker(self):
        """Without summary marker, should use fallback."""
        content = "This is just plain content without any summary section."
        summarizer = Summarizer()
        summary = summarizer.summarize(content, "ask_fast")
        # Should use fallback (content is short, so returned as-is)
        assert "plain content" in summary

    def test_empty_summary_section(self):
        """Empty summary section should trigger fallback."""
        content = """## Summary

## Next Section

Content here.
"""
        summarizer = Summarizer()
        summary = summarizer.summarize(content, "ask_fast")
        # Should fall back since summary section is empty
        assert summary  # Should return something

    def test_multiline_summary(self):
        """Should extract multiline summaries."""
        content = """## Summary

First point of the summary.
Second point of the summary.
Third point continues here.

## Details
"""
        summarizer = Summarizer()
        summary = summarizer.summarize(content, "ask_reasoner")
        assert "First point" in summary
        assert "Second point" in summary


class TestSummarizerWithFastDelegate:
    """Tests for summary generation via fast delegate."""

    def test_uses_fast_delegate_when_no_embedded_summary(self):
        """Should call fast delegate when no embedded summary."""
        delegate_called = []

        def mock_delegate(prompt: str) -> str:
            delegate_called.append(prompt)
            return "Generated summary from delegate."

        content = "Long content without any summary marker."
        summarizer = Summarizer(fast_delegate_caller=mock_delegate)
        summary = summarizer.summarize(content, "ask_coder")

        assert len(delegate_called) == 1
        assert "Generated summary from delegate." in summary

    def test_prefers_embedded_summary_over_delegate(self):
        """Should use embedded summary even when delegate is available."""
        delegate_called = []

        def mock_delegate(prompt: str) -> str:
            delegate_called.append(prompt)
            return "Delegate response"

        content = """## Summary

Embedded summary here.

## Details

Long details...
"""
        summarizer = Summarizer(fast_delegate_caller=mock_delegate)
        summary = summarizer.summarize(content, "ask_reasoner")

        assert len(delegate_called) == 0
        assert "Embedded summary here." in summary

    def test_delegate_failure_uses_fallback(self):
        """Delegate failure should trigger fallback summarization."""

        def failing_delegate(prompt: str) -> str:
            raise Exception("Connection failed")

        content = "Plain content here that needs summarization."
        summarizer = Summarizer(fast_delegate_caller=failing_delegate)
        summary = summarizer.summarize(content, "ask_fast")

        # Should still return something via fallback
        assert summary
        assert "Plain content" in summary

    def test_delegate_returns_none_uses_fallback(self):
        """Delegate returning None should trigger fallback."""

        def none_delegate(prompt: str) -> None:
            return None

        content = "Content that needs summarization without delegate help."
        summarizer = Summarizer(fast_delegate_caller=none_delegate)
        summary = summarizer.summarize(content, "ask_fast")

        assert summary

    def test_delegate_returns_short_response_uses_fallback(self):
        """Very short delegate response should trigger fallback."""

        def short_delegate(prompt: str) -> str:
            return "OK"  # Too short to be useful

        content = "Content that needs a proper summary."
        summarizer = Summarizer(fast_delegate_caller=short_delegate)
        summary = summarizer.summarize(content, "ask_fast")

        # Should fall back since "OK" is too short
        assert len(summary) > 5


class TestSummarizerFallback:
    """Tests for fallback summarization."""

    def test_short_content_returned_as_is(self):
        """Short content should be returned without truncation."""
        content = "This is a short piece of content."
        summarizer = Summarizer()
        summary = summarizer.summarize(content, "ask_fast")
        assert summary == content

    def test_long_content_truncated(self):
        """Long content should be truncated with ellipsis."""
        content = "Word " * 1000  # ~5000 chars
        summarizer = Summarizer(max_chars=200)
        summary = summarizer.summarize(content, "ask_fast")
        assert len(summary) <= 203  # max_chars + "..."
        assert summary.endswith("...")

    def test_extracts_first_paragraph(self):
        """Should extract the first meaningful paragraph."""
        content = """# Title

This is the first paragraph that explains the main concept.
It continues for a bit more detail.

## Section 1

More content in section 1.
"""
        summarizer = Summarizer()
        summary = summarizer.summarize(content, "ask_reasoner")
        assert "first paragraph" in summary

    def test_skips_headers_and_code_blocks(self):
        """Should skip headers and code blocks in fallback."""
        content = """# Header

```python
code_block_here()
```

The actual content starts here.
"""
        summarizer = Summarizer()
        summary = summarizer.summarize(content, "ask_coder")
        assert "actual content" in summary

    def test_truncates_at_word_boundary(self):
        """Truncation should occur at word boundaries."""
        content = "Word1 Word2 Word3 Word4 Word5"
        summarizer = Summarizer(max_chars=15)
        summary = summarizer.summarize(content, "ask_fast")
        # Should not cut in the middle of a word
        assert not summary.rstrip("...").endswith("Wor")


class TestSummarizerLengthLimits:
    """Tests for summary length enforcement."""

    def test_summary_respects_max_chars(self):
        """Summary should not exceed max_chars."""
        content = "## Summary\n\n" + "x" * 5000
        summarizer = Summarizer(max_chars=1000)
        summary = summarizer.summarize(content, "ask_reasoner")
        assert len(summary) <= 1003  # max_chars + "..."

    def test_custom_target_chars(self):
        """Should use custom target_chars in fallback."""
        content = "Word " * 100
        summarizer = Summarizer(target_chars=50, max_chars=100)
        summary = summarizer.summarize(content, "ask_fast")
        # Fallback uses target_chars for opening, then truncates at max_chars
        assert len(summary) <= 103  # max_chars + "..."

    def test_very_long_embedded_summary_truncated(self):
        """Very long embedded summaries should be truncated."""
        long_summary = "Important point. " * 200
        content = f"## Summary\n\n{long_summary}\n\n## Details"
        summarizer = Summarizer(max_chars=500)
        summary = summarizer.summarize(content, "ask_reasoner")
        assert len(summary) <= 503


class TestSummarizerConstants:
    """Tests for module constants."""

    def test_summary_marker_format(self):
        """SUMMARY_MARKER should be the expected header format."""
        assert SUMMARY_MARKER == "## Summary"

    def test_target_chars_reasonable(self):
        """TARGET_SUMMARY_CHARS should be reasonable (~300 tokens)."""
        assert 800 <= TARGET_SUMMARY_CHARS <= 1600

    def test_max_chars_larger_than_target(self):
        """MAX_SUMMARY_CHARS should be larger than target."""
        assert MAX_SUMMARY_CHARS > TARGET_SUMMARY_CHARS
