"""Tests for the ContentStore module."""

from src.orchestration.content_store import ContentStore, StoredContent


class TestStoredContent:
    """Tests for StoredContent dataclass."""

    def test_char_count_computed(self):
        """char_count should be computed from full_text length."""
        stored = StoredContent(
            content_id="ctx_test123",
            full_text="Hello, world!",
            tool_name="ask_reasoner",
            step_number=1,
            summary="A greeting",
        )
        assert stored.char_count == 13

    def test_empty_content(self):
        """Empty content should have char_count of 0."""
        stored = StoredContent(
            content_id="ctx_empty",
            full_text="",
            tool_name="ask_fast",
            step_number=2,
            summary="Empty response",
        )
        assert stored.char_count == 0


class TestContentStore:
    """Tests for ContentStore."""

    def test_empty_store(self):
        """New store should be empty."""
        store = ContentStore()
        assert len(store) == 0
        assert store.list_content_ids() == []

    def test_store_returns_content_id(self):
        """store() should return a unique content ID."""
        store = ContentStore()
        content_id = store.store(
            content="Test content",
            tool_name="ask_reasoner",
            step_number=1,
            summary="A test",
        )
        assert content_id.startswith("ctx_")
        assert len(content_id) == 16  # ctx_ + 12 hex chars

    def test_store_increments_length(self):
        """Storing content should increase store length."""
        store = ContentStore()
        store.store("Content 1", "ask_fast", 1, "Summary 1")
        assert len(store) == 1
        store.store("Content 2", "ask_coder", 2, "Summary 2")
        assert len(store) == 2

    def test_retrieve_returns_content(self):
        """retrieve() should return the stored content."""
        store = ContentStore()
        content_id = store.store(
            content="The full content here",
            tool_name="ask_reasoner",
            step_number=1,
            summary="A summary",
        )
        retrieved = store.retrieve(content_id)
        assert retrieved == "The full content here"

    def test_retrieve_unknown_id_returns_none(self):
        """retrieve() should return None for unknown IDs."""
        store = ContentStore()
        result = store.retrieve("ctx_nonexistent")
        assert result is None

    def test_retrieve_with_offset(self):
        """retrieve() should support offset parameter."""
        store = ContentStore()
        content_id = store.store(
            content="0123456789ABCDEF",
            tool_name="ask_fast",
            step_number=1,
            summary="Numbers",
        )
        # Retrieve from offset 10
        retrieved = store.retrieve(content_id, offset=10, limit=100)
        assert "ABCDEF" in retrieved
        assert "[Showing chars 10-16 of 16]" in retrieved

    def test_retrieve_with_limit(self):
        """retrieve() should support limit parameter."""
        store = ContentStore()
        long_content = "x" * 10000
        content_id = store.store(
            content=long_content,
            tool_name="ask_reasoner",
            step_number=1,
            summary="Long",
        )
        retrieved = store.retrieve(content_id, offset=0, limit=100)
        # Should have header + content
        assert "[Showing chars 0-100 of 10000]" in retrieved
        assert "[9900 chars remaining]" in retrieved

    def test_retrieve_pagination_header(self):
        """Pagination header should show correct offsets."""
        store = ContentStore()
        content_id = store.store(
            content="A" * 1000,
            tool_name="ask_fast",
            step_number=1,
            summary="Letters",
        )
        # Middle chunk
        retrieved = store.retrieve(content_id, offset=200, limit=300)
        assert "[Showing chars 200-500 of 1000]" in retrieved
        assert "[500 chars remaining]" in retrieved

    def test_retrieve_full_content_no_header(self):
        """Full content retrieval should not have pagination header."""
        store = ContentStore()
        content_id = store.store(
            content="Short content",
            tool_name="ask_fast",
            step_number=1,
            summary="Short",
        )
        retrieved = store.retrieve(content_id)
        assert "[Showing" not in retrieved
        assert retrieved == "Short content"

    def test_retrieve_offset_beyond_length(self):
        """Offset beyond content length should return empty string."""
        store = ContentStore()
        content_id = store.store(
            content="Short",
            tool_name="ask_fast",
            step_number=1,
            summary="Short",
        )
        retrieved = store.retrieve(content_id, offset=100)
        assert retrieved == ""

    def test_retrieve_negative_offset_clamped(self):
        """Negative offset should be clamped to 0."""
        store = ContentStore()
        content_id = store.store(
            content="Content here",
            tool_name="ask_fast",
            step_number=1,
            summary="Content",
        )
        retrieved = store.retrieve(content_id, offset=-10)
        assert "Content here" in retrieved

    def test_contains(self):
        """__contains__ should check for content ID existence."""
        store = ContentStore()
        content_id = store.store("Test", "ask_fast", 1, "Test")
        assert content_id in store
        assert "ctx_nonexistent" not in store

    def test_get_metadata(self):
        """get_metadata() should return content metadata."""
        store = ContentStore()
        content_id = store.store(
            content="Test content",
            tool_name="ask_reasoner",
            step_number=3,
            summary="A test summary",
        )
        metadata = store.get_metadata(content_id)
        assert metadata is not None
        assert metadata["content_id"] == content_id
        assert metadata["tool_name"] == "ask_reasoner"
        assert metadata["step_number"] == 3
        assert metadata["char_count"] == 12
        assert metadata["summary"] == "A test summary"

    def test_get_metadata_unknown_id(self):
        """get_metadata() should return None for unknown IDs."""
        store = ContentStore()
        assert store.get_metadata("ctx_unknown") is None

    def test_list_content_ids(self):
        """list_content_ids() should return all stored IDs."""
        store = ContentStore()
        id1 = store.store("Content 1", "ask_fast", 1, "Summary 1")
        id2 = store.store("Content 2", "ask_coder", 2, "Summary 2")
        ids = store.list_content_ids()
        assert id1 in ids
        assert id2 in ids
        assert len(ids) == 2

    def test_clear(self):
        """clear() should remove all stored content."""
        store = ContentStore()
        store.store("Content 1", "ask_fast", 1, "Summary 1")
        store.store("Content 2", "ask_coder", 2, "Summary 2")
        assert len(store) == 2
        store.clear()
        assert len(store) == 0
        assert store.list_content_ids() == []

    def test_unique_ids_per_store(self):
        """Each store call should generate a unique ID."""
        store = ContentStore()
        ids = set()
        for i in range(100):
            content_id = store.store(f"Content {i}", "ask_fast", i, f"Summary {i}")
            ids.add(content_id)
        assert len(ids) == 100
