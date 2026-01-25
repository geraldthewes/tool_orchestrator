"""
Tests for DSPy LM Factory functions.

Tests the factory functions for creating DSPy LM instances.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.prompts.adapters.lm_factory import (
    get_teacher_lm,
    _estimate_tokens,
    _estimate_messages_tokens,
    TokenAwareLM,
)


class TestGetTeacherLM:
    """Tests for get_teacher_lm function."""

    def test_raises_error_when_base_url_missing(self):
        """Test that missing base URL raises ValueError."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="Teacher LLM base URL not configured"):
                get_teacher_lm(model="test-model")

    def test_raises_error_when_model_missing(self):
        """Test that missing model raises ValueError."""
        with patch.dict(
            "os.environ", {"TEACHER_BASE_URL": "http://example.com"}, clear=True
        ):
            with pytest.raises(ValueError, match="Teacher LLM model not configured"):
                get_teacher_lm()

    def test_uses_env_vars(self):
        """Test that environment variables are used when params not provided."""
        with patch.dict(
            "os.environ",
            {
                "TEACHER_BASE_URL": "http://env-url.com/v1/",
                "TEACHER_MODEL": "env-model",
            },
            clear=True,
        ):
            with patch("src.prompts.adapters.lm_factory.dspy.LM") as mock_lm:
                mock_lm.return_value = MagicMock()
                get_teacher_lm()

                mock_lm.assert_called_once()
                call_kwargs = mock_lm.call_args[1]
                assert call_kwargs["model"] == "openai/env-model"
                assert call_kwargs["api_base"] == "http://env-url.com/v1/"

    def test_params_override_env_vars(self):
        """Test that function parameters override environment variables."""
        with patch.dict(
            "os.environ",
            {
                "TEACHER_BASE_URL": "http://env-url.com/v1/",
                "TEACHER_MODEL": "env-model",
            },
            clear=True,
        ):
            with patch("src.prompts.adapters.lm_factory.dspy.LM") as mock_lm:
                mock_lm.return_value = MagicMock()
                get_teacher_lm(
                    base_url="http://param-url.com/v1/",
                    model="param-model",
                )

                mock_lm.assert_called_once()
                call_kwargs = mock_lm.call_args[1]
                assert call_kwargs["model"] == "openai/param-model"
                assert call_kwargs["api_base"] == "http://param-url.com/v1/"

    def test_default_temperature(self):
        """Test that default temperature is 0.7."""
        with patch.dict(
            "os.environ",
            {
                "TEACHER_BASE_URL": "http://example.com/v1/",
                "TEACHER_MODEL": "test-model",
            },
            clear=True,
        ):
            with patch("src.prompts.adapters.lm_factory.dspy.LM") as mock_lm:
                mock_lm.return_value = MagicMock()
                get_teacher_lm()

                call_kwargs = mock_lm.call_args[1]
                assert call_kwargs["temperature"] == 0.7

    def test_custom_temperature(self):
        """Test that custom temperature is used."""
        with patch.dict(
            "os.environ",
            {
                "TEACHER_BASE_URL": "http://example.com/v1/",
                "TEACHER_MODEL": "test-model",
            },
            clear=True,
        ):
            with patch("src.prompts.adapters.lm_factory.dspy.LM") as mock_lm:
                mock_lm.return_value = MagicMock()
                get_teacher_lm(temperature=0.3)

                call_kwargs = mock_lm.call_args[1]
                assert call_kwargs["temperature"] == 0.3

    def test_default_max_tokens(self):
        """Test that default max_tokens comes from config.max_tokens."""
        with patch.dict(
            "os.environ",
            {
                "TEACHER_BASE_URL": "http://example.com/v1/",
                "TEACHER_MODEL": "test-model",
            },
            clear=True,
        ):
            with patch("src.prompts.adapters.lm_factory.dspy.LM") as mock_lm:
                with patch("src.prompts.adapters.lm_factory.config") as mock_config:
                    mock_config.max_tokens = 8192
                    mock_lm.return_value = MagicMock()
                    get_teacher_lm()

                    call_kwargs = mock_lm.call_args[1]
                    assert call_kwargs["max_tokens"] == 8192

    def test_custom_max_tokens_param(self):
        """Test that custom max_tokens parameter overrides config."""
        with patch.dict(
            "os.environ",
            {
                "TEACHER_BASE_URL": "http://example.com/v1/",
                "TEACHER_MODEL": "test-model",
            },
            clear=True,
        ):
            with patch("src.prompts.adapters.lm_factory.dspy.LM") as mock_lm:
                with patch("src.prompts.adapters.lm_factory.config") as mock_config:
                    mock_config.max_tokens = 8192
                    mock_lm.return_value = MagicMock()
                    get_teacher_lm(max_tokens=4096)

                    call_kwargs = mock_lm.call_args[1]
                    assert call_kwargs["max_tokens"] == 4096

    def test_config_max_tokens_used_when_no_override(self):
        """Test that config.max_tokens is used when no param override."""
        with patch.dict(
            "os.environ",
            {
                "TEACHER_BASE_URL": "http://example.com/v1/",
                "TEACHER_MODEL": "test-model",
            },
            clear=True,
        ):
            with patch("src.prompts.adapters.lm_factory.dspy.LM") as mock_lm:
                with patch("src.prompts.adapters.lm_factory.config") as mock_config:
                    mock_config.max_tokens = 6000
                    mock_lm.return_value = MagicMock()
                    get_teacher_lm()

                    call_kwargs = mock_lm.call_args[1]
                    assert call_kwargs["max_tokens"] == 6000


class TestEstimateTokens:
    """Tests for _estimate_tokens function."""

    def test_empty_string_returns_zero(self):
        """Test that empty string returns 0 tokens."""
        assert _estimate_tokens("") == 0

    def test_none_returns_zero(self):
        """Test that None returns 0 tokens."""
        assert _estimate_tokens(None) == 0

    def test_short_text_estimation(self):
        """Test token estimation for short text."""
        # 14 chars / 3.5 = 4 tokens
        result = _estimate_tokens("Hello, world!")
        assert result == 3  # 13 chars / 3.5 = 3.7 -> 3

    def test_longer_text_estimation(self):
        """Test token estimation for longer text."""
        # 35 characters / 3.5 = 10 tokens
        text = "The quick brown fox jumps over the"
        result = _estimate_tokens(text)
        assert result == int(len(text) / 3.5)

    def test_conservative_ratio(self):
        """Test that 3.5 chars/token ratio overestimates tokens."""
        # Average English is ~4 chars/token, using 3.5 is conservative
        text = "This is a sample text for testing"
        actual_approx = len(text) / 4  # ~8.25 tokens
        estimated = _estimate_tokens(text)
        assert estimated >= actual_approx  # Should overestimate


class TestEstimateMessagesTokens:
    """Tests for _estimate_messages_tokens function."""

    def test_empty_list_returns_zero(self):
        """Test that empty message list returns 0 tokens."""
        assert _estimate_messages_tokens([]) == 0

    def test_single_message_dict(self):
        """Test token estimation for a single message dict."""
        messages = [{"role": "user", "content": "Hello"}]
        # 4 overhead + 5 chars / 3.5 = 4 + 1 = 5
        result = _estimate_messages_tokens(messages)
        assert result == 5

    def test_multiple_messages(self):
        """Test token estimation for multiple messages."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        # Message 1: 4 overhead + 15 chars / 3.5 = 4 + 4 = 8
        # Message 2: 4 overhead + 5 chars / 3.5 = 4 + 1 = 5
        # Total: 13
        result = _estimate_messages_tokens(messages)
        assert result == 13

    def test_multipart_content(self):
        """Test token estimation with multi-part content (text + images)."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image_url", "image_url": {"url": "data:image/..."}},
                ],
            }
        ]
        # 4 overhead + 19 chars / 3.5 = 4 + 5 = 9
        # Image parts without "text" key are ignored
        result = _estimate_messages_tokens(messages)
        assert result == 9

    def test_message_with_object_attribute(self):
        """Test token estimation for message objects with content attribute."""

        class MockMessage:
            content = "Test message"

        messages = [MockMessage()]
        # 4 overhead + 12 chars / 3.5 = 4 + 3 = 7
        result = _estimate_messages_tokens(messages)
        assert result == 7


class TestTokenAwareLM:
    """Tests for TokenAwareLM wrapper class."""

    def _create_mock_lm(self):
        """Create a mock LM with required attributes."""
        mock_lm = MagicMock()
        mock_lm.model = "openai/test-model"
        mock_lm.cache = None
        mock_lm.history = []
        mock_lm.callbacks = []
        mock_lm.kwargs = {"max_tokens": 8192}
        mock_lm.forward.return_value = ["test response"]
        return mock_lm

    def test_passes_through_when_within_limits(self):
        """Test that requests within limits pass through unchanged."""
        mock_lm = self._create_mock_lm()
        wrapper = TokenAwareLM(mock_lm, context_length=16384)

        # Short message, plenty of room
        messages = [{"role": "user", "content": "Hello"}]
        wrapper.forward(messages=messages, max_tokens=8192)

        # Should pass through without adjustment
        call_kwargs = mock_lm.forward.call_args[1]
        assert call_kwargs.get("max_tokens", 8192) == 8192

    def test_adjusts_max_tokens_when_input_large(self):
        """Test that max_tokens is reduced when input is large."""
        mock_lm = self._create_mock_lm()
        wrapper = TokenAwareLM(mock_lm, context_length=1000, safety_buffer=100)

        # Large message: 2800 chars / 3.5 = 800 tokens
        # Available: 1000 - 800 - 100 = 100 tokens
        large_content = "x" * 2800
        messages = [{"role": "user", "content": large_content}]

        with patch("src.prompts.adapters.lm_factory.logger") as mock_logger:
            wrapper.forward(messages=messages, max_tokens=8192)

            # Should have logged a warning
            mock_logger.warning.assert_called_once()
            assert "Adjusting max_tokens" in mock_logger.warning.call_args[0][0]

        # Should have been clamped to available
        call_kwargs = mock_lm.forward.call_args[1]
        # 800 tokens input + 4 overhead = 804, available = 1000 - 804 - 100 = 96
        # But minimum is 256, so it should be 256
        assert call_kwargs["max_tokens"] == 256

    def test_uses_minimum_output_tokens(self):
        """Test that minimum output tokens is enforced."""
        mock_lm = self._create_mock_lm()
        wrapper = TokenAwareLM(
            mock_lm, context_length=500, min_output_tokens=256, safety_buffer=100
        )

        # Very large input that would leave < 256 tokens
        large_content = "x" * 1750  # 500 tokens
        messages = [{"role": "user", "content": large_content}]

        with patch("src.prompts.adapters.lm_factory.logger"):
            wrapper.forward(messages=messages, max_tokens=8192)

        call_kwargs = mock_lm.forward.call_args[1]
        assert call_kwargs["max_tokens"] == 256

    def test_handles_prompt_instead_of_messages(self):
        """Test token estimation works with prompt parameter."""
        mock_lm = self._create_mock_lm()
        wrapper = TokenAwareLM(mock_lm, context_length=1000, safety_buffer=100)

        # Large prompt: 2800 chars / 3.5 = 800 tokens
        large_prompt = "y" * 2800

        with patch("src.prompts.adapters.lm_factory.logger"):
            wrapper.forward(prompt=large_prompt, max_tokens=8192)

        call_kwargs = mock_lm.forward.call_args[1]
        assert call_kwargs["max_tokens"] == 256

    def test_copies_lm_attributes(self):
        """Test that wrapper copies required LM attributes."""
        mock_lm = self._create_mock_lm()
        wrapper = TokenAwareLM(mock_lm, context_length=16384)

        assert wrapper.model == "openai/test-model"
        assert wrapper.cache == mock_lm.cache
        assert wrapper.history == mock_lm.history
        assert wrapper.callbacks == mock_lm.callbacks
        assert wrapper.kwargs == mock_lm.kwargs

    def test_delegates_attribute_access(self):
        """Test that unknown attributes are delegated to wrapped LM."""
        mock_lm = self._create_mock_lm()
        mock_lm.custom_attr = "custom_value"
        wrapper = TokenAwareLM(mock_lm, context_length=16384)

        assert wrapper.custom_attr == "custom_value"

    def test_uses_lm_kwargs_max_tokens_as_default(self):
        """Test that wrapper uses LM's kwargs max_tokens when not in call."""
        mock_lm = self._create_mock_lm()
        mock_lm.kwargs = {"max_tokens": 4096}
        wrapper = TokenAwareLM(mock_lm, context_length=5000, safety_buffer=100)

        # Large input: 1750 chars / 3.5 = 500 tokens + 4 overhead = 504
        # Available: 5000 - 504 - 100 = 4396
        # Requested from kwargs: 4096, which is < 4396, so no adjustment
        messages = [{"role": "user", "content": "x" * 1750}]

        wrapper.forward(messages=messages)

        # No max_tokens override should be needed
        call_kwargs = mock_lm.forward.call_args[1]
        assert "max_tokens" not in call_kwargs or call_kwargs.get("max_tokens") == 4096
