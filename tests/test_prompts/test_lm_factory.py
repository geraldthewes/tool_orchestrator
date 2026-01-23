"""
Tests for DSPy LM Factory functions.

Tests the factory functions for creating DSPy LM instances.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.prompts.adapters.lm_factory import get_teacher_lm


class TestGetTeacherLM:
    """Tests for get_teacher_lm function."""

    def test_raises_error_when_base_url_missing(self):
        """Test that missing base URL raises ValueError."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="Teacher LLM base URL not configured"):
                get_teacher_lm(model="test-model")

    def test_raises_error_when_model_missing(self):
        """Test that missing model raises ValueError."""
        with patch.dict("os.environ", {"TEACHER_BASE_URL": "http://example.com"}, clear=True):
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
        """Test that default max_tokens is 1024."""
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
                assert call_kwargs["max_tokens"] == 1024
