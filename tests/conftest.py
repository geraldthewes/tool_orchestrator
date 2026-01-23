"""
Pytest configuration and fixtures for ToolOrchestra tests.
"""

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_dspy_lm():
    """Create a mock DSPy LM."""
    mock_lm = MagicMock()
    mock_lm.model = "test-model"
    return mock_lm


@pytest.fixture
def mock_dspy_context(mock_dspy_lm):
    """Create a mock DSPy context manager."""
    with patch("dspy.context") as mock_context:
        mock_context.return_value.__enter__ = MagicMock(return_value=mock_dspy_lm)
        mock_context.return_value.__exit__ = MagicMock(return_value=False)
        yield mock_context


@pytest.fixture
def mock_orchestrator_lm(mock_dspy_lm):
    """Mock the get_orchestrator_lm function."""
    with patch("src.prompts.adapters.lm_factory.get_orchestrator_lm") as mock_get:
        mock_get.return_value = mock_dspy_lm
        yield mock_get


@pytest.fixture
def mock_teacher_lm(mock_dspy_lm):
    """Mock the get_teacher_lm function."""
    with patch("src.prompts.adapters.lm_factory.get_teacher_lm") as mock_get:
        mock_get.return_value = mock_dspy_lm
        yield mock_get


@pytest.fixture
def mock_delegate_lm(mock_dspy_lm):
    """Mock the get_delegate_lm function."""
    with patch("src.prompts.adapters.lm_factory.get_delegate_lm") as mock_get:
        mock_get.return_value = mock_dspy_lm
        yield mock_get


@pytest.fixture
def mock_fast_lm(mock_dspy_lm):
    """Mock the get_fast_lm function."""
    with patch("src.prompts.adapters.lm_factory.get_fast_lm") as mock_get:
        mock_get.return_value = mock_dspy_lm
        yield mock_get


@pytest.fixture(autouse=True)
def reset_dspy_settings():
    """Reset DSPy settings before each test."""
    import dspy

    # Reset any configured LMs
    try:
        dspy.settings.configure(lm=None)
    except Exception:
        pass  # May not be configured
    yield
