"""
Tests for DSPy Prompt Optimizer.

Tests optimizer functionality with mocked optimizers and LMs.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

import dspy

from src.prompts.optimization.optimizer import PromptOptimizer, optimize_all_modules


class TestPromptOptimizer:
    """Tests for PromptOptimizer class."""

    def test_default_strategy_is_gepa(self):
        """Test that GEPA is the default optimization strategy."""
        optimizer = PromptOptimizer()
        assert optimizer.strategy == "gepa"

    def test_custom_strategy(self):
        """Test setting a custom strategy."""
        optimizer = PromptOptimizer(strategy="bootstrap")
        assert optimizer.strategy == "bootstrap"

        optimizer = PromptOptimizer(strategy="mipro")
        assert optimizer.strategy == "mipro"

    def test_gepa_parameters(self):
        """Test GEPA-specific parameters."""
        mock_lm = MagicMock()
        optimizer = PromptOptimizer(
            strategy="gepa",
            reflection_lm=mock_lm,
            gepa_auto="heavy",
        )
        assert optimizer.reflection_lm is mock_lm
        assert optimizer.gepa_auto == "heavy"

    def test_default_gepa_auto(self):
        """Test default gepa_auto preset."""
        optimizer = PromptOptimizer()
        assert optimizer.gepa_auto == "light"

    @patch("src.prompts.optimization.optimizer.get_teacher_lm")
    @patch("src.prompts.optimization.optimizer.GEPA")
    def test_gepa_optimization(self, mock_gepa_class, mock_get_lm):
        """Test GEPA optimization flow."""
        mock_lm = MagicMock()
        mock_get_lm.return_value = mock_lm

        mock_optimizer = MagicMock()
        mock_optimized = MagicMock()
        mock_optimizer.compile.return_value = mock_optimized
        mock_gepa_class.return_value = mock_optimizer

        optimizer = PromptOptimizer(strategy="gepa", gepa_auto="medium")
        module = MagicMock(spec=dspy.Module)
        trainset = [dspy.Example(question="test").with_inputs("question")]

        result = optimizer._optimize(module, trainset, Mock(), mock_lm)

        mock_gepa_class.assert_called_once()
        call_kwargs = mock_gepa_class.call_args[1]
        assert call_kwargs["auto"] == "medium"
        assert call_kwargs["reflection_lm"] is mock_lm
        mock_optimizer.compile.assert_called_once()
        assert result is mock_optimized

    @patch("src.prompts.optimization.optimizer.get_teacher_lm")
    @patch("src.prompts.optimization.optimizer.GEPA")
    def test_gepa_with_devset(self, mock_gepa_class, mock_get_lm):
        """Test GEPA optimization with validation set."""
        mock_lm = MagicMock()
        mock_get_lm.return_value = mock_lm

        mock_optimizer = MagicMock()
        mock_optimizer.compile.return_value = MagicMock()
        mock_gepa_class.return_value = mock_optimizer

        optimizer = PromptOptimizer(strategy="gepa")
        module = MagicMock(spec=dspy.Module)
        trainset = [dspy.Example(question="train").with_inputs("question")]
        devset = [dspy.Example(question="dev").with_inputs("question")]

        optimizer._optimize(module, trainset, Mock(), mock_lm, devset=devset)

        compile_call = mock_optimizer.compile.call_args
        assert "valset" in compile_call[1]
        assert compile_call[1]["valset"] is devset

    @patch("src.prompts.optimization.optimizer.get_teacher_lm")
    @patch("src.prompts.optimization.optimizer.GEPA")
    def test_gepa_with_custom_reflection_lm(self, mock_gepa_class, mock_get_lm):
        """Test GEPA with custom reflection LM."""
        teacher_lm = MagicMock()
        reflection_lm = MagicMock()
        mock_get_lm.return_value = teacher_lm

        mock_optimizer = MagicMock()
        mock_optimizer.compile.return_value = MagicMock()
        mock_gepa_class.return_value = mock_optimizer

        optimizer = PromptOptimizer(strategy="gepa", reflection_lm=reflection_lm)
        module = MagicMock(spec=dspy.Module)
        trainset = [dspy.Example(question="test").with_inputs("question")]

        optimizer._optimize(module, trainset, Mock(), teacher_lm)

        call_kwargs = mock_gepa_class.call_args[1]
        assert call_kwargs["reflection_lm"] is reflection_lm

    @patch("src.prompts.optimization.optimizer.get_teacher_lm")
    @patch("src.prompts.optimization.optimizer.BootstrapFewShot")
    def test_bootstrap_optimization(self, mock_bootstrap_class, mock_get_lm):
        """Test BootstrapFewShot optimization flow."""
        mock_lm = MagicMock()
        mock_get_lm.return_value = mock_lm

        mock_optimizer = MagicMock()
        mock_optimized = MagicMock()
        mock_optimizer.compile.return_value = mock_optimized
        mock_bootstrap_class.return_value = mock_optimizer

        optimizer = PromptOptimizer(strategy="bootstrap", max_bootstrapped_demos=8)
        module = MagicMock(spec=dspy.Module)
        trainset = [dspy.Example(question="test").with_inputs("question")]

        result = optimizer._optimize(module, trainset, Mock(), mock_lm)

        mock_bootstrap_class.assert_called_once()
        call_kwargs = mock_bootstrap_class.call_args[1]
        assert call_kwargs["max_bootstrapped_demos"] == 8
        assert result is mock_optimized

    @patch("src.prompts.optimization.optimizer.get_teacher_lm")
    @patch("src.prompts.optimization.optimizer.MIPROv2")
    def test_mipro_optimization(self, mock_mipro_class, mock_get_lm):
        """Test MIPROv2 optimization flow."""
        mock_lm = MagicMock()
        mock_get_lm.return_value = mock_lm

        mock_optimizer = MagicMock()
        mock_optimized = MagicMock()
        mock_optimizer.compile.return_value = mock_optimized
        mock_mipro_class.return_value = mock_optimizer

        optimizer = PromptOptimizer(strategy="mipro", max_labeled_demos=8)
        module = MagicMock(spec=dspy.Module)
        trainset = [dspy.Example(question="test").with_inputs("question")]

        result = optimizer._optimize(module, trainset, Mock(), mock_lm)

        mock_mipro_class.assert_called_once()
        assert result is mock_optimized

    def test_unknown_strategy_raises_error(self):
        """Test that unknown strategy raises ValueError."""
        optimizer = PromptOptimizer(strategy="unknown")
        module = MagicMock(spec=dspy.Module)
        trainset = [dspy.Example(question="test").with_inputs("question")]

        with pytest.raises(ValueError, match="Unknown optimization strategy"):
            with patch("src.prompts.optimization.optimizer.dspy.context"):
                optimizer._optimize(module, trainset, Mock(), MagicMock())


class TestOptimizeOrchestrator:
    """Tests for optimize_orchestrator method."""

    @patch("src.prompts.optimization.optimizer.load_all_training_examples")
    @patch("src.prompts.optimization.optimizer.get_train_dev_split")
    @patch("src.prompts.optimization.optimizer.get_teacher_lm")
    @patch("src.prompts.optimization.optimizer.GEPA")
    def test_auto_loads_training_examples(
        self, mock_gepa, mock_get_lm, mock_split, mock_load
    ):
        """Test that optimize_orchestrator auto-loads training examples."""
        mock_examples = [
            dspy.Example(question=f"q{i}").with_inputs("question") for i in range(100)
        ]
        mock_load.return_value = mock_examples
        mock_split.return_value = (mock_examples[:20], mock_examples[20:])
        mock_get_lm.return_value = MagicMock()
        mock_gepa.return_value.compile.return_value = MagicMock()

        optimizer = PromptOptimizer(strategy="gepa")
        module = MagicMock(spec=dspy.Module)

        optimizer.optimize_orchestrator(module)

        mock_load.assert_called_once()
        mock_split.assert_called_once_with(mock_examples, dev_ratio=0.8)

    @patch("src.prompts.optimization.optimizer.load_orchestration_dataset")
    @patch("src.prompts.optimization.optimizer.load_all_training_examples")
    @patch("src.prompts.optimization.optimizer.get_teacher_lm")
    @patch("src.prompts.optimization.optimizer.BootstrapFewShot")
    def test_falls_back_to_legacy_dataset(
        self, mock_bootstrap, mock_get_lm, mock_load_all, mock_load_legacy
    ):
        """Test fallback to legacy dataset when no training examples."""
        mock_load_all.return_value = []
        mock_load_legacy.return_value = [
            dspy.Example(question="q1").with_inputs("question")
        ]
        mock_get_lm.return_value = MagicMock()
        mock_bootstrap.return_value.compile.return_value = MagicMock()

        optimizer = PromptOptimizer(strategy="bootstrap")
        module = MagicMock(spec=dspy.Module)

        optimizer.optimize_orchestrator(module)

        mock_load_legacy.assert_called_once()

    @patch("src.prompts.optimization.optimizer.get_teacher_lm")
    def test_returns_module_when_no_data(self, mock_get_lm):
        """Test returns original module when no training data."""
        mock_get_lm.return_value = MagicMock()

        optimizer = PromptOptimizer()
        module = MagicMock(spec=dspy.Module)

        with patch(
            "src.prompts.optimization.optimizer.load_all_training_examples"
        ) as mock_load:
            with patch(
                "src.prompts.optimization.optimizer.load_orchestration_dataset"
            ) as mock_legacy:
                mock_load.return_value = []
                mock_legacy.return_value = []

                result = optimizer.optimize_orchestrator(module)

        assert result is module


class TestSaveLoad:
    """Tests for save and load functionality."""

    def test_save_module(self, tmp_path):
        """Test saving an optimized module."""
        module = MagicMock(spec=dspy.Module)
        save_path = tmp_path / "optimized" / "module.json"

        PromptOptimizer.save(module, str(save_path))

        module.save.assert_called_once_with(str(save_path))
        assert save_path.parent.exists()

    def test_load_module(self, tmp_path):
        """Test loading an optimized module."""
        module = MagicMock(spec=dspy.Module)
        load_path = tmp_path / "module.json"
        load_path.write_text("{}")  # Create file

        result = PromptOptimizer.load(module, str(load_path))

        module.load.assert_called_once_with(str(load_path))
        assert result is module

    def test_load_missing_file_raises_error(self, tmp_path):
        """Test loading from missing file raises FileNotFoundError."""
        module = MagicMock(spec=dspy.Module)
        missing_path = tmp_path / "missing.json"

        with pytest.raises(FileNotFoundError):
            PromptOptimizer.load(module, str(missing_path))


class TestOptimizeAllModules:
    """Tests for optimize_all_modules function."""

    def test_default_strategy_is_gepa(self):
        """Test that optimize_all_modules uses GEPA by default."""
        import inspect
        from src.prompts.optimization.optimizer import optimize_all_modules

        sig = inspect.signature(optimize_all_modules)
        assert sig.parameters["strategy"].default == "gepa"

    @patch("src.prompts.modules.QueryRouterModule")
    @patch("src.prompts.modules.ToolOrchestratorModule")
    def test_optimizes_all_modules(self, mock_orchestrator, mock_router, tmp_path):
        """Test optimizing all modules."""
        mock_router.return_value = MagicMock(spec=dspy.Module)
        mock_orchestrator.return_value = MagicMock(spec=dspy.Module)

        with patch.object(PromptOptimizer, "save"):
            with patch.object(
                PromptOptimizer, "optimize_router", return_value=MagicMock()
            ):
                with patch.object(
                    PromptOptimizer, "optimize_orchestrator", return_value=MagicMock()
                ):
                    result = optimize_all_modules(
                        output_dir=str(tmp_path), strategy="bootstrap"
                    )

        assert "router" in result
        assert "orchestrator" in result
