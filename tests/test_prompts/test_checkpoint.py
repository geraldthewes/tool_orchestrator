"""
Tests for DSPy Checkpoint Manager.

Tests checkpoint save, load, and manifest functionality.
"""

import json
import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import dspy

from src.prompts.optimization.checkpoint import CheckpointManager


class TestCheckpointManagerInit:
    """Tests for CheckpointManager initialization."""

    def test_creates_checkpoint_directory(self, tmp_path):
        """Test that checkpoint directory is created on init."""
        checkpoint_dir = tmp_path / "checkpoints"
        manager = CheckpointManager(checkpoint_dir, "test_module")

        assert (checkpoint_dir / "test_module").exists()
        assert manager.checkpoint_dir == checkpoint_dir / "test_module"

    def test_initializes_manifest(self, tmp_path):
        """Test that manifest is initialized with correct structure."""
        manager = CheckpointManager(tmp_path, "orchestrator", strategy="gepa")

        assert manager.manifest["module_name"] == "orchestrator"
        assert manager.manifest["strategy"] == "gepa"
        assert "started_at" in manager.manifest
        assert manager.manifest["checkpoints"] == []
        assert manager.manifest["best"] is None

    def test_initial_best_score(self, tmp_path):
        """Test that initial best score is negative infinity."""
        manager = CheckpointManager(tmp_path, "test")

        assert manager.best_score == float("-inf")
        assert manager.checkpoint_count == 0


class TestMetricWrapper:
    """Tests for metric wrapping functionality."""

    def test_creates_metric_wrapper(self, tmp_path):
        """Test creating a wrapped metric function."""
        manager = CheckpointManager(tmp_path, "test")
        module = MagicMock(spec=dspy.Module)

        base_metric = lambda e, p, t=None: 0.5

        wrapped = manager.create_metric_wrapper(base_metric, module)

        assert callable(wrapped)
        assert manager._module_ref is module

    def test_wrapped_metric_returns_score(self, tmp_path):
        """Test that wrapped metric returns the base metric score."""
        manager = CheckpointManager(tmp_path, "test")
        module = MagicMock(spec=dspy.Module)

        base_metric = lambda e, p, t=None: 0.75

        wrapped = manager.create_metric_wrapper(base_metric, module)
        score = wrapped(MagicMock(), MagicMock())

        assert score == 0.75

    def test_wrapped_metric_checkpoints_on_improvement(self, tmp_path):
        """Test that checkpoint is saved when score improves."""
        manager = CheckpointManager(tmp_path, "test")
        module = MagicMock(spec=dspy.Module)

        base_metric = lambda e, p, t=None: 0.8

        wrapped = manager.create_metric_wrapper(base_metric, module)
        wrapped(MagicMock(), MagicMock())

        assert manager.best_score == 0.8
        assert manager.checkpoint_count == 1
        assert len(manager.manifest["checkpoints"]) == 1

    def test_no_checkpoint_when_score_not_improved(self, tmp_path):
        """Test that no checkpoint is saved when score doesn't improve."""
        manager = CheckpointManager(tmp_path, "test")
        module = MagicMock(spec=dspy.Module)

        # First call improves
        metric1 = lambda e, p, t=None: 0.8
        wrapped1 = manager.create_metric_wrapper(metric1, module)
        wrapped1(MagicMock(), MagicMock())

        # Second call with same score doesn't improve
        metric2 = lambda e, p, t=None: 0.8
        wrapped2 = manager.create_metric_wrapper(metric2, module)
        wrapped2(MagicMock(), MagicMock())

        assert manager.checkpoint_count == 1

    def test_creates_gepa_metric_wrapper(self, tmp_path):
        """Test creating a GEPA-compatible wrapped metric."""
        manager = CheckpointManager(tmp_path, "test")
        module = MagicMock(spec=dspy.Module)

        # GEPA metrics have different signature
        def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
            return 0.9

        wrapped = manager.create_gepa_metric_wrapper(gepa_metric, module)
        score = wrapped(MagicMock(), MagicMock())

        assert score == 0.9
        assert manager.best_score == 0.9


class TestCheckpointSaving:
    """Tests for checkpoint save functionality."""

    def test_saves_checkpoint_file(self, tmp_path):
        """Test that checkpoint file is saved."""
        manager = CheckpointManager(tmp_path, "test")
        module = MagicMock(spec=dspy.Module)
        manager._module_ref = module

        path = manager._save_checkpoint(0.85)

        module.save.assert_called_once()
        assert "checkpoint_001.json" in str(path)

    def test_updates_manifest_on_checkpoint(self, tmp_path):
        """Test that manifest is updated when checkpoint is saved."""
        manager = CheckpointManager(tmp_path, "test")
        module = MagicMock(spec=dspy.Module)
        manager._module_ref = module

        manager._save_checkpoint(0.85)

        assert len(manager.manifest["checkpoints"]) == 1
        assert manager.manifest["checkpoints"][0]["score"] == 0.85
        assert manager.manifest["best"]["score"] == 0.85

    def test_increments_checkpoint_count(self, tmp_path):
        """Test that checkpoint count is incremented."""
        manager = CheckpointManager(tmp_path, "test")
        module = MagicMock(spec=dspy.Module)
        manager._module_ref = module

        manager._save_checkpoint(0.8)
        manager._save_checkpoint(0.9)

        assert manager.checkpoint_count == 2
        assert manager.manifest["checkpoints"][-1]["id"] == 2

    def test_checkpoint_entry_structure(self, tmp_path):
        """Test the structure of checkpoint entries."""
        manager = CheckpointManager(tmp_path, "test")
        module = MagicMock(spec=dspy.Module)
        manager._module_ref = module

        manager._save_checkpoint(0.75)

        entry = manager.manifest["checkpoints"][0]
        assert "id" in entry
        assert "score" in entry
        assert "timestamp" in entry
        assert "path" in entry
        assert entry["path"] == "checkpoint_001.json"

    def test_no_save_without_module_ref(self, tmp_path):
        """Test that save fails gracefully without module reference."""
        manager = CheckpointManager(tmp_path, "test")

        result = manager._save_checkpoint(0.5)

        assert result is None


class TestManifestHandling:
    """Tests for manifest save and load functionality."""

    def test_saves_manifest_to_file(self, tmp_path):
        """Test that manifest is saved to JSON file."""
        manager = CheckpointManager(tmp_path, "test")
        manager.save_manifest()

        manifest_path = tmp_path / "test" / "manifest.json"
        assert manifest_path.exists()

        with open(manifest_path) as f:
            saved = json.load(f)

        assert saved["module_name"] == "test"

    def test_finalize_saves_completion_info(self, tmp_path):
        """Test that finalize adds completion metadata."""
        manager = CheckpointManager(tmp_path, "test")
        manager.best_score = 0.95

        manager.finalize()

        assert "completed_at" in manager.manifest
        assert manager.manifest["final_best_score"] == 0.95


class TestCheckpointLoading:
    """Tests for checkpoint loading functionality."""

    def test_get_best_checkpoint_returns_path(self, tmp_path):
        """Test getting best checkpoint path from manifest."""
        checkpoint_dir = tmp_path / "test"
        checkpoint_dir.mkdir()

        manifest = {
            "module_name": "test",
            "checkpoints": [
                {"id": 1, "score": 0.8, "path": "checkpoint_001.json"},
                {"id": 2, "score": 0.9, "path": "checkpoint_002.json"},
            ],
            "best": {"id": 2, "score": 0.9, "path": "checkpoint_002.json"},
        }

        with open(checkpoint_dir / "manifest.json", "w") as f:
            json.dump(manifest, f)

        # Create the checkpoint file
        (checkpoint_dir / "checkpoint_002.json").write_text("{}")

        result = CheckpointManager.get_best_checkpoint(checkpoint_dir)

        assert result == checkpoint_dir / "checkpoint_002.json"

    def test_get_best_checkpoint_returns_none_if_no_manifest(self, tmp_path):
        """Test returns None when no manifest exists."""
        result = CheckpointManager.get_best_checkpoint(tmp_path)

        assert result is None

    def test_get_best_checkpoint_returns_none_if_no_best(self, tmp_path):
        """Test returns None when no best checkpoint in manifest."""
        checkpoint_dir = tmp_path / "test"
        checkpoint_dir.mkdir()

        manifest = {"module_name": "test", "checkpoints": [], "best": None}

        with open(checkpoint_dir / "manifest.json", "w") as f:
            json.dump(manifest, f)

        result = CheckpointManager.get_best_checkpoint(checkpoint_dir)

        assert result is None

    def test_load_checkpoint_loads_module_state(self, tmp_path):
        """Test loading checkpoint into module."""
        checkpoint_path = tmp_path / "checkpoint.json"
        checkpoint_path.write_text("{}")

        module = MagicMock(spec=dspy.Module)

        result = CheckpointManager.load_checkpoint(module, checkpoint_path)

        module.load.assert_called_once_with(str(checkpoint_path))
        assert result is module

    def test_load_checkpoint_raises_for_missing_file(self, tmp_path):
        """Test that loading missing checkpoint raises error."""
        missing_path = tmp_path / "missing.json"
        module = MagicMock(spec=dspy.Module)

        with pytest.raises(FileNotFoundError):
            CheckpointManager.load_checkpoint(module, missing_path)

    def test_get_manifest_returns_manifest_dict(self, tmp_path):
        """Test getting manifest as dictionary."""
        checkpoint_dir = tmp_path / "test"
        checkpoint_dir.mkdir()

        manifest = {"module_name": "test", "strategy": "gepa", "checkpoints": []}

        with open(checkpoint_dir / "manifest.json", "w") as f:
            json.dump(manifest, f)

        result = CheckpointManager.get_manifest(checkpoint_dir)

        assert result["module_name"] == "test"
        assert result["strategy"] == "gepa"

    def test_get_best_score_returns_score(self, tmp_path):
        """Test getting best score from manifest."""
        checkpoint_dir = tmp_path / "test"
        checkpoint_dir.mkdir()

        manifest = {"best": {"id": 1, "score": 0.92, "path": "checkpoint_001.json"}}

        with open(checkpoint_dir / "manifest.json", "w") as f:
            json.dump(manifest, f)

        result = CheckpointManager.get_best_score(checkpoint_dir)

        assert result == 0.92

    def test_get_best_score_returns_none_if_no_manifest(self, tmp_path):
        """Test returns None when no manifest."""
        result = CheckpointManager.get_best_score(tmp_path)

        assert result is None


class TestCheckpointIntegration:
    """Integration tests for checkpoint workflow."""

    def test_full_checkpoint_workflow(self, tmp_path):
        """Test complete checkpoint save and resume workflow."""
        # Phase 1: Run optimization with checkpointing
        manager1 = CheckpointManager(tmp_path, "orchestrator", strategy="gepa")
        checkpoint_dir = tmp_path / "orchestrator"

        # Create a mock module that writes to actual files when save is called
        module1 = MagicMock(spec=dspy.Module)
        module1.save.side_effect = lambda path: Path(path).write_text("{}")

        def base_metric(e, p, t=None):
            return 0.7

        wrapped = manager1.create_metric_wrapper(base_metric, module1)

        # Simulate metric calls with improving scores
        wrapped(MagicMock(), MagicMock())  # Score: 0.7, creates checkpoint_001

        # Simulate another improvement
        manager1.best_score = 0.7  # Reset to current best
        manager1._save_checkpoint(0.85)  # Creates checkpoint_002

        manager1.finalize()

        # Phase 2: Resume from checkpoint
        checkpoint_path = CheckpointManager.get_best_checkpoint(checkpoint_dir)
        assert checkpoint_path is not None
        assert checkpoint_path.name == "checkpoint_002.json"

        module2 = MagicMock(spec=dspy.Module)
        CheckpointManager.load_checkpoint(module2, checkpoint_path)

        module2.load.assert_called_once()

    def test_multiple_checkpoint_sequence(self, tmp_path):
        """Test saving multiple checkpoints in sequence."""
        manager = CheckpointManager(tmp_path, "test")
        module = MagicMock(spec=dspy.Module)
        manager._module_ref = module

        # Save multiple improving checkpoints
        manager.best_score = float("-inf")
        manager._save_checkpoint(0.6)
        manager._save_checkpoint(0.7)
        manager._save_checkpoint(0.85)

        assert manager.checkpoint_count == 3
        assert len(manager.manifest["checkpoints"]) == 3
        assert manager.manifest["best"]["score"] == 0.85
        assert manager.manifest["best"]["path"] == "checkpoint_003.json"
