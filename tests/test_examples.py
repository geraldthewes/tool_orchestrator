"""Tests for DSPy example generation and loading."""

import json

import dspy
import pytest

from src.prompts.optimization.datasets import (
    create_orchestration_example,
    create_routing_example,
    get_examples_by_category,
    get_examples_by_tool,
    get_train_dev_split,
    load_all_routing_examples,
    load_all_training_examples,
    load_examples_from_directory,
)


class TestExampleCreation:
    """Test example creation functions."""

    def test_create_orchestration_example(self):
        """Test creating a basic orchestration example."""
        example = create_orchestration_example(
            question="What is 2^10?",
            answer="1024",
            expected_keywords=["1024"],
        )
        assert isinstance(example, dspy.Example)
        assert example.question == "What is 2^10?"
        assert example.answer == "1024"
        assert example.expected_keywords == ["1024"]
        assert "question" in example.inputs()

    def test_create_orchestration_example_with_tool(self):
        """Test creating orchestration example with tool metadata."""
        example = create_orchestration_example(
            question="Calculate sqrt(144)",
            answer="12",
            expected_keywords=["12"],
            tool="calculate",
            category="single_tool",
        )
        assert example.tool == "calculate"
        assert example.category == "single_tool"

    def test_create_routing_example(self):
        """Test creating a routing example."""
        example = create_routing_example(
            query="Hello",
            needs_tools=False,
            reasoning="Simple greeting",
            direct_answer="Hello! How can I help?",
        )
        assert isinstance(example, dspy.Example)
        assert example.query == "Hello"
        assert example.needs_tools is False
        assert example.reasoning == "Simple greeting"
        assert "query" in example.inputs()


class TestExampleLoading:
    """Test example loading from files."""

    @pytest.fixture
    def temp_examples_dir(self, tmp_path):
        """Create temporary directory with test examples."""
        examples_dir = tmp_path / "examples"
        examples_dir.mkdir()

        # Create test orchestration file
        orch_examples = [
            {
                "question": "Test Q1",
                "answer": "A1",
                "expected_keywords": ["test"],
                "tool": "calculate",
            },
            {
                "question": "Test Q2",
                "answer": "A2",
                "expected_keywords": [],
                "tool": "python_execute",
            },
        ]
        with open(examples_dir / "test_examples.jsonl", "w") as f:
            for ex in orch_examples:
                f.write(json.dumps(ex) + "\n")

        # Create test routing file
        routing_examples = [
            {
                "query": "Hello",
                "needs_tools": False,
                "reasoning": "Greeting",
                "direct_answer": "Hi!",
            },
            {"query": "Search news", "needs_tools": True, "reasoning": "Needs search"},
        ]
        with open(examples_dir / "routing_test.jsonl", "w") as f:
            for ex in routing_examples:
                f.write(json.dumps(ex) + "\n")

        return examples_dir

    def test_load_examples_from_directory_orchestration(self, temp_examples_dir):
        """Test loading orchestration examples from directory."""
        examples = load_examples_from_directory(
            str(temp_examples_dir),
            example_type="orchestration",
        )
        assert len(examples) == 2
        assert examples[0].question == "Test Q1"

    def test_load_examples_from_directory_routing(self, temp_examples_dir):
        """Test loading routing examples from directory."""
        examples = load_examples_from_directory(
            str(temp_examples_dir),
            example_type="routing",
        )
        assert len(examples) == 2
        assert examples[0].query == "Hello"

    def test_load_examples_from_directory_all(self, temp_examples_dir):
        """Test loading all examples from directory."""
        examples = load_examples_from_directory(
            str(temp_examples_dir),
            example_type="all",
        )
        assert len(examples) == 4


class TestExampleFiltering:
    """Test example filtering functions."""

    @pytest.fixture
    def sample_examples(self):
        """Create sample examples for filtering tests."""
        return [
            create_orchestration_example(
                "Q1", "A1", tool="calculate", category="single_tool"
            ),
            create_orchestration_example(
                "Q2", "A2", tool="calculate", category="single_tool"
            ),
            create_orchestration_example(
                "Q3", "A3", tool="python_execute", category="single_tool"
            ),
            create_orchestration_example(
                "Q4", "A4", tool="web_search", category="multi_tool"
            ),
        ]

    def test_get_examples_by_tool(self, sample_examples):
        """Test filtering by tool."""
        calc_examples = get_examples_by_tool(sample_examples, "calculate")
        assert len(calc_examples) == 2

        python_examples = get_examples_by_tool(sample_examples, "python_execute")
        assert len(python_examples) == 1

    def test_get_examples_by_category(self, sample_examples):
        """Test filtering by category."""
        single_tool = get_examples_by_category(sample_examples, "single_tool")
        assert len(single_tool) == 3

        multi_tool = get_examples_by_category(sample_examples, "multi_tool")
        assert len(multi_tool) == 1


class TestTrainDevSplit:
    """Test train/dev split functionality."""

    def test_split_with_seed(self):
        """Test that split is reproducible with seed."""
        examples = [create_orchestration_example(f"Q{i}", f"A{i}") for i in range(10)]

        train1, dev1 = get_train_dev_split(examples, dev_ratio=0.2, seed=42)
        train2, dev2 = get_train_dev_split(examples, dev_ratio=0.2, seed=42)

        assert len(train1) == len(train2)
        assert len(dev1) == len(dev2)
        assert [e.question for e in train1] == [e.question for e in train2]

    def test_split_ratio(self):
        """Test that split respects ratio."""
        examples = [create_orchestration_example(f"Q{i}", f"A{i}") for i in range(100)]

        train, dev = get_train_dev_split(examples, dev_ratio=0.2)
        assert len(train) == 80
        assert len(dev) == 20

    def test_split_preserves_count(self):
        """Test that no examples are lost in split."""
        examples = [create_orchestration_example(f"Q{i}", f"A{i}") for i in range(50)]

        train, dev = get_train_dev_split(examples, dev_ratio=0.3)
        assert len(train) + len(dev) == 50


class TestGeneratedExamples:
    """Test the actually generated examples in data/examples/."""

    def test_load_generated_examples(self):
        """Test that generated examples can be loaded."""
        examples = load_all_training_examples(include_legacy=False)
        # Should have 125 orchestration examples (150 - 25 routing)
        assert (
            len(examples) >= 100
        ), f"Expected at least 100 examples, got {len(examples)}"

    def test_load_generated_routing(self):
        """Test that generated routing examples can be loaded."""
        examples = load_all_routing_examples(include_legacy=False)
        assert (
            len(examples) >= 20
        ), f"Expected at least 20 routing examples, got {len(examples)}"

    def test_example_structure(self):
        """Test that examples have correct structure."""
        examples = load_all_training_examples(include_legacy=False)

        for ex in examples[:10]:  # Check first 10
            assert hasattr(ex, "question")
            assert hasattr(ex, "answer")
            assert hasattr(ex, "expected_keywords")
            assert "question" in ex.inputs()
