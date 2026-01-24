"""
Tests for DSPy Modules.

Tests module behavior with mocked LMs.
"""

import json
from unittest.mock import Mock, patch, MagicMock

import dspy

from src.prompts.modules.router import QueryRouterModule, RoutingResult
from src.prompts.modules.orchestrator import (
    ToolOrchestratorModule,
    OrchestrationStep,
    ToolResult,
    create_dspy_tool,
)


class MockLMResponse:
    """Mock LM response for testing."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class TestQueryRouterModule:
    """Tests for QueryRouterModule."""

    def test_module_initialization(self):
        """Test module initializes correctly."""
        module = QueryRouterModule()
        assert module.router is not None
        assert hasattr(module, "forward")

    def test_routing_result_dataclass(self):
        """Test RoutingResult dataclass."""
        result = RoutingResult(
            needs_orchestration=True,
            direct_response=None,
            reason="Tools required",
        )
        assert result.needs_orchestration is True
        assert result.direct_response is None
        assert result.reason == "Tools required"

    def test_routing_result_direct_response(self):
        """Test RoutingResult with direct response."""
        result = RoutingResult(
            needs_orchestration=False,
            direct_response="Hello there!",
            reason="Simple greeting",
        )
        assert result.needs_orchestration is False
        assert result.direct_response == "Hello there!"

    @patch("src.prompts.modules.router.get_fast_lm")
    @patch("src.prompts.modules.router.dspy.context")
    def test_route_handles_errors_gracefully(self, mock_context, mock_get_lm):
        """Test route handles errors and defaults to orchestration."""
        mock_lm = MagicMock()
        mock_get_lm.return_value = mock_lm

        # Make the context manager work but have router fail
        mock_context.return_value.__enter__ = Mock()
        mock_context.return_value.__exit__ = Mock(return_value=False)

        module = QueryRouterModule()

        # Patch the internal router to raise an exception
        with patch.object(module, "router", side_effect=Exception("LLM error")):
            result = module.route("test query")

        assert result.needs_orchestration is True
        assert "failed" in result.reason.lower()


class TestToolOrchestratorModule:
    """Tests for ToolOrchestratorModule."""

    def test_module_initialization(self):
        """Test module initializes correctly."""
        module = ToolOrchestratorModule(max_steps=5, verbose=True)
        assert module.max_steps == 5
        assert module.verbose is True
        assert len(module.steps) == 0

    def test_module_with_execution_id(self):
        """Test module with execution_id."""
        module = ToolOrchestratorModule(execution_id="test-123")
        assert module.execution_id == "test-123"

    def test_orchestration_step_dataclass(self):
        """Test OrchestrationStep dataclass defaults."""
        step = OrchestrationStep(step_number=1)
        assert step.step_number == 1
        assert step.reasoning is None
        assert step.action is None
        assert step.is_final is False

    def test_orchestration_step_final(self):
        """Test OrchestrationStep with final answer."""
        step = OrchestrationStep(
            step_number=3,
            reasoning="Have enough info",
            action="Final Answer",
            is_final=True,
            final_answer="The answer is 42",
        )
        assert step.is_final is True
        assert step.final_answer == "The answer is 42"

    def test_tool_result_success(self):
        """Test ToolResult dataclass for success."""
        result = ToolResult(
            tool_name="calculate",
            success=True,
            result="2 + 2 = 4",
            raw_data={"expression": "2 + 2", "result": 4},
        )
        assert result.success is True
        assert "4" in result.result

    def test_tool_result_failure(self):
        """Test ToolResult dataclass for failure."""
        result = ToolResult(
            tool_name="web_search",
            success=False,
            result="Connection timeout",
        )
        assert result.success is False
        assert "timeout" in result.result.lower()

    def test_get_trace_empty(self):
        """Test get_trace with no steps."""
        module = ToolOrchestratorModule()
        trace = module.get_trace()
        assert trace == []

    def test_get_trace_with_steps(self):
        """Test get_trace with steps."""
        module = ToolOrchestratorModule()
        module.steps = [
            OrchestrationStep(
                step_number=1,
                reasoning="First step",
                action="calculate",
                action_input={"expression": "1 + 1"},
                observation="2",
            ),
        ]

        trace = module.get_trace()
        assert len(trace) == 1
        assert trace[0]["step"] == 1
        assert trace[0]["reasoning"] == "First step"


class TestCreateDspyTool:
    """Tests for create_dspy_tool helper function."""

    def test_create_tool_basic(self):
        """Test creating a basic DSPy tool."""

        def handler(params):
            return {"result": params.get("x", 0) * 2}

        def formatter(result):
            return f"Result: {result['result']}"

        tool = create_dspy_tool(
            name="double",
            description="Doubles a number",
            handler=handler,
            formatter=formatter,
        )

        assert tool.__name__ == "double"
        assert "Doubles" in tool.__doc__

    def test_create_tool_execution(self):
        """Test executing a created DSPy tool."""

        def handler(params):
            return {"sum": params.get("a", 0) + params.get("b", 0)}

        def formatter(result):
            return f"Sum is {result['sum']}"

        tool = create_dspy_tool(
            name="add",
            description="Adds two numbers",
            handler=handler,
            formatter=formatter,
        )

        result = tool(a=5, b=3)
        assert "8" in result

    def test_create_tool_handles_errors(self):
        """Test tool handles errors gracefully."""

        def handler(params):
            raise ValueError("Test error")

        def formatter(result):
            return str(result)

        tool = create_dspy_tool(
            name="failing_tool",
            description="A tool that fails",
            handler=handler,
            formatter=formatter,
        )

        result = tool()
        assert "error" in result.lower()


class TestDatasets:
    """Tests for dataset loading."""

    def test_load_routing_dataset(self):
        """Test loading routing dataset."""
        from src.prompts.optimization.datasets import load_routing_dataset

        examples = load_routing_dataset()
        # Should have examples from the sample data
        assert isinstance(examples, list)

    def test_load_orchestration_dataset(self):
        """Test loading orchestration dataset."""
        from src.prompts.optimization.datasets import load_orchestration_dataset

        examples = load_orchestration_dataset()
        assert isinstance(examples, list)

    def test_create_routing_example(self):
        """Test creating a routing example."""
        from src.prompts.optimization.datasets import create_routing_example

        example = create_routing_example(
            query="Hello",
            needs_tools=False,
            reasoning="Simple greeting",
            direct_answer="Hello!",
        )

        assert example.query == "Hello"
        assert example.needs_tools is False

    def test_create_orchestration_example(self):
        """Test creating an orchestration example."""
        from src.prompts.optimization.datasets import create_orchestration_example

        example = create_orchestration_example(
            question="What is 2 + 2?",
            answer="4",
            expected_keywords=["4"],
        )

        assert example.question == "What is 2 + 2?"
        assert example.answer == "4"


class TestMetrics:
    """Tests for optimization metrics."""

    def test_routing_accuracy_correct(self):
        """Test routing_accuracy with correct prediction."""
        from src.prompts.optimization.metrics import routing_accuracy

        example = dspy.Example(needs_tools=True)
        prediction = MockLMResponse(needs_tools=True)

        score = routing_accuracy(example, prediction)
        assert score == 1.0

    def test_routing_accuracy_incorrect(self):
        """Test routing_accuracy with incorrect prediction."""
        from src.prompts.optimization.metrics import routing_accuracy

        example = dspy.Example(needs_tools=True)
        prediction = MockLMResponse(needs_tools=False)

        score = routing_accuracy(example, prediction)
        assert score == 0.0

    def test_routing_accuracy_string_conversion(self):
        """Test routing_accuracy handles string values."""
        from src.prompts.optimization.metrics import routing_accuracy

        example = dspy.Example(needs_tools=True)
        prediction = MockLMResponse(needs_tools="True")

        score = routing_accuracy(example, prediction)
        assert score == 1.0

    def test_orchestration_quality_with_keywords(self):
        """Test orchestration_quality with keywords."""
        from src.prompts.optimization.metrics import orchestration_quality

        example = dspy.Example(expected_keywords=["hello", "world"])
        prediction = MockLMResponse(answer="Hello world!")

        score = orchestration_quality(example, prediction)
        assert score == 1.0

    def test_orchestration_quality_partial_keywords(self):
        """Test orchestration_quality with partial keyword match."""
        from src.prompts.optimization.metrics import orchestration_quality

        example = dspy.Example(expected_keywords=["hello", "world", "test"])
        prediction = MockLMResponse(answer="Hello!")

        score = orchestration_quality(example, prediction)
        # Only 1 out of 3 keywords matched
        assert 0.3 <= score <= 0.4


class TestCheckpointLoading:
    """Tests for optimized checkpoint loading."""

    def test_loads_checkpoint_when_configured(self, tmp_path):
        """Test that checkpoint is loaded when path is configured."""
        # Setup checkpoint directory with manifest
        checkpoint_dir = tmp_path / "orchestrator"
        checkpoint_dir.mkdir(parents=True)

        manifest = {"best": {"id": 1, "score": 1.0, "path": "checkpoint_001.json"}}
        (checkpoint_dir / "manifest.json").write_text(json.dumps(manifest))
        (checkpoint_dir / "checkpoint_001.json").write_text("{}")

        # Mock config to return our temp path
        with patch("src.prompts.modules.orchestrator.config") as mock_config:
            mock_config.dspy.optimized_prompts_path = str(tmp_path)
            mock_config.delegates = {}

            # Mock the load method
            with patch.object(ToolOrchestratorModule, "load") as mock_load:
                _module = ToolOrchestratorModule()
                mock_load.assert_called_once()

    def test_no_load_when_path_not_configured(self):
        """Test that no checkpoint is loaded when path is empty."""
        with patch("src.prompts.modules.orchestrator.config") as mock_config:
            mock_config.dspy.optimized_prompts_path = ""
            mock_config.delegates = {}

            with patch.object(ToolOrchestratorModule, "load") as mock_load:
                _module = ToolOrchestratorModule()
                mock_load.assert_not_called()

    def test_no_load_when_directory_missing(self, tmp_path):
        """Test that no checkpoint is loaded when directory doesn't exist."""
        with patch("src.prompts.modules.orchestrator.config") as mock_config:
            mock_config.dspy.optimized_prompts_path = str(tmp_path / "nonexistent")
            mock_config.delegates = {}

            with patch.object(ToolOrchestratorModule, "load") as mock_load:
                _module = ToolOrchestratorModule()
                mock_load.assert_not_called()

    def test_no_load_when_manifest_missing(self, tmp_path):
        """Test that no checkpoint is loaded when manifest is missing."""
        # Create directory but no manifest
        checkpoint_dir = tmp_path / "orchestrator"
        checkpoint_dir.mkdir(parents=True)

        with patch("src.prompts.modules.orchestrator.config") as mock_config:
            mock_config.dspy.optimized_prompts_path = str(tmp_path)
            mock_config.delegates = {}

            with patch.object(ToolOrchestratorModule, "load") as mock_load:
                _module = ToolOrchestratorModule()
                mock_load.assert_not_called()
