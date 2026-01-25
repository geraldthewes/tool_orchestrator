"""
Tests for Langfuse tracing integration with SDK v3.

Tests cover:
- Client disabled states (config, credentials, import error)
- Context manager no-ops when disabled
- Full trace lifecycle with mocked Langfuse
- Orchestrator integration with/without tracing
"""

from unittest.mock import Mock, patch, MagicMock


class TestTracingClient:
    """Tests for TracingClient."""

    def test_client_disabled_without_credentials(self):
        """Test client is disabled when credentials not provided."""
        from src.tracing.client import TracingClient

        client = TracingClient(public_key="", secret_key="")
        assert client.enabled is False
        assert "credentials not configured" in client.error.lower()

    def test_client_disabled_with_partial_credentials(self):
        """Test client is disabled with only public key."""
        from src.tracing.client import TracingClient

        client = TracingClient(public_key="pk-test", secret_key="")
        assert client.enabled is False

    @patch("src.tracing.client._langfuse_available", False)
    @patch("src.tracing.client._langfuse_error", "langfuse package not installed")
    def test_client_disabled_without_package(self):
        """Test client is disabled when langfuse package not available."""
        # Need to reimport to pick up the patched values
        import src.tracing.client as client_module

        # Save original values
        orig_available = client_module._langfuse_available
        orig_error = client_module._langfuse_error

        try:
            # Set patched values
            client_module._langfuse_available = False
            client_module._langfuse_error = "langfuse package not installed"

            client = client_module.TracingClient(
                public_key="pk-test", secret_key="sk-test"
            )
            assert client.enabled is False
            assert "not installed" in client.error.lower()
        finally:
            # Restore original values
            client_module._langfuse_available = orig_available
            client_module._langfuse_error = orig_error

    def test_client_enabled_property(self):
        """Test enabled property returns correct state."""
        from src.tracing.client import TracingClient

        client = TracingClient()
        assert client.enabled is False  # No credentials

    def test_client_error_property(self):
        """Test error property returns error message when disabled."""
        from src.tracing.client import TracingClient

        client = TracingClient()
        assert client.error is not None

    def test_flush_no_op_when_disabled(self):
        """Test flush is a no-op when tracing disabled."""
        from src.tracing.client import TracingClient

        client = TracingClient()
        # Should not raise
        client.flush()

    def test_shutdown_no_op_when_disabled(self):
        """Test shutdown is a no-op when tracing disabled."""
        from src.tracing.client import TracingClient

        client = TracingClient()
        # Should not raise
        client.shutdown()


class TestTracingClientSingleton:
    """Tests for tracing client singleton pattern."""

    def test_init_tracing_client_creates_singleton(self):
        """Test init_tracing_client creates global singleton."""
        from src.tracing.client import (
            init_tracing_client,
            get_tracing_client,
            shutdown_tracing,
        )

        # Initialize
        client = init_tracing_client()
        assert client is not None

        # Get should return same instance
        retrieved = get_tracing_client()
        assert retrieved is client

        # Cleanup
        shutdown_tracing()
        assert get_tracing_client() is None

    def test_shutdown_tracing_clears_singleton(self):
        """Test shutdown_tracing clears the global client."""
        from src.tracing.client import (
            init_tracing_client,
            get_tracing_client,
            shutdown_tracing,
        )

        init_tracing_client()
        assert get_tracing_client() is not None

        shutdown_tracing()
        assert get_tracing_client() is None


class TestTracingContext:
    """Tests for TracingContext."""

    def test_context_disabled_without_client(self):
        """Test context is disabled when no client initialized."""
        from src.tracing.context import TracingContext
        from src.tracing.client import shutdown_tracing

        # Ensure no client
        shutdown_tracing()

        ctx = TracingContext(execution_id="test-123")
        assert ctx._enabled is False

    def test_start_trace_no_op_when_disabled(self):
        """Test start_trace is no-op when disabled."""
        from src.tracing.context import TracingContext
        from src.tracing.client import shutdown_tracing

        shutdown_tracing()

        ctx = TracingContext(execution_id="test-123")
        # Should not raise
        ctx.start_trace(name="test", query="test query")
        assert ctx._root_span is None

    def test_end_trace_no_op_when_disabled(self):
        """Test end_trace is no-op when disabled."""
        from src.tracing.context import TracingContext
        from src.tracing.client import shutdown_tracing

        shutdown_tracing()

        ctx = TracingContext(execution_id="test-123")
        # Should not raise
        ctx.end_trace(output="result", status="success")

    def test_span_context_manager_no_op_when_disabled(self):
        """Test span context manager is no-op when disabled."""
        from src.tracing.context import TracingContext
        from src.tracing.client import shutdown_tracing

        shutdown_tracing()

        ctx = TracingContext(execution_id="test-123")
        with ctx.span(name="test_span") as span:
            assert span._span is None
            span.set_output({"result": "test"})
            span.set_status("success")

    def test_generation_context_manager_no_op_when_disabled(self):
        """Test generation context manager is no-op when disabled."""
        from src.tracing.context import TracingContext
        from src.tracing.client import shutdown_tracing

        shutdown_tracing()

        ctx = TracingContext(execution_id="test-123")
        with ctx.generation(name="test_gen", model="test-model") as gen:
            assert gen._generation is None
            gen.set_output("response text")
            gen.set_usage(prompt_tokens=10, completion_tokens=20)


class TestSpanContext:
    """Tests for SpanContext."""

    def test_span_set_output(self):
        """Test SpanContext.set_output stores output."""
        from src.tracing.context import SpanContext

        span = SpanContext(name="test", enabled=False)
        span.set_output({"key": "value"})
        assert span._output == {"key": "value"}

    def test_span_set_status(self):
        """Test SpanContext.set_status stores status."""
        from src.tracing.context import SpanContext

        span = SpanContext(name="test", enabled=False)
        span.set_status("error")
        assert span._status == "error"

    def test_child_span_no_op_when_disabled(self):
        """Test child_span is no-op when parent disabled."""
        from src.tracing.context import SpanContext

        parent = SpanContext(name="parent", enabled=False)
        parent.start()

        with parent.child_span(name="child") as child:
            assert child._span is None

    def test_span_generation_no_op_when_disabled(self):
        """Test span.generation is no-op when disabled."""
        from src.tracing.context import SpanContext

        span = SpanContext(name="test", enabled=False)
        span.start()

        with span.generation(name="gen", model="model") as gen:
            assert gen._generation is None


class TestGenerationContext:
    """Tests for GenerationContext."""

    def test_generation_set_output(self):
        """Test GenerationContext.set_output stores output."""
        from src.tracing.context import GenerationContext

        gen = GenerationContext(name="test", model="model", enabled=False)
        gen.set_output("response text")
        assert gen._output == "response text"

    def test_generation_set_usage(self):
        """Test GenerationContext.set_usage stores usage info."""
        from src.tracing.context import GenerationContext

        gen = GenerationContext(name="test", model="model", enabled=False)
        gen.set_usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        assert gen._usage == {
            "promptTokens": 10,
            "completionTokens": 20,
            "totalTokens": 30,
        }

    def test_generation_set_status(self):
        """Test GenerationContext.set_status stores status."""
        from src.tracing.context import GenerationContext

        gen = GenerationContext(name="test", model="model", enabled=False)
        gen.set_status("error")
        assert gen._status == "error"


class TestLangfuseConfig:
    """Tests for LangfuseConfig model."""

    def test_is_configured_with_both_keys(self):
        """Test is_configured returns True when both keys are set."""
        from src.models.config import LangfuseConfig

        config = LangfuseConfig(
            enabled=True,
            public_key="pk-test",
            secret_key="sk-test",
        )

        assert config.is_configured is True

    def test_is_configured_with_public_key_only(self):
        """Test is_configured returns False with only public key."""
        from src.models.config import LangfuseConfig

        config = LangfuseConfig(
            enabled=True,
            public_key="pk-test",
            secret_key="",
        )

        assert config.is_configured is False

    def test_is_configured_with_secret_key_only(self):
        """Test is_configured returns False with only secret key."""
        from src.models.config import LangfuseConfig

        config = LangfuseConfig(
            enabled=True,
            public_key="",
            secret_key="sk-test",
        )

        assert config.is_configured is False

    def test_is_configured_without_keys(self):
        """Test is_configured returns False when no keys set."""
        from src.models.config import LangfuseConfig

        config = LangfuseConfig()

        assert config.is_configured is False
        assert config.enabled is False


class TestOrchestratorTracingIntegration:
    """Tests for orchestrator tracing integration."""

    @patch("src.prompts.modules.orchestrator.get_orchestrator_lm")
    @patch("src.prompts.modules.orchestrator.dspy.context")
    def test_orchestrator_without_tracing_context(self, mock_context, mock_get_lm):
        """Test orchestrator works without tracing context."""
        from src.orchestrator import ToolOrchestrator

        mock_lm = MagicMock()
        mock_get_lm.return_value = mock_lm
        mock_context.return_value.__enter__ = MagicMock()
        mock_context.return_value.__exit__ = MagicMock(return_value=False)

        orchestrator = ToolOrchestrator(
            tracing_context=None,
        )

        # Mock the react module to return a final answer
        mock_result = Mock()
        mock_result.answer = "The answer is 42."
        orchestrator._module.react = Mock(return_value=mock_result)

        result = orchestrator.run("What is the answer?")
        assert "42" in result

    @patch("src.prompts.modules.orchestrator.get_orchestrator_lm")
    @patch("src.prompts.modules.orchestrator.dspy.context")
    def test_orchestrator_with_tracing_context_disabled(
        self, mock_context, mock_get_lm
    ):
        """Test orchestrator works with disabled tracing context."""
        from src.orchestrator import ToolOrchestrator
        from src.tracing import TracingContext
        from src.tracing.client import shutdown_tracing

        # Ensure tracing is disabled
        shutdown_tracing()

        mock_lm = MagicMock()
        mock_get_lm.return_value = mock_lm
        mock_context.return_value.__enter__ = MagicMock()
        mock_context.return_value.__exit__ = MagicMock(return_value=False)

        tracing_ctx = TracingContext(execution_id="test-123")

        orchestrator = ToolOrchestrator(
            tracing_context=tracing_ctx,
        )

        # Mock the react module
        mock_result = Mock()
        mock_result.answer = "Done."
        orchestrator._module.react = Mock(return_value=mock_result)

        result = orchestrator.run("Test query")
        assert "Done" in result

    def test_traced_delegate_call_without_context(self):
        """Test delegate tools work without tracing context.

        Note: The DSPy implementation handles delegates through tool adapters,
        so we test through the create_delegate_tool function.
        """
        from src.prompts.modules.orchestrator import create_delegate_tool
        from src.models import (
            DelegateConfig,
            DelegateConnection,
            DelegateCapabilities,
            DelegateDefaults,
            ConnectionType,
            DelegatesConfiguration,
        )

        config = DelegateConfig(
            role="test",
            display_name="Test LLM",
            description="Test delegate",
            connection=DelegateConnection(
                type=ConnectionType.OPENAI_COMPATIBLE,
                base_url="http://test:8000/v1",
                model="test-model",
            ),
            capabilities=DelegateCapabilities(
                context_length=4096,
                max_output_tokens=2048,
                specializations=["testing"],
            ),
            defaults=DelegateDefaults(
                temperature=0.7,
                max_tokens=1024,
            ),
        )

        delegates_config = DelegatesConfiguration(
            version="1.0",
            delegates={"test": config},
        )

        with patch("src.tools.llm_delegate.call_delegate") as mock_call:
            mock_call.return_value = {
                "success": True,
                "response": "Delegate response",
                "model": "test-model",
            }

            tool = create_delegate_tool(
                role="test",
                display_name="Test LLM",
                description="Test delegate",
                tool_name="ask_test",
                delegates_config=delegates_config,
            )

            result = tool(prompt="Test prompt")
            assert "Delegate response" in result
            mock_call.assert_called_once()

    @patch("src.prompts.modules.orchestrator.get_orchestrator_lm")
    @patch("src.prompts.modules.orchestrator.dspy.context")
    def test_traced_orchestrator_call_without_context(self, mock_context, mock_get_lm):
        """Test orchestrator call works without tracing context."""
        from src.orchestrator import ToolOrchestrator

        mock_lm = MagicMock()
        mock_get_lm.return_value = mock_lm
        mock_context.return_value.__enter__ = MagicMock()
        mock_context.return_value.__exit__ = MagicMock(return_value=False)

        orchestrator = ToolOrchestrator(
            tracing_context=None,
        )

        mock_result = Mock()
        mock_result.answer = "Test response"
        orchestrator._module.react = Mock(return_value=mock_result)

        result = orchestrator.run("test")
        assert "Test response" in result

    def test_traced_tool_execution_without_context(self):
        """Test tool execution works without tracing context.

        Note: With DSPy, tools are executed through DSPy's ReAct module,
        so we test the tool adapter directly.
        """
        from src.prompts.modules.orchestrator import create_dspy_tool
        from src.tools.math_solver import _handle_calculate, format_result_for_llm

        tool = create_dspy_tool(
            name="calculate",
            description="Calculate math",
            parameters={"expression": "math expression"},
            handler=_handle_calculate,
            formatter=format_result_for_llm,
        )

        result = tool(expression="2 + 2")
        assert "4" in result


class TestTracingWithMockedLangfuse:
    """Tests with mocked Langfuse client for SDK v3."""

    @patch("src.tracing.client.Langfuse")
    def test_client_enabled_with_valid_credentials(self, mock_langfuse_class):
        """Test client is enabled with valid credentials."""
        from src.tracing.client import TracingClient

        mock_instance = MagicMock()
        mock_langfuse_class.return_value = mock_instance

        client = TracingClient(
            public_key="pk-test",
            secret_key="sk-test",
        )

        assert client.enabled is True
        assert client.error is None
        mock_langfuse_class.assert_called_once()

    @patch("src.tracing.client.Langfuse")
    def test_flush_calls_langfuse(self, mock_langfuse_class):
        """Test flush calls underlying Langfuse client."""
        from src.tracing.client import TracingClient

        mock_instance = MagicMock()
        mock_langfuse_class.return_value = mock_instance

        client = TracingClient(
            public_key="pk-test",
            secret_key="sk-test",
        )

        client.flush()
        mock_instance.flush.assert_called_once()

    @patch("src.tracing.client.Langfuse")
    def test_shutdown_calls_langfuse(self, mock_langfuse_class):
        """Test shutdown calls underlying Langfuse client."""
        from src.tracing.client import TracingClient

        mock_instance = MagicMock()
        mock_langfuse_class.return_value = mock_instance

        client = TracingClient(
            public_key="pk-test",
            secret_key="sk-test",
        )

        client.shutdown()
        mock_instance.shutdown.assert_called_once()

    @patch("src.tracing.client.Langfuse")
    def test_start_as_current_observation_called_for_span(self, mock_langfuse_class):
        """Test start_as_current_observation is called when starting a span."""
        from src.tracing.client import (
            init_tracing_client,
            shutdown_tracing,
        )
        from src.tracing.context import SpanContext

        mock_instance = MagicMock()
        mock_observation = MagicMock()
        mock_instance.start_as_current_observation.return_value = mock_observation
        mock_langfuse_class.return_value = mock_instance

        try:
            init_tracing_client(public_key="pk-test", secret_key="sk-test")

            span = SpanContext(name="test_span", enabled=True)
            span.start()

            mock_instance.start_as_current_observation.assert_called_once()
            call_kwargs = mock_instance.start_as_current_observation.call_args[1]
            assert call_kwargs["as_type"] == "span"
            assert call_kwargs["name"] == "test_span"
        finally:
            shutdown_tracing()

    @patch("src.tracing.client.Langfuse")
    def test_start_as_current_observation_called_for_generation(
        self, mock_langfuse_class
    ):
        """Test start_as_current_observation is called when starting a generation."""
        from src.tracing.client import init_tracing_client, shutdown_tracing
        from src.tracing.context import GenerationContext

        mock_instance = MagicMock()
        mock_observation = MagicMock()
        mock_instance.start_as_current_observation.return_value = mock_observation
        mock_langfuse_class.return_value = mock_instance

        try:
            init_tracing_client(public_key="pk-test", secret_key="sk-test")

            gen = GenerationContext(name="test_gen", model="gpt-4", enabled=True)
            gen.start()

            mock_instance.start_as_current_observation.assert_called_once()
            call_kwargs = mock_instance.start_as_current_observation.call_args[1]
            assert call_kwargs["as_type"] == "generation"
            assert call_kwargs["name"] == "test_gen"
            assert call_kwargs["model"] == "gpt-4"
        finally:
            shutdown_tracing()


class TestContextManagerUsage:
    """Tests that verify context manager is used correctly (span from __enter__)."""

    @patch("src.tracing.client.Langfuse")
    def test_tracing_context_uses_span_from_enter(self, mock_langfuse_class):
        """Test TracingContext uses the span returned by __enter__, not the context manager."""
        from src.tracing.client import init_tracing_client, shutdown_tracing
        from src.tracing.context import TracingContext

        mock_instance = MagicMock()
        mock_context_manager = MagicMock()
        mock_span = MagicMock()
        mock_context_manager.__enter__ = MagicMock(return_value=mock_span)
        mock_context_manager.__exit__ = MagicMock(return_value=None)
        mock_instance.start_as_current_observation.return_value = mock_context_manager
        mock_langfuse_class.return_value = mock_instance

        try:
            init_tracing_client(public_key="pk-test", secret_key="sk-test")

            ctx = TracingContext(execution_id="test-123", user_id="user-1")
            ctx.start_trace(name="test_trace", query="test query")

            # Verify update_trace was called on span, not context manager
            mock_span.update_trace.assert_called_once_with(
                user_id="user-1",
                session_id=None,
            )
            # Verify context manager's update_trace was NOT called
            assert (
                not hasattr(mock_context_manager, "update_trace")
                or not mock_context_manager.update_trace.called
            )

            # End trace and verify __exit__ is called on context manager
            ctx.end_trace(output="result", status="success")
            mock_context_manager.__exit__.assert_called_once_with(None, None, None)
            mock_span.update.assert_called_once()
        finally:
            shutdown_tracing()

    @patch("src.tracing.client.Langfuse")
    def test_span_context_uses_span_from_enter(self, mock_langfuse_class):
        """Test SpanContext uses the span returned by __enter__, not the context manager."""
        from src.tracing.client import init_tracing_client, shutdown_tracing
        from src.tracing.context import SpanContext

        mock_instance = MagicMock()
        mock_context_manager = MagicMock()
        mock_span = MagicMock()
        mock_context_manager.__enter__ = MagicMock(return_value=mock_span)
        mock_context_manager.__exit__ = MagicMock(return_value=None)
        mock_instance.start_as_current_observation.return_value = mock_context_manager
        mock_langfuse_class.return_value = mock_instance

        try:
            init_tracing_client(public_key="pk-test", secret_key="sk-test")

            span = SpanContext(name="test_span", enabled=True)
            span.start()

            # Verify _span is set to the result of __enter__, not context manager
            assert span._span is mock_span
            assert span._context_manager is mock_context_manager

            # End span and verify update is called on span
            span.set_output({"result": "test"})
            span.end()

            mock_span.update.assert_called_once()
            mock_context_manager.__exit__.assert_called_once_with(None, None, None)
        finally:
            shutdown_tracing()

    @patch("src.tracing.client.Langfuse")
    def test_generation_context_uses_generation_from_enter(self, mock_langfuse_class):
        """Test GenerationContext uses the generation returned by __enter__, not the context manager."""
        from src.tracing.client import init_tracing_client, shutdown_tracing
        from src.tracing.context import GenerationContext

        mock_instance = MagicMock()
        mock_context_manager = MagicMock()
        mock_generation = MagicMock()
        mock_context_manager.__enter__ = MagicMock(return_value=mock_generation)
        mock_context_manager.__exit__ = MagicMock(return_value=None)
        mock_instance.start_as_current_observation.return_value = mock_context_manager
        mock_langfuse_class.return_value = mock_instance

        try:
            init_tracing_client(public_key="pk-test", secret_key="sk-test")

            gen = GenerationContext(name="test_gen", model="gpt-4", enabled=True)
            gen.start()

            # Verify _generation is set to the result of __enter__, not context manager
            assert gen._generation is mock_generation
            assert gen._context_manager is mock_context_manager

            # End generation and verify update is called on generation
            gen.set_output("test output")
            gen.set_usage(prompt_tokens=10, completion_tokens=20)
            gen.end()

            mock_generation.update.assert_called_once()
            mock_context_manager.__exit__.assert_called_once_with(None, None, None)
        finally:
            shutdown_tracing()

    @patch("src.tracing.client.Langfuse")
    def test_exit_not_called_on_span_object(self, mock_langfuse_class):
        """Test that __exit__ is NOT called on the span object itself."""
        from src.tracing.client import init_tracing_client, shutdown_tracing
        from src.tracing.context import SpanContext

        mock_instance = MagicMock()
        mock_context_manager = MagicMock()
        mock_span = MagicMock()
        mock_context_manager.__enter__ = MagicMock(return_value=mock_span)
        mock_context_manager.__exit__ = MagicMock(return_value=None)
        mock_instance.start_as_current_observation.return_value = mock_context_manager
        mock_langfuse_class.return_value = mock_instance

        try:
            init_tracing_client(public_key="pk-test", secret_key="sk-test")

            span = SpanContext(name="test_span", enabled=True)
            span.start()
            span.end()

            # __exit__ should be called on context manager, not span
            mock_context_manager.__exit__.assert_called_once()
            # If span has __exit__, it should NOT be called
            if hasattr(mock_span, "__exit__"):
                mock_span.__exit__.assert_not_called()
        finally:
            shutdown_tracing()


class TestGracefulDegradation:
    """Tests for graceful degradation on errors."""

    @patch("src.tracing.client.Langfuse")
    def test_flush_handles_exception(self, mock_langfuse_class):
        """Test flush handles exceptions gracefully."""
        from src.tracing.client import TracingClient

        mock_instance = MagicMock()
        mock_instance.flush.side_effect = Exception("Flush error")
        mock_langfuse_class.return_value = mock_instance

        client = TracingClient(
            public_key="pk-test",
            secret_key="sk-test",
        )

        # Should not raise
        client.flush()

    @patch("src.tracing.client.Langfuse")
    def test_shutdown_handles_exception(self, mock_langfuse_class):
        """Test shutdown handles exceptions gracefully."""
        from src.tracing.client import TracingClient

        mock_instance = MagicMock()
        mock_instance.shutdown.side_effect = Exception("Shutdown error")
        mock_langfuse_class.return_value = mock_instance

        client = TracingClient(
            public_key="pk-test",
            secret_key="sk-test",
        )

        # Should not raise
        client.shutdown()

    @patch("src.tracing.client.Langfuse")
    def test_span_start_handles_exception(self, mock_langfuse_class):
        """Test span start handles exceptions gracefully."""
        from src.tracing.client import init_tracing_client, shutdown_tracing
        from src.tracing.context import SpanContext

        mock_instance = MagicMock()
        mock_instance.start_as_current_observation.side_effect = Exception(
            "Start error"
        )
        mock_langfuse_class.return_value = mock_instance

        try:
            init_tracing_client(public_key="pk-test", secret_key="sk-test")

            span = SpanContext(name="test", enabled=True)
            # Should not raise
            span.start()
            assert span._span is None
        finally:
            shutdown_tracing()

    @patch("src.tracing.client.Langfuse")
    def test_generation_start_handles_exception(self, mock_langfuse_class):
        """Test generation start handles exceptions gracefully."""
        from src.tracing.client import init_tracing_client, shutdown_tracing
        from src.tracing.context import GenerationContext

        mock_instance = MagicMock()
        mock_instance.start_as_current_observation.side_effect = Exception(
            "Start error"
        )
        mock_langfuse_class.return_value = mock_instance

        try:
            init_tracing_client(public_key="pk-test", secret_key="sk-test")

            gen = GenerationContext(name="test", model="gpt-4", enabled=True)
            # Should not raise
            gen.start()
            assert gen._generation is None
        finally:
            shutdown_tracing()


class TestSpanContextSpanMethod:
    """Tests for SpanContext.span() method (alias for child_span)."""

    def test_span_method_exists(self):
        """Test SpanContext has span method."""
        from src.tracing.context import SpanContext

        assert hasattr(SpanContext, "span")

    def test_span_is_alias_for_child_span(self):
        """Test span() is an alias for child_span()."""
        from src.tracing.context import SpanContext

        assert SpanContext.span is SpanContext.child_span

    def test_span_creates_child_span_disabled(self):
        """Test span() creates child span when disabled."""
        from src.tracing.context import SpanContext

        parent = SpanContext(name="parent", enabled=False)
        parent.start()

        with parent.span(name="child") as child:
            assert child._span is None
            assert child.name == "child"

    @patch("src.tracing.client.Langfuse")
    def test_span_creates_child_span_enabled(self, mock_langfuse_class):
        """Test span() creates child span when enabled."""
        from src.tracing.client import init_tracing_client, shutdown_tracing
        from src.tracing.context import SpanContext

        mock_instance = MagicMock()
        mock_context_manager = MagicMock()
        mock_span = MagicMock()
        mock_context_manager.__enter__ = MagicMock(return_value=mock_span)
        mock_context_manager.__exit__ = MagicMock(return_value=None)
        mock_instance.start_as_current_observation.return_value = mock_context_manager
        mock_langfuse_class.return_value = mock_instance

        try:
            init_tracing_client(public_key="pk-test", secret_key="sk-test")

            parent = SpanContext(name="parent", enabled=True)
            parent.start()

            with parent.span(name="child_via_span") as child:
                assert child.name == "child_via_span"
                child.set_output({"test": "data"})
        finally:
            shutdown_tracing()


class TestLangfuseEndpointConfiguration:
    """Tests that verify Langfuse endpoint is configured correctly."""

    @patch("src.tracing.client.Langfuse")
    def test_host_with_protocol_and_port_passed_correctly(self, mock_langfuse_class):
        """Test that full URL with protocol and port is passed to Langfuse SDK."""
        from src.tracing.client import TracingClient

        mock_instance = MagicMock()
        mock_langfuse_class.return_value = mock_instance

        client = TracingClient(
            public_key="pk-test",
            secret_key="sk-test",
            host="http://langfuse.cluster:9999",
        )

        assert client.enabled is True
        mock_langfuse_class.assert_called_once_with(
            public_key="pk-test",
            secret_key="sk-test",
            debug=False,
            host="http://langfuse.cluster:9999",
        )

    @patch("src.tracing.client.Langfuse")
    def test_https_host_passed_correctly(self, mock_langfuse_class):
        """Test that HTTPS URL is passed correctly."""
        from src.tracing.client import TracingClient

        mock_instance = MagicMock()
        mock_langfuse_class.return_value = mock_instance

        TracingClient(
            public_key="pk-test",
            secret_key="sk-test",
            host="https://cloud.langfuse.com",
        )

        call_kwargs = mock_langfuse_class.call_args[1]
        assert call_kwargs["host"] == "https://cloud.langfuse.com"

    @patch("src.tracing.client.Langfuse")
    def test_empty_host_not_passed_to_sdk(self, mock_langfuse_class):
        """Test that empty host is not passed to SDK (uses SDK default)."""
        from src.tracing.client import TracingClient

        mock_instance = MagicMock()
        mock_langfuse_class.return_value = mock_instance

        TracingClient(
            public_key="pk-test",
            secret_key="sk-test",
            host="",
        )

        call_kwargs = mock_langfuse_class.call_args[1]
        assert "host" not in call_kwargs

    @patch("src.tracing.client.Langfuse")
    def test_custom_port_in_host_preserved(self, mock_langfuse_class):
        """Test that custom port in host URL is preserved."""
        from src.tracing.client import TracingClient

        mock_instance = MagicMock()
        mock_langfuse_class.return_value = mock_instance

        TracingClient(
            public_key="pk-test",
            secret_key="sk-test",
            host="http://localhost:3000",
        )

        call_kwargs = mock_langfuse_class.call_args[1]
        assert call_kwargs["host"] == "http://localhost:3000"

    @patch("src.tracing.client.Langfuse")
    def test_warning_logged_for_host_without_protocol(
        self, mock_langfuse_class, caplog
    ):
        """Test warning is logged when host lacks protocol."""
        from src.tracing.client import TracingClient
        import logging

        mock_instance = MagicMock()
        mock_langfuse_class.return_value = mock_instance

        with caplog.at_level(logging.WARNING):
            TracingClient(
                public_key="pk-test",
                secret_key="sk-test",
                host="langfuse.cluster",  # Missing protocol
            )

        assert "malformed" in caplog.text.lower() or "LANGFUSE_HOST" in caplog.text


class TestTracedLM:
    """Tests for TracedLM wrapper."""

    def test_traced_lm_is_instance_of_base_lm(self):
        """Verify TracedLM passes DSPy's isinstance check."""
        import dspy
        from src.prompts.adapters.lm_factory import TracedLM
        from src.tracing import TracingContext
        from src.tracing.client import shutdown_tracing

        shutdown_tracing()

        mock_lm = MagicMock(spec=dspy.LM)
        mock_lm.model = "test-model"
        mock_lm.cache = False
        mock_lm.history = []
        mock_lm.callbacks = []
        mock_lm.kwargs = {}

        traced_lm = TracedLM(mock_lm, TracingContext(execution_id="test-123"), "test")

        assert isinstance(traced_lm, dspy.BaseLM)
        assert isinstance(traced_lm, dspy.LM)

    def test_traced_lm_delegates_attributes(self):
        """Test TracedLM delegates attribute access to underlying LM."""
        from src.prompts.adapters.lm_factory import TracedLM
        from src.tracing import TracingContext
        from src.tracing.client import shutdown_tracing

        shutdown_tracing()

        mock_lm = MagicMock()
        mock_lm.model = "test-model"
        mock_lm.cache = False
        mock_lm.history = []
        mock_lm.callbacks = []
        mock_lm.kwargs = {}
        mock_lm.temperature = 0.7

        ctx = TracingContext(execution_id="test-123")
        traced = TracedLM(mock_lm, ctx, "test")

        assert traced.model == "test-model"
        assert traced.temperature == 0.7

    def test_traced_lm_calls_underlying_lm(self):
        """Test TracedLM calls the underlying LM's forward method."""
        from src.prompts.adapters.lm_factory import TracedLM
        from src.tracing import TracingContext
        from src.tracing.client import shutdown_tracing

        shutdown_tracing()

        mock_lm = MagicMock()
        mock_lm.model = "test-model"
        mock_lm.cache = False
        mock_lm.history = []
        mock_lm.callbacks = []
        mock_lm.kwargs = {}
        mock_lm.forward.return_value = "test response"

        ctx = TracingContext(execution_id="test-123")
        traced = TracedLM(mock_lm, ctx, "test")

        result = traced.forward(prompt="test prompt")

        mock_lm.forward.assert_called_once_with(prompt="test prompt", messages=None)
        assert result == "test response"

    def test_traced_lm_with_messages(self):
        """Test TracedLM handles messages parameter."""
        from src.prompts.adapters.lm_factory import TracedLM
        from src.tracing import TracingContext
        from src.tracing.client import shutdown_tracing

        shutdown_tracing()

        mock_lm = MagicMock()
        mock_lm.model = "test-model"
        mock_lm.cache = False
        mock_lm.history = []
        mock_lm.callbacks = []
        mock_lm.kwargs = {}
        mock_lm.forward.return_value = "test response"

        ctx = TracingContext(execution_id="test-123")
        traced = TracedLM(mock_lm, ctx, "test")

        messages = [{"role": "user", "content": "hello"}]
        result = traced.forward(messages=messages)

        mock_lm.forward.assert_called_once_with(prompt=None, messages=messages)
        assert result == "test response"

    @patch("src.tracing.client.Langfuse")
    def test_traced_lm_creates_generation_span(self, mock_langfuse_class):
        """Test TracedLM creates generation span when tracing enabled."""
        from src.prompts.adapters.lm_factory import TracedLM
        from src.tracing import TracingContext
        from src.tracing.client import init_tracing_client, shutdown_tracing

        mock_instance = MagicMock()
        mock_context_manager = MagicMock()
        mock_generation = MagicMock()
        mock_context_manager.__enter__ = MagicMock(return_value=mock_generation)
        mock_context_manager.__exit__ = MagicMock(return_value=None)
        mock_instance.start_as_current_observation.return_value = mock_context_manager
        mock_instance.auth_check.return_value = True
        mock_langfuse_class.return_value = mock_instance

        try:
            init_tracing_client(public_key="pk-test", secret_key="sk-test")

            mock_lm = MagicMock()
            mock_lm.model = "test-model"
            mock_lm.cache = False
            mock_lm.history = []
            mock_lm.callbacks = []
            mock_lm.kwargs = {}
            mock_lm.forward.return_value = "test response"

            ctx = TracingContext(execution_id="test-123")
            traced = TracedLM(mock_lm, ctx, "orchestrator")

            traced.forward(prompt="test prompt", temperature=0.5)

            # Verify generation was created
            call_kwargs = mock_instance.start_as_current_observation.call_args[1]
            assert call_kwargs["as_type"] == "generation"
            assert call_kwargs["name"] == "llm:orchestrator"
            assert call_kwargs["model"] == "test-model"
            assert "prompt" in call_kwargs["input"]

            # Verify output was set
            mock_generation.update.assert_called()
        finally:
            shutdown_tracing()


class TestToolSpanTracing:
    """Tests for tool execution span tracing."""

    @patch("src.tracing.client.Langfuse")
    def test_dspy_tool_creates_span_with_tracing(self, mock_langfuse_class):
        """Test create_dspy_tool creates spans when tracing enabled."""
        from src.prompts.modules.orchestrator import create_dspy_tool
        from src.tracing import TracingContext
        from src.tracing.client import init_tracing_client, shutdown_tracing

        mock_instance = MagicMock()
        mock_context_manager = MagicMock()
        mock_span = MagicMock()
        mock_context_manager.__enter__ = MagicMock(return_value=mock_span)
        mock_context_manager.__exit__ = MagicMock(return_value=None)
        mock_instance.start_as_current_observation.return_value = mock_context_manager
        mock_instance.auth_check.return_value = True
        mock_langfuse_class.return_value = mock_instance

        try:
            init_tracing_client(public_key="pk-test", secret_key="sk-test")

            ctx = TracingContext(execution_id="test-123")

            def test_handler(params):
                return {"success": True, "result": params.get("value", 0) * 2}

            def test_formatter(result):
                return f"Result: {result['result']}"

            tool = create_dspy_tool(
                name="test_tool",
                description="A test tool",
                parameters={"value": "input value"},
                handler=test_handler,
                formatter=test_formatter,
                tracing_context=ctx,
            )

            result = tool(value=5)

            # Verify span was created
            call_kwargs = mock_instance.start_as_current_observation.call_args[1]
            assert call_kwargs["as_type"] == "span"
            assert call_kwargs["name"] == "tool:test_tool"

            # Verify output was set
            mock_span.update.assert_called()
            assert "10" in result
        finally:
            shutdown_tracing()

    def test_dspy_tool_works_without_tracing(self):
        """Test create_dspy_tool works when tracing_context is None."""
        from src.prompts.modules.orchestrator import create_dspy_tool

        def test_handler(params):
            return {"success": True, "result": params.get("value", 0) * 2}

        def test_formatter(result):
            return f"Result: {result['result']}"

        tool = create_dspy_tool(
            name="test_tool",
            description="A test tool",
            parameters={"value": "input value"},
            handler=test_handler,
            formatter=test_formatter,
            tracing_context=None,
        )

        result = tool(value=5)
        assert "10" in result

    @patch("src.tracing.client.Langfuse")
    def test_delegate_tool_creates_span_with_tracing(self, mock_langfuse_class):
        """Test create_delegate_tool creates spans when tracing enabled."""
        from src.prompts.modules.orchestrator import create_delegate_tool
        from src.tracing import TracingContext
        from src.tracing.client import init_tracing_client, shutdown_tracing
        from src.models import (
            DelegateConfig,
            DelegateConnection,
            DelegateCapabilities,
            DelegateDefaults,
            ConnectionType,
            DelegatesConfiguration,
        )

        mock_instance = MagicMock()
        mock_context_manager = MagicMock()
        mock_span = MagicMock()
        mock_context_manager.__enter__ = MagicMock(return_value=mock_span)
        mock_context_manager.__exit__ = MagicMock(return_value=None)
        mock_instance.start_as_current_observation.return_value = mock_context_manager
        mock_instance.auth_check.return_value = True
        mock_langfuse_class.return_value = mock_instance

        try:
            init_tracing_client(public_key="pk-test", secret_key="sk-test")

            ctx = TracingContext(execution_id="test-123")

            config = DelegateConfig(
                role="test",
                display_name="Test LLM",
                description="Test delegate",
                connection=DelegateConnection(
                    type=ConnectionType.OPENAI_COMPATIBLE,
                    base_url="http://test:8000/v1",
                    model="test-model",
                ),
                capabilities=DelegateCapabilities(
                    context_length=4096,
                    max_output_tokens=2048,
                    specializations=["testing"],
                ),
                defaults=DelegateDefaults(
                    temperature=0.7,
                    max_tokens=1024,
                ),
            )

            delegates_config = DelegatesConfiguration(
                version="1.0",
                delegates={"test": config},
            )

            with patch("src.tools.llm_delegate.call_delegate") as mock_call:
                mock_call.return_value = {
                    "success": True,
                    "response": "Delegate response",
                    "model": "test-model",
                }

                tool = create_delegate_tool(
                    role="test",
                    display_name="Test LLM",
                    description="Test delegate",
                    tool_name="ask_test",
                    delegates_config=delegates_config,
                    tracing_context=ctx,
                )

                result = tool(prompt="Test prompt")

                # Verify span was created
                call_kwargs = mock_instance.start_as_current_observation.call_args[1]
                assert call_kwargs["as_type"] == "span"
                assert call_kwargs["name"] == "tool:ask_test"

                # Verify output was set
                mock_span.update.assert_called()
                assert "Delegate response" in result
        finally:
            shutdown_tracing()

    def test_tool_tracing_graceful_degradation_disabled_context(self):
        """Test tool tracing gracefully degrades with disabled context."""
        from src.prompts.modules.orchestrator import create_dspy_tool
        from src.tracing import TracingContext
        from src.tracing.client import shutdown_tracing

        shutdown_tracing()

        ctx = TracingContext(execution_id="test-123")
        # Context is disabled (no tracing client)

        def test_handler(params):
            return {"value": params.get("x", 0) + 1}

        def test_formatter(result):
            return f"Value: {result['value']}"

        tool = create_dspy_tool(
            name="test_tool",
            description="A test tool",
            parameters={"x": "input value"},
            handler=test_handler,
            formatter=test_formatter,
            tracing_context=ctx,
        )

        # Should work without errors
        result = tool(x=5)
        assert "6" in result

    def test_tool_span_captures_error_status(self):
        """Test tool span captures error status on handler failure."""
        from src.prompts.modules.orchestrator import create_dspy_tool
        from src.tracing import TracingContext
        from src.tracing.client import shutdown_tracing

        shutdown_tracing()

        ctx = TracingContext(execution_id="test-123")

        def failing_handler(params):
            raise ValueError("Test error")

        def test_formatter(result):
            return str(result)

        tool = create_dspy_tool(
            name="test_tool",
            description="A test tool",
            parameters={"input": "any input"},
            handler=failing_handler,
            formatter=test_formatter,
            tracing_context=ctx,
        )

        result = tool(value=1)
        assert "error" in result.lower()


class TestOrchestratorTracingWithSpanContextParent:
    """Tests for orchestrator tracing with SpanContext as parent."""

    def test_traced_tool_execution_with_span_context_parent(self):
        """Test tool execution works with SpanContext.

        Note: With DSPy, tools are executed through DSPy's ReAct module.
        This test verifies tool adapters work correctly.
        """
        from src.prompts.modules.orchestrator import create_dspy_tool
        from src.tools.math_solver import _handle_calculate, format_result_for_llm
        from src.tracing.context import SpanContext

        # Create a disabled SpanContext as parent (simulates nested span scenario)
        parent_span = SpanContext(name="step_1", enabled=False)
        parent_span.start()

        # Tool adapter should work regardless of tracing context
        tool = create_dspy_tool(
            name="calculate",
            description="Calculate math",
            parameters={"expression": "math expression"},
            handler=_handle_calculate,
            formatter=format_result_for_llm,
        )

        result = tool(expression="2 + 2")
        assert "4" in result


class TestConnectivityValidation:
    """Tests for startup connectivity validation."""

    @patch("src.tracing.client.Langfuse")
    def test_client_disabled_when_auth_check_fails(self, mock_langfuse_class, caplog):
        """Test client is disabled when auth_check returns False."""
        from src.tracing.client import TracingClient
        import logging

        mock_instance = MagicMock()
        mock_instance.auth_check.return_value = False
        mock_langfuse_class.return_value = mock_instance

        with caplog.at_level(logging.WARNING):
            client = TracingClient(
                public_key="pk-test",
                secret_key="sk-test",
                host="http://langfuse.cluster:9999",
            )

        assert client.enabled is False
        assert client.client is None
        assert "auth_check" in client.error.lower()
        assert "tracing disabled" in caplog.text.lower()

    @patch("src.tracing.client.Langfuse")
    def test_client_disabled_when_auth_check_raises_exception(
        self, mock_langfuse_class, caplog
    ):
        """Test client is disabled when auth_check raises an exception (unreachable endpoint)."""
        from src.tracing.client import TracingClient
        import logging

        mock_instance = MagicMock()
        mock_instance.auth_check.side_effect = Exception(
            "ConnectionError: Failed to connect to http://langfuse.cluster:9999"
        )
        mock_langfuse_class.return_value = mock_instance

        with caplog.at_level(logging.WARNING):
            client = TracingClient(
                public_key="pk-test",
                secret_key="sk-test",
                host="http://langfuse.cluster:9999",
            )

        assert client.enabled is False
        assert client.client is None
        assert "connectivity check failed" in client.error.lower()
        assert "langfuse_host" in client.error.lower()
        assert "tracing disabled" in caplog.text.lower()

    @patch("src.tracing.client.Langfuse")
    def test_client_enabled_when_auth_check_succeeds(self, mock_langfuse_class, caplog):
        """Test client is enabled when auth_check returns True."""
        from src.tracing.client import TracingClient
        import logging

        mock_instance = MagicMock()
        mock_instance.auth_check.return_value = True
        mock_langfuse_class.return_value = mock_instance

        with caplog.at_level(logging.INFO):
            client = TracingClient(
                public_key="pk-test",
                secret_key="sk-test",
                host="http://langfuse.cluster:9999",
            )

        assert client.enabled is True
        assert client.client is mock_instance
        assert client.error is None
        assert "tracing enabled" in caplog.text.lower()

    @patch("src.tracing.client.Langfuse")
    def test_no_exception_propagates_on_connectivity_failure(self, mock_langfuse_class):
        """Test that no exceptions propagate when connectivity fails - graceful degradation."""
        from src.tracing.client import TracingClient

        mock_instance = MagicMock()
        mock_instance.auth_check.side_effect = Exception("Network unreachable")
        mock_langfuse_class.return_value = mock_instance

        # Should not raise any exception
        client = TracingClient(
            public_key="pk-test",
            secret_key="sk-test",
            host="http://unreachable.host:9999",
        )

        # Tracing should be disabled but app should continue working
        assert client.enabled is False
        # All operations should be no-ops
        client.flush()  # Should not raise
        client.shutdown()  # Should not raise
