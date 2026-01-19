"""
Tests for Langfuse tracing integration with SDK v3.

Tests cover:
- Client disabled states (config, credentials, import error)
- Context manager no-ops when disabled
- Full trace lifecycle with mocked Langfuse
- Orchestrator integration with/without tracing
"""

import pytest
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
        import importlib
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
    """Tests for LangfuseConfig."""

    def test_config_enabled_with_both_keys(self):
        """Test config.enabled returns True when both keys set."""
        import os
        from importlib import reload

        # Set env vars temporarily
        os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-test"
        os.environ["LANGFUSE_SECRET_KEY"] = "sk-test"

        try:
            import src.config as config_module

            reload(config_module)

            assert config_module.LangfuseConfig().enabled is True
        finally:
            os.environ.pop("LANGFUSE_PUBLIC_KEY", None)
            os.environ.pop("LANGFUSE_SECRET_KEY", None)
            reload(config_module)

    def test_config_disabled_without_keys(self):
        """Test config.enabled returns False when keys not set."""
        import os
        from importlib import reload

        # Ensure env vars are not set
        os.environ.pop("LANGFUSE_PUBLIC_KEY", None)
        os.environ.pop("LANGFUSE_SECRET_KEY", None)

        import src.config as config_module

        reload(config_module)

        assert config_module.LangfuseConfig().enabled is False


class TestOrchestratorTracingIntegration:
    """Tests for orchestrator tracing integration."""

    def test_orchestrator_without_tracing_context(self):
        """Test orchestrator works without tracing context."""
        from src.orchestrator import ToolOrchestrator
        from unittest.mock import Mock

        mock_client = Mock()
        mock_client.call_orchestrator.return_value = {
            "success": True,
            "response": """
**Thought**: I know this.
**Action**: Final Answer
**Action Input**: The answer is 42.
""",
        }

        orchestrator = ToolOrchestrator(
            llm_client=mock_client,
            tracing_context=None,
        )

        result = orchestrator.run("What is the answer?")
        assert "42" in result

    def test_orchestrator_with_tracing_context_disabled(self):
        """Test orchestrator works with disabled tracing context."""
        from src.orchestrator import ToolOrchestrator
        from src.tracing import TracingContext
        from src.tracing.client import shutdown_tracing
        from unittest.mock import Mock

        # Ensure tracing is disabled
        shutdown_tracing()

        mock_client = Mock()
        mock_client.call_orchestrator.return_value = {
            "success": True,
            "response": """
**Thought**: Simple answer.
**Action**: Final Answer
**Action Input**: Done.
""",
        }

        tracing_ctx = TracingContext(execution_id="test-123")

        orchestrator = ToolOrchestrator(
            llm_client=mock_client,
            tracing_context=tracing_ctx,
        )

        result = orchestrator.run("Test query")
        assert "Done" in result

    @patch("src.orchestrator.call_delegate")
    def test_traced_delegate_call_without_context(self, mock_call_delegate):
        """Test _traced_delegate_call works without tracing context."""
        from src.orchestrator import ToolOrchestrator
        from src.models import (
            DelegateConfig,
            DelegateConnection,
            DelegateCapabilities,
            DelegateDefaults,
            ConnectionType,
        )

        mock_call_delegate.return_value = {
            "success": True,
            "response": "Delegate response",
        }

        mock_client = Mock()
        orchestrator = ToolOrchestrator(
            llm_client=mock_client,
            tracing_context=None,
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

        result = orchestrator._traced_delegate_call(
            config=config,
            prompt="Test prompt",
            temperature=0.7,
            max_tokens=100,
        )

        assert result["success"] is True
        mock_call_delegate.assert_called_once()

    def test_traced_orchestrator_call_without_context(self):
        """Test _traced_orchestrator_call works without tracing context."""
        from src.orchestrator import ToolOrchestrator
        from unittest.mock import Mock

        mock_client = Mock()
        mock_client.call_orchestrator.return_value = {
            "success": True,
            "response": "Test response",
        }

        orchestrator = ToolOrchestrator(
            llm_client=mock_client,
            tracing_context=None,
        )

        result = orchestrator._traced_orchestrator_call(
            messages=[{"role": "user", "content": "test"}],
            temperature=0.7,
            max_tokens=100,
            step_number=1,
        )

        assert result["success"] is True
        mock_client.call_orchestrator.assert_called_once()

    def test_traced_tool_execution_without_context(self):
        """Test _traced_tool_execution works without tracing context."""
        from src.orchestrator import ToolOrchestrator
        from unittest.mock import Mock

        mock_client = Mock()
        orchestrator = ToolOrchestrator(
            llm_client=mock_client,
            tracing_context=None,
        )

        result = orchestrator._traced_tool_execution(
            tool_name="calculate",
            params={"expression": "2 + 2"},
            step_number=1,
        )

        assert result.success is True
        assert "4" in result.result


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
        from src.tracing.client import TracingClient, init_tracing_client, shutdown_tracing
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
    def test_start_as_current_observation_called_for_generation(self, mock_langfuse_class):
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
            assert not hasattr(mock_context_manager, 'update_trace') or \
                not mock_context_manager.update_trace.called

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
            if hasattr(mock_span, '__exit__'):
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
        mock_instance.start_as_current_observation.side_effect = Exception("Start error")
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
        mock_instance.start_as_current_observation.side_effect = Exception("Start error")
        mock_langfuse_class.return_value = mock_instance

        try:
            init_tracing_client(public_key="pk-test", secret_key="sk-test")

            gen = GenerationContext(name="test", model="gpt-4", enabled=True)
            # Should not raise
            gen.start()
            assert gen._generation is None
        finally:
            shutdown_tracing()
