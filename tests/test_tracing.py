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
    """Tests for orchestrator tracing integration with the new loop."""

    @patch("src.orchestration.loop.OpenAI")
    def test_orchestrator_without_tracing_context(self, mock_openai_cls):
        """Test orchestrator works without tracing context."""
        import json

        from src.orchestrator import ToolOrchestrator

        mock_client = Mock()
        mock_openai_cls.return_value = mock_client

        # Return text with <tool_call> XML for the answer tool
        tc_json = json.dumps(
            {"name": "answer", "arguments": {"content": "The answer is 42."}}
        )
        content = f"<think>Simple</think>\n\n<tool_call>\n{tc_json}\n</tool_call>"

        msg = Mock()
        msg.content = content

        response = Mock()
        response.choices = [Mock(message=msg)]
        response.usage = None
        mock_client.chat.completions.create.return_value = response

        orchestrator = ToolOrchestrator(tracing_context=None)
        result = orchestrator.run("What is the answer?")
        assert "42" in result

    @patch("src.orchestration.loop.OpenAI")
    def test_orchestrator_with_tracing_context_disabled(self, mock_openai_cls):
        """Test orchestrator works with disabled tracing context."""
        import json

        from src.orchestrator import ToolOrchestrator
        from src.tracing import TracingContext
        from src.tracing.client import shutdown_tracing

        shutdown_tracing()

        mock_client = Mock()
        mock_openai_cls.return_value = mock_client

        # Return text with <tool_call> XML for the answer tool
        tc_json = json.dumps({"name": "answer", "arguments": {"content": "Done."}})
        content = f"<tool_call>\n{tc_json}\n</tool_call>"

        msg = Mock()
        msg.content = content

        response = Mock()
        response.choices = [Mock(message=msg)]
        response.usage = None
        mock_client.chat.completions.create.return_value = response

        tracing_ctx = TracingContext(execution_id="test-123")
        orchestrator = ToolOrchestrator(tracing_context=tracing_ctx)
        result = orchestrator.run("Test query")
        assert "Done" in result


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


class TestTracedLMTokenExtraction:
    """Tests for TracedLM token usage extraction."""

    @patch("src.tracing.client.Langfuse")
    def test_traced_lm_captures_dict_usage(self, mock_langfuse_class):
        """Test TracedLM extracts token counts from dict-style usage."""
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

            # Create result with dict-style usage
            mock_result = MagicMock()
            mock_result.usage = {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            }
            mock_lm.forward.return_value = mock_result

            ctx = TracingContext(execution_id="test-123")
            traced = TracedLM(mock_lm, ctx, "orchestrator")

            traced.forward(prompt="test prompt")

            # Verify generation.update was called with usage
            mock_generation.update.assert_called()
            update_kwargs = mock_generation.update.call_args[1]
            assert "usage" in update_kwargs
            assert update_kwargs["usage"]["promptTokens"] == 100
            assert update_kwargs["usage"]["completionTokens"] == 50
            assert update_kwargs["usage"]["totalTokens"] == 150
        finally:
            shutdown_tracing()

    @patch("src.tracing.client.Langfuse")
    def test_traced_lm_captures_object_usage(self, mock_langfuse_class):
        """Test TracedLM extracts token counts from object-style usage (LiteLLM)."""
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

            # Create result with object-style usage (like LiteLLM Usage class)
            mock_result = MagicMock()
            mock_usage = MagicMock()
            mock_usage.prompt_tokens = 200
            mock_usage.completion_tokens = 100
            mock_usage.total_tokens = 300
            mock_result.usage = mock_usage
            mock_lm.forward.return_value = mock_result

            ctx = TracingContext(execution_id="test-123")
            traced = TracedLM(mock_lm, ctx, "orchestrator")

            traced.forward(prompt="test prompt")

            # Verify generation.update was called with usage
            mock_generation.update.assert_called()
            update_kwargs = mock_generation.update.call_args[1]
            assert "usage" in update_kwargs
            assert update_kwargs["usage"]["promptTokens"] == 200
            assert update_kwargs["usage"]["completionTokens"] == 100
            assert update_kwargs["usage"]["totalTokens"] == 300
        finally:
            shutdown_tracing()

    @patch("src.tracing.client.Langfuse")
    def test_traced_lm_skips_empty_usage(self, mock_langfuse_class):
        """Test TracedLM doesn't set usage when all values are None."""
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

            # Create result with empty dict usage
            mock_result = MagicMock()
            mock_result.usage = {}
            mock_lm.forward.return_value = mock_result

            ctx = TracingContext(execution_id="test-123")
            traced = TracedLM(mock_lm, ctx, "orchestrator")

            traced.forward(prompt="test prompt")

            # Verify generation.update was called but without usage
            mock_generation.update.assert_called()
            update_kwargs = mock_generation.update.call_args[1]
            # Usage should not be set when all values are None
            assert "usage" not in update_kwargs or update_kwargs.get("usage") is None
        finally:
            shutdown_tracing()


class TestTokenAwareLMUsagePassthrough:
    """Tests for TokenAwareLM usage data passthrough."""

    def test_token_aware_lm_preserves_usage(self):
        """Test TokenAwareLM preserves usage from underlying LM result."""
        from src.prompts.adapters.lm_factory import TokenAwareLM

        mock_lm = MagicMock()
        mock_lm.model = "test-model"
        mock_lm.cache = False
        mock_lm.history = []
        mock_lm.callbacks = []
        mock_lm.kwargs = {"max_tokens": 1000}

        # Create result with usage
        mock_result = MagicMock()
        mock_result.usage = {
            "prompt_tokens": 50,
            "completion_tokens": 25,
            "total_tokens": 75,
        }
        mock_lm.forward.return_value = mock_result

        wrapper = TokenAwareLM(mock_lm, context_length=4096)
        result = wrapper.forward(prompt="test")

        assert result.usage["prompt_tokens"] == 50
        assert result.usage["completion_tokens"] == 25

    def test_token_aware_lm_recovers_usage_from_history(self):
        """Test TokenAwareLM recovers usage from LM history when result has empty usage."""
        from src.prompts.adapters.lm_factory import TokenAwareLM

        mock_lm = MagicMock()
        mock_lm.model = "test-model"
        mock_lm.cache = False
        mock_lm.callbacks = []
        mock_lm.kwargs = {"max_tokens": 1000}

        # Create history entry with valid usage
        history_entry = MagicMock()
        history_entry.usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }
        mock_lm.history = [history_entry]

        # Create result with empty usage (simulates caching issue)
        mock_result = MagicMock()
        mock_result.usage = {}
        mock_lm.forward.return_value = mock_result

        wrapper = TokenAwareLM(mock_lm, context_length=4096)
        result = wrapper.forward(prompt="test")

        # Usage should be recovered from history
        assert result.usage == history_entry.usage


class TestExplicitTraceContextPropagation:
    """Tests for explicit TraceContext propagation to child observations."""

    @patch("src.tracing.client.Langfuse")
    def test_tracing_context_get_trace_context_returns_valid_context(
        self, mock_langfuse_class
    ):
        """Test TracingContext.get_trace_context() returns valid TraceContext after start_trace()."""
        from src.tracing.client import init_tracing_client, shutdown_tracing
        from src.tracing.context import TracingContext

        mock_instance = MagicMock()
        mock_context_manager = MagicMock()
        mock_span = MagicMock()
        mock_span.trace_id = "test-trace-id-123"
        mock_span.id = "test-span-id-456"
        mock_context_manager.__enter__ = MagicMock(return_value=mock_span)
        mock_context_manager.__exit__ = MagicMock(return_value=None)
        mock_instance.start_as_current_observation.return_value = mock_context_manager
        mock_instance.auth_check.return_value = True
        mock_langfuse_class.return_value = mock_instance

        try:
            init_tracing_client(public_key="pk-test", secret_key="sk-test")

            ctx = TracingContext(execution_id="test-123")
            ctx.start_trace(name="test_trace", query="test query")

            trace_context = ctx.get_trace_context()

            # TraceContext is a TypedDict, so access as dict
            assert trace_context is not None
            assert trace_context["trace_id"] == "test-trace-id-123"
            assert trace_context["parent_span_id"] == "test-span-id-456"
        finally:
            shutdown_tracing()

    @patch("src.tracing.client.Langfuse")
    def test_tracing_context_get_trace_context_returns_none_before_start(
        self, mock_langfuse_class
    ):
        """Test TracingContext.get_trace_context() returns None before start_trace()."""
        from src.tracing.client import init_tracing_client, shutdown_tracing
        from src.tracing.context import TracingContext

        mock_instance = MagicMock()
        mock_instance.auth_check.return_value = True
        mock_langfuse_class.return_value = mock_instance

        try:
            init_tracing_client(public_key="pk-test", secret_key="sk-test")

            ctx = TracingContext(execution_id="test-123")
            # Don't call start_trace()

            trace_context = ctx.get_trace_context()

            assert trace_context is None
        finally:
            shutdown_tracing()

    @patch("src.tracing.client.Langfuse")
    def test_span_receives_trace_context_from_tracing_context(
        self, mock_langfuse_class
    ):
        """Test span created via TracingContext.span() receives trace_context."""
        from src.tracing.client import init_tracing_client, shutdown_tracing
        from src.tracing.context import TracingContext

        mock_instance = MagicMock()
        mock_context_manager = MagicMock()
        mock_span = MagicMock()
        mock_span.trace_id = "test-trace-id"
        mock_span.id = "root-span-id"
        mock_context_manager.__enter__ = MagicMock(return_value=mock_span)
        mock_context_manager.__exit__ = MagicMock(return_value=None)
        mock_instance.start_as_current_observation.return_value = mock_context_manager
        mock_instance.auth_check.return_value = True
        mock_langfuse_class.return_value = mock_instance

        try:
            init_tracing_client(public_key="pk-test", secret_key="sk-test")

            ctx = TracingContext(execution_id="test-123")
            ctx.start_trace(name="test_trace")

            # Reset mock to capture span creation call
            mock_instance.start_as_current_observation.reset_mock()
            mock_instance.start_as_current_observation.return_value = (
                mock_context_manager
            )

            with ctx.span(name="child_span") as _child:
                pass

            # Verify trace_context was passed to the child span (TraceContext is a TypedDict)
            call_kwargs = mock_instance.start_as_current_observation.call_args[1]
            assert "trace_context" in call_kwargs
            assert call_kwargs["trace_context"] is not None
            assert call_kwargs["trace_context"]["trace_id"] == "test-trace-id"
            assert call_kwargs["trace_context"]["parent_span_id"] == "root-span-id"
        finally:
            shutdown_tracing()

    @patch("src.tracing.client.Langfuse")
    def test_generation_receives_trace_context_from_tracing_context(
        self, mock_langfuse_class
    ):
        """Test generation created via TracingContext.generation() receives trace_context."""
        from src.tracing.client import init_tracing_client, shutdown_tracing
        from src.tracing.context import TracingContext

        mock_instance = MagicMock()
        mock_context_manager = MagicMock()
        mock_span = MagicMock()
        mock_span.trace_id = "test-trace-id"
        mock_span.id = "root-span-id"
        mock_context_manager.__enter__ = MagicMock(return_value=mock_span)
        mock_context_manager.__exit__ = MagicMock(return_value=None)
        mock_instance.start_as_current_observation.return_value = mock_context_manager
        mock_instance.auth_check.return_value = True
        mock_langfuse_class.return_value = mock_instance

        try:
            init_tracing_client(public_key="pk-test", secret_key="sk-test")

            ctx = TracingContext(execution_id="test-123")
            ctx.start_trace(name="test_trace")

            # Reset mock to capture generation creation call
            mock_instance.start_as_current_observation.reset_mock()
            mock_generation = MagicMock()
            mock_gen_context_manager = MagicMock()
            mock_gen_context_manager.__enter__ = MagicMock(return_value=mock_generation)
            mock_gen_context_manager.__exit__ = MagicMock(return_value=None)
            mock_instance.start_as_current_observation.return_value = (
                mock_gen_context_manager
            )

            with ctx.generation(name="llm_call", model="gpt-4") as _gen:
                pass

            # Verify trace_context was passed to the generation (TraceContext is a TypedDict)
            call_kwargs = mock_instance.start_as_current_observation.call_args[1]
            assert "trace_context" in call_kwargs
            assert call_kwargs["trace_context"] is not None
            assert call_kwargs["trace_context"]["trace_id"] == "test-trace-id"
        finally:
            shutdown_tracing()

    @patch("src.tracing.client.Langfuse")
    def test_child_span_receives_parent_span_as_parent(self, mock_langfuse_class):
        """Test child span created via SpanContext.child_span() uses parent span as parent."""
        from src.tracing.client import init_tracing_client, shutdown_tracing
        from src.tracing.context import TracingContext

        mock_instance = MagicMock()

        # First call creates root span
        mock_root_span = MagicMock()
        mock_root_span.trace_id = "test-trace-id"
        mock_root_span.id = "root-span-id"
        mock_root_context_manager = MagicMock()
        mock_root_context_manager.__enter__ = MagicMock(return_value=mock_root_span)
        mock_root_context_manager.__exit__ = MagicMock(return_value=None)

        # Second call creates first child span
        mock_parent_span = MagicMock()
        mock_parent_span.trace_id = "test-trace-id"
        mock_parent_span.id = "parent-span-id"
        mock_parent_context_manager = MagicMock()
        mock_parent_context_manager.__enter__ = MagicMock(return_value=mock_parent_span)
        mock_parent_context_manager.__exit__ = MagicMock(return_value=None)

        # Third call creates grandchild span
        mock_child_span = MagicMock()
        mock_child_span.trace_id = "test-trace-id"
        mock_child_span.id = "child-span-id"
        mock_child_context_manager = MagicMock()
        mock_child_context_manager.__enter__ = MagicMock(return_value=mock_child_span)
        mock_child_context_manager.__exit__ = MagicMock(return_value=None)

        mock_instance.start_as_current_observation.side_effect = [
            mock_root_context_manager,
            mock_parent_context_manager,
            mock_child_context_manager,
        ]
        mock_instance.auth_check.return_value = True
        mock_langfuse_class.return_value = mock_instance

        try:
            init_tracing_client(public_key="pk-test", secret_key="sk-test")

            ctx = TracingContext(execution_id="test-123")
            ctx.start_trace(name="test_trace")

            with ctx.span(name="parent_span") as parent:
                with parent.child_span(name="child_span") as _child:
                    pass

            # Get the call for the grandchild span (third call)
            calls = mock_instance.start_as_current_observation.call_args_list
            assert len(calls) == 3

            grandchild_call_kwargs = calls[2][1]
            assert "trace_context" in grandchild_call_kwargs
            # The grandchild should have the parent_span as its parent (TraceContext is a TypedDict)
            assert (
                grandchild_call_kwargs["trace_context"]["trace_id"] == "test-trace-id"
            )
            assert (
                grandchild_call_kwargs["trace_context"]["parent_span_id"]
                == "parent-span-id"
            )
        finally:
            shutdown_tracing()

    @patch("src.tracing.client.Langfuse")
    def test_generation_from_span_uses_span_as_parent(self, mock_langfuse_class):
        """Test generation created via SpanContext.generation() uses span as parent."""
        from src.tracing.client import init_tracing_client, shutdown_tracing
        from src.tracing.context import TracingContext

        mock_instance = MagicMock()

        # First call creates root span
        mock_root_span = MagicMock()
        mock_root_span.trace_id = "test-trace-id"
        mock_root_span.id = "root-span-id"
        mock_root_context_manager = MagicMock()
        mock_root_context_manager.__enter__ = MagicMock(return_value=mock_root_span)
        mock_root_context_manager.__exit__ = MagicMock(return_value=None)

        # Second call creates parent span
        mock_parent_span = MagicMock()
        mock_parent_span.trace_id = "test-trace-id"
        mock_parent_span.id = "parent-span-id"
        mock_parent_context_manager = MagicMock()
        mock_parent_context_manager.__enter__ = MagicMock(return_value=mock_parent_span)
        mock_parent_context_manager.__exit__ = MagicMock(return_value=None)

        # Third call creates generation
        mock_generation = MagicMock()
        mock_gen_context_manager = MagicMock()
        mock_gen_context_manager.__enter__ = MagicMock(return_value=mock_generation)
        mock_gen_context_manager.__exit__ = MagicMock(return_value=None)

        mock_instance.start_as_current_observation.side_effect = [
            mock_root_context_manager,
            mock_parent_context_manager,
            mock_gen_context_manager,
        ]
        mock_instance.auth_check.return_value = True
        mock_langfuse_class.return_value = mock_instance

        try:
            init_tracing_client(public_key="pk-test", secret_key="sk-test")

            ctx = TracingContext(execution_id="test-123")
            ctx.start_trace(name="test_trace")

            with ctx.span(name="parent_span") as parent:
                with parent.generation(name="llm_call", model="gpt-4") as _gen:
                    pass

            # Get the call for the generation (third call)
            calls = mock_instance.start_as_current_observation.call_args_list
            assert len(calls) == 3

            gen_call_kwargs = calls[2][1]
            assert "trace_context" in gen_call_kwargs
            assert gen_call_kwargs["as_type"] == "generation"
            # The generation should have the parent_span as its parent (TraceContext is a TypedDict)
            assert gen_call_kwargs["trace_context"]["trace_id"] == "test-trace-id"
            assert (
                gen_call_kwargs["trace_context"]["parent_span_id"] == "parent-span-id"
            )
        finally:
            shutdown_tracing()

    def test_span_context_get_child_trace_context_without_parent(self):
        """Test SpanContext._get_child_trace_context() returns None without parent context."""
        from src.tracing.context import SpanContext

        span = SpanContext(name="test", enabled=False)
        span.start()

        result = span._get_child_trace_context()
        assert result is None

    def test_get_trace_context_disabled_returns_none(self):
        """Test get_trace_context returns None when tracing disabled."""
        from src.tracing.context import TracingContext
        from src.tracing.client import shutdown_tracing

        shutdown_tracing()

        ctx = TracingContext(execution_id="test-123")
        ctx.start_trace(name="test_trace")

        result = ctx.get_trace_context()
        assert result is None


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
