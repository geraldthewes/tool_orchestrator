"""
Request-scoped tracing context using Langfuse SDK v3.

Uses explicit trace_context propagation to ensure proper parent-child linking.
Each span/generation passes its parent's trace_id and span_id to children,
ensuring correct nesting regardless of OTEL context state.

Provides context managers for managing trace lifecycle, spans, and generations
with automatic graceful degradation when tracing is disabled.
"""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generator, Optional

from .client import get_tracing_client

if TYPE_CHECKING:
    from langfuse.types import TraceContext

logger = logging.getLogger(__name__)


@dataclass
class TracingContext:
    """
    Request-scoped tracing context using Langfuse SDK v3.

    Manages the lifecycle of a trace for a single API request,
    including nested spans and generations. Relies on SDK v3's OTEL
    automatic context propagation for parent-child linking via
    nested `with` blocks.
    """

    execution_id: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    _context_manager: Any = field(default=None, repr=False)
    _root_span: Any = field(default=None, repr=False)
    _enabled: bool = field(default=False, repr=False)
    _start_time: float = field(default_factory=time.time, repr=False)
    _trace_id: Optional[str] = field(default=None, repr=False)
    _root_span_id: Optional[str] = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize tracing state based on client availability."""
        client = get_tracing_client()
        self._enabled = client is not None and client.enabled

    def start_trace(
        self,
        name: str = "api_request",
        query: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Start a new trace for this request.

        In v3, we create a root span and set trace attributes on it.

        Args:
            name: Name for the trace
            query: Input query to record
            metadata: Additional metadata
        """
        if not self._enabled:
            logger.debug(
                f"[{self.execution_id}] start_trace skipped: tracing disabled at context creation"
            )
            return

        client = get_tracing_client()
        if not client or not client.client:
            logger.debug(
                f"[{self.execution_id}] start_trace skipped: client not available"
            )
            return

        logger.debug(f"[{self.execution_id}] Starting trace '{name}'")

        try:
            input_data = {"query": query} if query else None
            trace_metadata = {"execution_id": self.execution_id}
            if metadata:
                trace_metadata.update(metadata)

            # In v3, create a root span that acts as the trace container
            # start_as_current_observation returns a context manager
            self._context_manager = client.client.start_as_current_observation(
                as_type="span",
                name=name,
                input=input_data,
                metadata=trace_metadata,
            )
            # Enter the context manager to get the actual span object
            self._root_span = self._context_manager.__enter__()

            # Capture trace_id and span_id for explicit context propagation to children
            self._trace_id = getattr(self._root_span, "trace_id", None)
            self._root_span_id = getattr(self._root_span, "id", None)
            logger.debug(
                f"[{self.execution_id}] Root span IDs: trace_id={self._trace_id}, "
                f"span_id={self._root_span_id}"
            )

            # Set trace-level attributes on the span (not the context manager)
            self._root_span.update_trace(
                user_id=self.user_id,
                session_id=self.session_id,
            )

            self._start_time = time.time()
            logger.debug(
                f"[{self.execution_id}] Trace started: root_span={self._root_span is not None}"
            )
        except Exception as e:
            logger.warning(f"[{self.execution_id}] Failed to start trace: {e}")
            self._root_span = None

    def get_trace_context(self) -> Optional["TraceContext"]:
        """
        Get TraceContext for explicit parent linking in child observations.

        Returns a TraceContext with trace_id and the root span as parent.
        Child spans created via TracingContext.span() will be direct children of root.

        Returns:
            TraceContext with trace_id and parent_span_id, or None if not available.
        """
        if not self._trace_id or not self._root_span_id:
            return None
        try:
            from langfuse.types import TraceContext

            return TraceContext(
                trace_id=self._trace_id, parent_span_id=self._root_span_id
            )
        except ImportError:
            logger.warning("Failed to import TraceContext from langfuse.types")
            return None

    def end_trace(
        self,
        output: Optional[str] = None,
        status: str = "success",
        metadata: Optional[dict] = None,
    ) -> None:
        """
        End the current trace.

        Args:
            output: Final output to record
            status: Status of the request (success, error)
            metadata: Additional metadata to add
        """
        if not self._enabled or not self._root_span:
            logger.debug(
                f"[{self.execution_id}] end_trace skipped: enabled={self._enabled}, "
                f"root_span={self._root_span is not None}"
            )
            return

        logger.debug(f"[{self.execution_id}] Ending trace")
        try:
            duration_ms = (time.time() - self._start_time) * 1000
            update_metadata = {
                "status": status,
                "duration_ms": round(duration_ms, 2),
                **(metadata or {}),
            }
            self._root_span.update(
                output=output,
                metadata=update_metadata,
            )
            # Exit the context manager (not the span)
            if self._context_manager:
                self._context_manager.__exit__(None, None, None)
        except Exception as e:
            logger.warning(f"Failed to end trace: {e}")

    @contextmanager
    def span(
        self,
        name: str,
        metadata: Optional[dict] = None,
        input: Optional[dict] = None,
    ) -> Generator["SpanContext", None, None]:
        """
        Create a span context manager.

        Args:
            name: Name of the span
            metadata: Additional metadata
            input: Input data for the span

        Yields:
            SpanContext for the span
        """
        span_ctx = SpanContext(
            name=name,
            enabled=self._enabled,
            metadata=metadata,
            input=input,
            _trace_context=self.get_trace_context(),
        )
        try:
            span_ctx.start()
            yield span_ctx
        finally:
            span_ctx.end()

    @contextmanager
    def generation(
        self,
        name: str,
        model: str,
        input: Optional[Any] = None,
        metadata: Optional[dict] = None,
        model_parameters: Optional[dict] = None,
    ) -> Generator["GenerationContext", None, None]:
        """
        Create a generation context manager for LLM calls.

        Args:
            name: Name of the generation (e.g., "orchestrator_llm_call")
            model: Model name/identifier
            input: Input to the LLM (messages, prompt)
            metadata: Additional metadata
            model_parameters: Parameters like temperature, max_tokens

        Yields:
            GenerationContext for the generation
        """
        gen_ctx = GenerationContext(
            name=name,
            model=model,
            enabled=self._enabled,
            input=input,
            metadata=metadata,
            model_parameters=model_parameters,
            _trace_context=self.get_trace_context(),
        )
        try:
            gen_ctx.start()
            yield gen_ctx
        finally:
            gen_ctx.end()


@dataclass
class SpanContext:
    """Context manager for a tracing span using SDK v3."""

    name: str
    enabled: bool = False
    metadata: Optional[dict] = None
    input: Optional[dict] = None
    _context_manager: Any = field(default=None, repr=False)
    _span: Any = field(default=None, repr=False)
    _start_time: float = field(default=0.0, repr=False)
    _output: Optional[dict] = field(default=None, repr=False)
    _status: str = field(default="success", repr=False)
    _trace_context: Optional[Any] = field(default=None, repr=False)
    _span_id: Optional[str] = field(default=None, repr=False)

    def start(self) -> None:
        """Start the span using v3 API."""
        if not self.enabled:
            return

        client = get_tracing_client()
        if not client or not client.client:
            return

        try:
            self._start_time = time.time()

            # Create span using v3 API with explicit trace_context for parent linking
            # start_as_current_observation returns a context manager
            self._context_manager = client.client.start_as_current_observation(
                trace_context=self._trace_context,
                as_type="span",
                name=self.name,
                metadata=self.metadata,
                input=self.input,
            )
            # Enter the context manager to get the actual span object
            self._span = self._context_manager.__enter__()

            # Capture span_id for child nesting
            self._span_id = getattr(self._span, "id", None)
        except Exception as e:
            logger.warning(f"Failed to start span '{self.name}': {e}")
            self._span = None

    def end(self) -> None:
        """End the span."""
        if not self.enabled or not self._span:
            return

        try:
            duration_ms = (time.time() - self._start_time) * 1000
            update_metadata = {
                "status": self._status,
                "duration_ms": round(duration_ms, 2),
            }
            update_kwargs = {"metadata": update_metadata}
            if self._output:
                update_kwargs["output"] = self._output

            self._span.update(**update_kwargs)
            # Exit the context manager (not the span)
            if self._context_manager:
                self._context_manager.__exit__(None, None, None)
        except Exception as e:
            logger.warning(f"Failed to end span '{self.name}': {e}")

    def set_output(self, output: dict) -> None:
        """Set the span output."""
        self._output = output

    def set_status(self, status: str) -> None:
        """Set the span status."""
        self._status = status

    def _get_child_trace_context(self) -> Optional[Any]:
        """
        Get TraceContext for child observations.

        Uses this span as the parent for proper nesting.
        Falls back to parent's trace_context if span_id not available.
        """
        if not self._trace_context:
            return None
        if not self._span_id:
            return self._trace_context  # Fall back to parent's context

        try:
            from langfuse.types import TraceContext

            # Get trace_id from parent context (TraceContext is a TypedDict)
            trace_id = self._trace_context.get("trace_id")
            if not trace_id:
                return self._trace_context

            return TraceContext(trace_id=trace_id, parent_span_id=self._span_id)
        except ImportError:
            logger.warning("Failed to import TraceContext from langfuse.types")
            return self._trace_context

    @contextmanager
    def child_span(
        self,
        name: str,
        metadata: Optional[dict] = None,
        input: Optional[dict] = None,
    ) -> Generator["SpanContext", None, None]:
        """Create a child span with this span as parent."""
        child = SpanContext(
            name=name,
            enabled=self.enabled,
            metadata=metadata,
            input=input,
            _trace_context=self._get_child_trace_context(),
        )
        try:
            child.start()
            yield child
        finally:
            child.end()

    # Alias for child_span - allows unified API when parent is SpanContext or TracingContext
    span = child_span

    @contextmanager
    def generation(
        self,
        name: str,
        model: str,
        input: Optional[Any] = None,
        metadata: Optional[dict] = None,
        model_parameters: Optional[dict] = None,
    ) -> Generator["GenerationContext", None, None]:
        """Create a generation within this span with this span as parent."""
        gen_ctx = GenerationContext(
            name=name,
            model=model,
            enabled=self.enabled,
            input=input,
            metadata=metadata,
            model_parameters=model_parameters,
            _trace_context=self._get_child_trace_context(),
        )
        try:
            gen_ctx.start()
            yield gen_ctx
        finally:
            gen_ctx.end()


@dataclass
class GenerationContext:
    """Context manager for LLM generation tracking using SDK v3."""

    name: str
    model: str
    enabled: bool = False
    input: Optional[Any] = None
    metadata: Optional[dict] = None
    model_parameters: Optional[dict] = None
    _context_manager: Any = field(default=None, repr=False)
    _generation: Any = field(default=None, repr=False)
    _start_time: float = field(default=0.0, repr=False)
    _output: Optional[str] = field(default=None, repr=False)
    _usage: Optional[dict] = field(default=None, repr=False)
    _status: str = field(default="success", repr=False)
    _trace_context: Optional[Any] = field(default=None, repr=False)

    def start(self) -> None:
        """Start the generation using v3 API."""
        if not self.enabled:
            return

        client = get_tracing_client()
        if not client or not client.client:
            return

        try:
            self._start_time = time.time()

            # Create generation using v3 API with explicit trace_context for parent linking
            # start_as_current_observation returns a context manager
            self._context_manager = client.client.start_as_current_observation(
                trace_context=self._trace_context,
                as_type="generation",
                name=self.name,
                model=self.model,
                input=self.input,
                metadata=self.metadata,
                model_parameters=self.model_parameters,
            )
            # Enter the context manager to get the actual generation object
            self._generation = self._context_manager.__enter__()
        except Exception as e:
            logger.warning(f"Failed to start generation '{self.name}': {e}")
            self._generation = None

    def end(self) -> None:
        """End the generation."""
        if not self.enabled or not self._generation:
            return

        try:
            duration_ms = (time.time() - self._start_time) * 1000
            update_metadata = {
                "status": self._status,
                "duration_ms": round(duration_ms, 2),
            }
            update_kwargs: dict[str, Any] = {"metadata": update_metadata}
            if self._output is not None:
                update_kwargs["output"] = self._output
            if self._usage:
                update_kwargs["usage"] = self._usage

            self._generation.update(**update_kwargs)
            # Exit the context manager (not the generation)
            if self._context_manager:
                self._context_manager.__exit__(None, None, None)
        except Exception as e:
            logger.warning(f"Failed to end generation '{self.name}': {e}")

    def set_output(self, output: str) -> None:
        """Set the generation output."""
        self._output = output

    def set_usage(
        self,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
    ) -> None:
        """Set token usage for the generation."""
        self._usage = {}
        if prompt_tokens is not None:
            self._usage["promptTokens"] = prompt_tokens
        if completion_tokens is not None:
            self._usage["completionTokens"] = completion_tokens
        if total_tokens is not None:
            self._usage["totalTokens"] = total_tokens

    def set_status(self, status: str) -> None:
        """Set the generation status."""
        self._status = status
