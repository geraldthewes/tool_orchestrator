"""
Langfuse tracing client wrapper with graceful degradation.

Provides a singleton client that handles:
- Missing langfuse package
- Missing credentials
- Connection failures

All operations are no-ops when tracing is disabled.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Track if langfuse is available
_langfuse_available = False
_langfuse_error: Optional[str] = None

try:
    from langfuse import Langfuse
    _langfuse_available = True
except ImportError as e:
    _langfuse_error = f"langfuse package not installed: {e}"
    Langfuse = None  # type: ignore


class TracingClient:
    """
    Langfuse client wrapper with graceful degradation.

    Handles missing package, credentials, or connection failures
    without affecting core functionality.
    """

    def __init__(
        self,
        public_key: str = "",
        secret_key: str = "",
        host: str = "",
        flush_at: int = 10,
        flush_interval: float = 1.0,
        debug: bool = False,
    ):
        """
        Initialize the tracing client.

        Args:
            public_key: Langfuse public key
            secret_key: Langfuse secret key
            host: Langfuse host URL (optional)
            flush_at: Number of events before auto-flush
            flush_interval: Seconds between auto-flushes
            debug: Enable debug logging in langfuse
        """
        self._client: Optional["Langfuse"] = None
        self._enabled = False
        self._error: Optional[str] = None

        # Check if langfuse is available
        if not _langfuse_available:
            self._error = _langfuse_error
            logger.debug(f"Tracing disabled: {self._error}")
            return

        # Check if credentials are provided
        if not public_key or not secret_key:
            self._error = "Langfuse credentials not configured"
            logger.debug(f"Tracing disabled: {self._error}")
            return

        # Try to initialize the client
        try:
            kwargs = {
                "public_key": public_key,
                "secret_key": secret_key,
                "flush_at": flush_at,
                "flush_interval": flush_interval,
                "debug": debug,
            }
            if host:
                kwargs["host"] = host

            self._client = Langfuse(**kwargs)
            self._enabled = True
            logger.info(f"Langfuse tracing enabled (host: {host or 'default'})")
        except Exception as e:
            self._error = f"Failed to initialize Langfuse client: {e}"
            logger.warning(f"Tracing disabled: {self._error}")

    @property
    def enabled(self) -> bool:
        """Check if tracing is enabled."""
        return self._enabled

    @property
    def error(self) -> Optional[str]:
        """Get error message if tracing is disabled."""
        return self._error

    @property
    def client(self) -> Optional["Langfuse"]:
        """Get the underlying Langfuse client (None if disabled)."""
        return self._client

    def create_trace(
        self,
        name: str,
        trace_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[dict] = None,
        tags: Optional[list[str]] = None,
        input: Optional[dict] = None,
    ):
        """
        Create a new trace.

        Returns None if tracing is disabled.
        """
        if not self._enabled or not self._client:
            return None

        try:
            kwargs = {"name": name}
            if trace_id:
                kwargs["id"] = trace_id
            if session_id:
                kwargs["session_id"] = session_id
            if user_id:
                kwargs["user_id"] = user_id
            if metadata:
                kwargs["metadata"] = metadata
            if tags:
                kwargs["tags"] = tags
            if input is not None:
                kwargs["input"] = input

            return self._client.trace(**kwargs)
        except Exception as e:
            logger.warning(f"Failed to create trace: {e}")
            return None

    def flush(self) -> None:
        """Flush any pending events to Langfuse."""
        if not self._enabled or not self._client:
            return

        try:
            self._client.flush()
        except Exception as e:
            logger.warning(f"Failed to flush tracing events: {e}")

    def shutdown(self) -> None:
        """Shutdown the tracing client, flushing any remaining events."""
        if not self._enabled or not self._client:
            return

        try:
            self._client.shutdown()
            logger.info("Langfuse tracing client shutdown complete")
        except Exception as e:
            logger.warning(f"Error during tracing client shutdown: {e}")


# Global singleton instance
_tracing_client: Optional[TracingClient] = None


def init_tracing_client(
    public_key: str = "",
    secret_key: str = "",
    host: str = "",
    flush_at: int = 10,
    flush_interval: float = 1.0,
    debug: bool = False,
) -> TracingClient:
    """
    Initialize the global tracing client singleton.

    Args:
        public_key: Langfuse public key
        secret_key: Langfuse secret key
        host: Langfuse host URL
        flush_at: Number of events before auto-flush
        flush_interval: Seconds between auto-flushes
        debug: Enable debug logging

    Returns:
        The initialized TracingClient instance
    """
    global _tracing_client
    _tracing_client = TracingClient(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
        flush_at=flush_at,
        flush_interval=flush_interval,
        debug=debug,
    )
    return _tracing_client


def get_tracing_client() -> Optional[TracingClient]:
    """Get the global tracing client instance."""
    return _tracing_client


def shutdown_tracing() -> None:
    """Shutdown the global tracing client."""
    global _tracing_client
    if _tracing_client:
        _tracing_client.shutdown()
        _tracing_client = None
