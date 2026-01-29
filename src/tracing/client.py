"""
Langfuse tracing client wrapper with graceful degradation.

Uses Langfuse SDK v3 (OpenTelemetry-based) API.

Provides a singleton client that handles:
- Missing langfuse package
- Missing credentials
- Connection failures

All operations are no-ops when tracing is disabled.
"""

import logging
from typing import Any, Optional

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

    Uses SDK v3 API with OpenTelemetry-based context propagation.
    Handles missing package, credentials, or connection failures
    without affecting core functionality.
    """

    def __init__(
        self,
        public_key: str = "",
        secret_key: str = "",
        host: str = "",
        debug: bool = False,
    ):
        """
        Initialize the tracing client.

        Args:
            public_key: Langfuse public key
            secret_key: Langfuse secret key
            host: Langfuse host URL (optional)
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

        # Warn if host is provided but appears malformed (no protocol)
        if host and not host.startswith(("http://", "https://")):
            logger.warning(
                f"LANGFUSE_HOST '{host}' may be malformed. "
                "Expected format: http://hostname:port or https://hostname:port. "
                "SDK may default to HTTPS port 443."
            )

        # Try to initialize the client
        try:
            kwargs: dict[str, Any] = {
                "public_key": public_key,
                "secret_key": secret_key,
                "debug": debug,
            }
            if host:
                kwargs["host"] = host

            self._client = Langfuse(**kwargs)

            # Validate connectivity at startup to catch misconfiguration early
            if not self._validate_connectivity():
                return

            self._enabled = True
            logger.info(f"Langfuse tracing enabled (host: {host or 'default'})")
        except Exception as e:
            self._error = f"Failed to initialize Langfuse client: {e}"
            logger.warning(f"Tracing disabled: {self._error}")

    def _validate_connectivity(self) -> bool:
        """
        Validate connectivity to Langfuse server at startup.

        Uses auth_check() to verify the endpoint is reachable and credentials are valid.
        If validation fails, disables tracing and logs a warning.

        Returns:
            True if connectivity is valid, False otherwise
        """
        if not self._client:
            return False

        try:
            result = self._client.auth_check()
            if not result:
                self._error = (
                    "Langfuse auth_check() failed - endpoint may be unreachable or "
                    "credentials may be invalid. Check LANGFUSE_HOST configuration."
                )
                logger.warning(f"Tracing disabled: {self._error}")
                self._client = None
                return False
            return True
        except Exception as e:
            self._error = (
                f"Langfuse connectivity check failed: {e}. "
                "Check LANGFUSE_HOST configuration (expected format: http://hostname:port)"
            )
            logger.warning(f"Tracing disabled: {self._error}")
            self._client = None
            return False

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

    def flush(self) -> None:
        """Flush any pending events to Langfuse."""
        if not self._enabled or not self._client:
            logger.debug("Flush skipped: tracing not enabled")
            return

        try:
            logger.debug("Flushing Langfuse events...")
            self._client.flush()
            logger.debug("Langfuse flush completed")
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
    debug: bool = False,
) -> TracingClient:
    """
    Initialize the global tracing client singleton.

    Args:
        public_key: Langfuse public key
        secret_key: Langfuse secret key
        host: Langfuse host URL
        debug: Enable debug logging

    Returns:
        The initialized TracingClient instance
    """
    global _tracing_client
    _tracing_client = TracingClient(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
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
