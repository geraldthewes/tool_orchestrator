"""
ToolOrchestra Main Orchestrator

Implements tool orchestration using a custom Nemotron-native loop
with stateless prompt reconstruction and OpenAI function-calling.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from .orchestration.loop import (
    OrchestrationLoop,
    OrchestrationStep,
)
from .config_loader import get_delegates_from_app_config
from .models import DelegatesConfiguration
from .tracing import TracingContext

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result from a tool execution."""

    tool_name: str
    success: bool
    result: str
    raw_data: dict = field(default_factory=dict)


# Re-export for backward compatibility
__all__ = [
    "ToolOrchestrator",
    "ToolResult",
    "OrchestrationStep",
    "run_query",
]


class ToolOrchestrator:
    """
    Main orchestrator using the Nemotron-native orchestration loop.

    Uses stateless prompt reconstruction with structured observation
    buffers and OpenAI function-calling format to coordinate tools
    and delegate LLMs.
    """

    def __init__(
        self,
        max_steps: int = 10,
        verbose: bool = False,
        delegates_config: Optional[DelegatesConfiguration] = None,
        execution_id: Optional[str] = None,
        tracing_context: Optional[TracingContext] = None,
        base_url: Optional[str] = None,
        **kwargs: object,
    ):
        """
        Initialize the orchestrator.

        Args:
            max_steps: Maximum number of reasoning steps.
            verbose: Enable verbose logging.
            delegates_config: Configuration for delegate LLMs
                (loaded from YAML if not provided).
            execution_id: Optional ID for correlating logs.
            tracing_context: Optional tracing context for Langfuse.
            base_url: Optional override for the orchestrator LLM endpoint.
            **kwargs: Ignored for backward compatibility (e.g. llm_client).
        """
        self.max_steps = max_steps
        self.verbose = verbose
        self.execution_id = execution_id
        self.tracing_context = tracing_context
        self.delegates_config = delegates_config or get_delegates_from_app_config()

        self._loop = OrchestrationLoop(
            max_steps=max_steps,
            verbose=verbose,
            delegates_config=self.delegates_config,
            execution_id=execution_id,
            tracing_context=tracing_context,
            base_url=base_url,
        )

    @property
    def steps(self) -> list[OrchestrationStep]:
        """Get the orchestration steps from the underlying loop."""
        return self._loop.steps

    @steps.setter
    def steps(self, value: list[OrchestrationStep]) -> None:
        """Set the orchestration steps on the underlying loop."""
        self._loop.steps = value

    @property
    def llm_client(self) -> None:
        """Backward compatibility property - returns None as LLM is internal."""
        return None

    @llm_client.setter
    def llm_client(self, value: object) -> None:
        """Backward compatibility setter - ignored."""
        pass  # noqa: PIE790

    @property
    def delegate_handlers(self) -> dict:
        """Backward compatibility property - returns empty dict."""
        return {}

    def run(self, query: str) -> str:
        """
        Run the orchestration loop for a given query.

        Args:
            query: The user's question or task.

        Returns:
            The final answer string.
        """
        result = self._loop.run(query)
        return result.answer

    def get_trace(self) -> list[dict]:
        """
        Get a trace of all orchestration steps.

        Returns:
            List of step dictionaries.
        """
        return self._loop.get_trace()

    def close(self) -> None:
        """Close the underlying OpenAI client."""
        self._loop.close()


def run_query(query: str, verbose: bool = False) -> str:
    """
    Convenience function to run a single query.

    Args:
        query: The user's question or task.
        verbose: Enable verbose output.

    Returns:
        The final answer.
    """
    orchestrator = ToolOrchestrator(verbose=verbose)
    return orchestrator.run(query)
