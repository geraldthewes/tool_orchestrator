"""
ToolOrchestra Main Orchestrator

Implements the ReAct (Reasoning-Action-Observation) loop using DSPy
to coordinate tools and delegate LLMs.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from .prompts.modules.orchestrator import (
    ToolOrchestratorModule,
    ToolResult,
    OrchestrationStep,
)
from .config import config
from .config_loader import get_delegates_from_app_config
from .models import DelegatesConfiguration
from .tracing import TracingContext

logger = logging.getLogger(__name__)

# Re-export for backward compatibility
__all__ = [
    "ToolOrchestrator",
    "ToolResult",
    "OrchestrationStep",
    "run_query",
]


class ToolOrchestrator:
    """
    Main orchestrator that implements the ReAct loop using DSPy.

    Uses DSPy's ReAct module to:
    1. Reason about the task
    2. Select and call appropriate tools
    3. Process observations
    4. Produce final answers
    """

    def __init__(
        self,
        llm_client=None,  # Kept for backward compatibility but not used
        max_steps: int = 10,
        verbose: bool = False,
        delegates_config: Optional[DelegatesConfiguration] = None,
        execution_id: Optional[str] = None,
        tracing_context: Optional[TracingContext] = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            llm_client: Deprecated - kept for backward compatibility
            max_steps: Maximum number of reasoning steps
            verbose: Enable verbose logging
            delegates_config: Configuration for delegate LLMs (loaded from YAML if not provided)
            execution_id: Optional ID for correlating logs across the orchestration
            tracing_context: Optional tracing context for Langfuse observability
        """
        self.max_steps = max_steps
        self.verbose = verbose
        self.execution_id = execution_id
        self.tracing_context = tracing_context

        # Load delegates configuration
        self.delegates_config = delegates_config or get_delegates_from_app_config()

        # Create the DSPy-based orchestrator module
        self._module = ToolOrchestratorModule(
            max_steps=max_steps,
            verbose=verbose,
            delegates_config=self.delegates_config,
            execution_id=execution_id,
            tracing_context=tracing_context,
        )

    @property
    def steps(self) -> list[OrchestrationStep]:
        """Get the orchestration steps from the underlying module."""
        return self._module.steps

    @property
    def delegate_handlers(self) -> dict:
        """Backward compatibility property - returns empty dict as handlers are now in DSPy tools."""
        return {}

    def run(self, query: str) -> str:
        """
        Run the orchestration loop for a given query.

        Args:
            query: The user's question or task

        Returns:
            The final answer string
        """
        return self._module.run(query)

    def get_trace(self) -> list[dict]:
        """
        Get a trace of all orchestration steps.

        Returns:
            List of step dictionaries
        """
        return self._module.get_trace()


def run_query(query: str, verbose: bool = False) -> str:
    """
    Convenience function to run a single query.

    Args:
        query: The user's question or task
        verbose: Enable verbose output

    Returns:
        The final answer
    """
    orchestrator = ToolOrchestrator(verbose=verbose)
    return orchestrator.run(query)
