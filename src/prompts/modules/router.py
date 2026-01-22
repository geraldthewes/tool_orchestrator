"""
DSPy Module for Query Routing.

Routes queries to either direct response or full tool orchestration
using DSPy's ChainOfThought pattern.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import dspy

from ..signatures import QueryRouting
from ..adapters import get_fast_lm
from ...tools.registry import ToolRegistry
from ...config_loader import load_delegates_config

logger = logging.getLogger(__name__)


@dataclass
class RoutingResult:
    """Result of query routing decision."""

    needs_orchestration: bool
    direct_response: Optional[str]
    reason: str


class QueryRouterModule(dspy.Module):
    """
    DSPy module for routing queries to orchestration or direct response.

    Uses ChainOfThought for explicit reasoning about whether tools are needed.
    """

    def __init__(self):
        super().__init__()
        self.router = dspy.ChainOfThought(QueryRouting)
        self._tools_list: Optional[str] = None

    def _build_tools_list(self) -> str:
        """Build tool list from registry + delegates."""
        lines = []

        # From tool registry (static tools)
        for name, tool in ToolRegistry.all_tools().items():
            lines.append(f"- {name}: {tool.description}")

        # From delegates config
        delegates_config = load_delegates_config()
        for role, delegate in delegates_config.delegates.items():
            specs = ", ".join(delegate.capabilities.specializations)
            lines.append(
                f"- {delegate.tool_name}: {delegate.description} (Best for: {specs})"
            )

        return "\n".join(lines)

    def forward(self, query: str) -> RoutingResult:
        """
        Route a query to determine if orchestration is needed.

        Args:
            query: The user's query

        Returns:
            RoutingResult with routing decision and optional direct response
        """
        if self._tools_list is None:
            self._tools_list = self._build_tools_list()

        try:
            # Call the DSPy router with ChainOfThought
            result = self.router(
                query=query,
                available_tools=self._tools_list,
            )

            needs_tools = result.needs_tools
            reasoning = result.reasoning
            direct_answer = result.direct_answer

            # Handle string "True"/"False" if model returns strings
            if isinstance(needs_tools, str):
                needs_tools = needs_tools.lower() in ("true", "yes", "1")

            if needs_tools:
                return RoutingResult(
                    needs_orchestration=True,
                    direct_response=None,
                    reason=reasoning or "LLM determined tools required",
                )

            return RoutingResult(
                needs_orchestration=False,
                direct_response=direct_answer or "",
                reason=reasoning or "LLM answered directly",
            )

        except Exception as e:
            logger.warning(f"Router failed: {e}, defaulting to orchestration")
            return RoutingResult(
                needs_orchestration=True,
                direct_response=None,
                reason=f"Router failed: {e}",
            )

    def route(self, query: str) -> RoutingResult:
        """
        Convenience method matching the original QueryRouter interface.

        Args:
            query: The user's query

        Returns:
            RoutingResult with routing decision
        """
        # Configure DSPy to use the fast LM for routing
        fast_lm = get_fast_lm(temperature=0.3)
        with dspy.context(lm=fast_lm):
            return self.forward(query)
