"""
Query Router - DSPy-based routing for simple vs complex queries.

Routes simple queries directly to Fast LLM, bypassing orchestration.
Uses DSPy signatures and modules for declarative prompt programming.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from ..prompts.modules.router import QueryRouterModule, RoutingResult

logger = logging.getLogger(__name__)

# Re-export RoutingResult for backward compatibility
__all__ = ["QueryRouter", "RoutingResult"]


class QueryRouter:
    """
    Routes queries to either direct response or full orchestration.

    Uses DSPy-based QueryRouterModule internally for routing decisions.
    """

    def __init__(self):
        self._module = QueryRouterModule()

    def route(self, query: str) -> RoutingResult:
        """
        Route a query to determine if orchestration is needed.

        Args:
            query: The user's query

        Returns:
            RoutingResult with routing decision and optional direct response
        """
        return self._module.route(query)
