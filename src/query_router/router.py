"""
Query Router - LLM-based routing for simple vs complex queries.

Routes simple queries directly to Fast LLM, bypassing orchestration.
Uses the tool registry to dynamically build the routing prompt.
"""

import logging
from dataclasses import dataclass

from ..tools.registry import ToolRegistry
from ..tools.llm_delegate import call_fast_llm
from ..config_loader import load_delegates_config

logger = logging.getLogger(__name__)

ROUTING_PROMPT_TEMPLATE = """You are a query router. Analyze the user's query and determine if it requires external tools or can be answered directly.

Available tools in the system:
{tools_list}

RULES:
1. If the query can be answered from your knowledge without tools, respond with your answer directly.
2. If the query REQUIRES any tool (current info, calculations, code execution, web search), respond EXACTLY with: ROUTE_TO_ORCHESTRATOR

Examples of queries you should answer directly:
- "Hello" → Greet the user
- "Suggest follow-up questions" → Generate suggestions
- "What is the capital of France?" → Answer from knowledge
- "Summarize the conversation" → Summarize
- "Thanks!" → Acknowledge

Examples of queries requiring ROUTE_TO_ORCHESTRATOR:
- "What's the weather today?" → Needs current info (web search)
- "Calculate 2^100" → Needs calculation tool
- "Search for latest news on AI" → Needs web search
- "Run this Python code" → Needs code execution

User query: {query}

Your response (either answer directly OR respond with ROUTE_TO_ORCHESTRATOR):"""


@dataclass
class RoutingResult:
    """Result of query routing decision."""

    needs_orchestration: bool
    direct_response: str | None
    reason: str


class QueryRouter:
    """
    Routes queries to either direct response or full orchestration.

    Uses Fast LLM to analyze queries and determine if they need tools.
    """

    def __init__(self):
        self._tools_list: str | None = None

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

    def route(self, query: str) -> RoutingResult:
        """
        Route a query to determine if orchestration is needed.

        Args:
            query: The user's query

        Returns:
            RoutingResult with routing decision and optional direct response
        """
        if self._tools_list is None:
            self._tools_list = self._build_tools_list()

        prompt = ROUTING_PROMPT_TEMPLATE.format(
            tools_list=self._tools_list, query=query
        )
        result = call_fast_llm(prompt, temperature=0.3)

        if not result["success"]:
            logger.warning(
                f"Router failed: {result['error']}, defaulting to orchestration"
            )
            return RoutingResult(
                needs_orchestration=True,
                direct_response=None,
                reason=f"Router failed: {result['error']}",
            )

        response = result["response"].strip()
        if response.startswith("ROUTE_TO_ORCHESTRATOR"):
            return RoutingResult(
                needs_orchestration=True,
                direct_response=None,
                reason="LLM determined tools required",
            )

        return RoutingResult(
            needs_orchestration=False,
            direct_response=response,
            reason="LLM answered directly",
        )
