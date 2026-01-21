"""
Tests for the Query Router.

Tests cover routing decisions for simple vs complex queries.
"""

from unittest.mock import patch


class TestQueryRouter:
    """Tests for the QueryRouter class."""

    @patch("src.query_router.router.call_delegate_by_role")
    def test_simple_query_returns_direct_response(self, mock_llm):
        """Test that simple queries return direct responses."""
        mock_llm.return_value = {
            "success": True,
            "response": "Hello! How can I help you today?",
        }
        from src.query_router import QueryRouter

        router = QueryRouter()
        result = router.route("hello")

        assert not result.needs_orchestration
        assert result.direct_response == "Hello! How can I help you today?"
        assert "directly" in result.reason.lower()

    @patch("src.query_router.router.call_delegate_by_role")
    def test_complex_query_routes_to_orchestrator(self, mock_llm):
        """Test that complex queries requiring tools route to orchestrator."""
        mock_llm.return_value = {
            "success": True,
            "response": "ROUTE_TO_ORCHESTRATOR",
        }
        from src.query_router import QueryRouter

        router = QueryRouter()
        result = router.route("search for latest AI news")

        assert result.needs_orchestration
        assert result.direct_response is None
        assert "tools required" in result.reason.lower()

    @patch("src.query_router.router.call_delegate_by_role")
    def test_route_to_orchestrator_with_extra_text(self, mock_llm):
        """Test that ROUTE_TO_ORCHESTRATOR prefix works even with trailing text."""
        mock_llm.return_value = {
            "success": True,
            "response": "ROUTE_TO_ORCHESTRATOR - this needs web search",
        }
        from src.query_router import QueryRouter

        router = QueryRouter()
        result = router.route("what's the weather today?")

        assert result.needs_orchestration
        assert result.direct_response is None

    @patch("src.query_router.router.call_delegate_by_role")
    def test_llm_failure_defaults_to_orchestration(self, mock_llm):
        """Test that LLM failures default to orchestration (safe fallback)."""
        mock_llm.return_value = {
            "success": False,
            "error": "Connection failed",
        }
        from src.query_router import QueryRouter

        router = QueryRouter()
        result = router.route("hello")

        assert result.needs_orchestration
        assert result.direct_response is None
        assert "failed" in result.reason.lower()

    @patch("src.query_router.router.call_delegate_by_role")
    def test_follow_up_questions_query(self, mock_llm):
        """Test that follow-up question requests are handled directly."""
        mock_llm.return_value = {
            "success": True,
            "response": "Here are some follow-up questions you might ask:\n1. What is...\n2. How does...",
        }
        from src.query_router import QueryRouter

        router = QueryRouter()
        result = router.route("suggest follow-up questions")

        assert not result.needs_orchestration
        assert "follow-up" in result.direct_response.lower()

    @patch("src.query_router.router.call_delegate_by_role")
    def test_calculation_query_routes_to_orchestrator(self, mock_llm):
        """Test that calculation queries route to orchestrator."""
        mock_llm.return_value = {
            "success": True,
            "response": "ROUTE_TO_ORCHESTRATOR",
        }
        from src.query_router import QueryRouter

        router = QueryRouter()
        result = router.route("calculate 2^100")

        assert result.needs_orchestration

    @patch("src.query_router.router.call_delegate_by_role")
    @patch("src.query_router.router.load_delegates_config")
    def test_tools_list_includes_registry_tools(self, mock_delegates, mock_llm):
        """Test that tools list includes tools from registry."""
        from src.models import DelegatesConfiguration

        mock_delegates.return_value = DelegatesConfiguration(
            version="1.0", delegates={}
        )
        mock_llm.return_value = {
            "success": True,
            "response": "Hello!",
        }
        from src.query_router import QueryRouter

        router = QueryRouter()
        router.route("test")

        # Verify the prompt was called with tool information
        call_args = mock_llm.call_args
        # call_delegate_by_role(role, prompt, temperature=...) - prompt is second positional arg
        prompt = call_args[0][1]

        # Should include static tools from registry
        assert "web_search" in prompt
        assert "calculate" in prompt
        assert "python_execute" in prompt

    @patch("src.query_router.router.call_delegate_by_role")
    def test_router_uses_low_temperature(self, mock_llm):
        """Test that router uses low temperature for deterministic routing."""
        mock_llm.return_value = {
            "success": True,
            "response": "Hello!",
        }
        from src.query_router import QueryRouter

        router = QueryRouter()
        router.route("hello")

        call_kwargs = mock_llm.call_args[1]
        assert call_kwargs.get("temperature") == 0.3


class TestRoutingResult:
    """Tests for the RoutingResult dataclass."""

    def test_routing_result_direct(self):
        """Test RoutingResult for direct responses."""
        from src.query_router.router import RoutingResult

        result = RoutingResult(
            needs_orchestration=False,
            direct_response="Hello!",
            reason="LLM answered directly",
        )

        assert not result.needs_orchestration
        assert result.direct_response == "Hello!"
        assert result.reason == "LLM answered directly"

    def test_routing_result_orchestration(self):
        """Test RoutingResult for orchestration."""
        from src.query_router.router import RoutingResult

        result = RoutingResult(
            needs_orchestration=True,
            direct_response=None,
            reason="LLM determined tools required",
        )

        assert result.needs_orchestration
        assert result.direct_response is None
        assert result.reason == "LLM determined tools required"
