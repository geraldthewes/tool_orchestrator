"""
Tests for the Query Router.

Tests cover routing decisions for simple vs complex queries using DSPy.
"""

from unittest.mock import patch, MagicMock


class TestQueryRouter:
    """Tests for the QueryRouter class."""

    @patch("src.prompts.modules.router.get_fast_lm")
    @patch("src.prompts.modules.router.dspy.context")
    def test_simple_query_returns_direct_response(self, mock_context, mock_get_lm):
        """Test that simple queries return direct responses."""
        mock_lm = MagicMock()
        mock_get_lm.return_value = mock_lm
        mock_context.return_value.__enter__ = MagicMock()
        mock_context.return_value.__exit__ = MagicMock(return_value=False)

        from src.query_router import QueryRouter

        router = QueryRouter()

        # Mock the internal router to return a direct response
        mock_result = MagicMock()
        mock_result.needs_tools = False
        mock_result.reasoning = "Simple greeting"
        mock_result.direct_answer = "Hello! How can I help you today?"
        router._module.router = MagicMock(return_value=mock_result)

        result = router.route("hello")

        assert not result.needs_orchestration
        assert result.direct_response == "Hello! How can I help you today?"

    @patch("src.prompts.modules.router.get_fast_lm")
    @patch("src.prompts.modules.router.dspy.context")
    def test_complex_query_routes_to_orchestrator(self, mock_context, mock_get_lm):
        """Test that complex queries requiring tools route to orchestrator."""
        mock_lm = MagicMock()
        mock_get_lm.return_value = mock_lm
        mock_context.return_value.__enter__ = MagicMock()
        mock_context.return_value.__exit__ = MagicMock(return_value=False)

        from src.query_router import QueryRouter

        router = QueryRouter()

        # Mock the internal router to route to orchestrator
        mock_result = MagicMock()
        mock_result.needs_tools = True
        mock_result.reasoning = "Needs web search for current info"
        mock_result.direct_answer = ""
        router._module.router = MagicMock(return_value=mock_result)

        result = router.route("search for latest AI news")

        assert result.needs_orchestration
        assert result.direct_response is None

    @patch("src.prompts.modules.router.get_fast_lm")
    @patch("src.prompts.modules.router.dspy.context")
    def test_route_to_orchestrator_with_extra_text(self, mock_context, mock_get_lm):
        """Test routing to orchestrator based on needs_tools flag."""
        mock_lm = MagicMock()
        mock_get_lm.return_value = mock_lm
        mock_context.return_value.__enter__ = MagicMock()
        mock_context.return_value.__exit__ = MagicMock(return_value=False)

        from src.query_router import QueryRouter

        router = QueryRouter()

        mock_result = MagicMock()
        mock_result.needs_tools = True
        mock_result.reasoning = "This needs web search"
        mock_result.direct_answer = ""
        router._module.router = MagicMock(return_value=mock_result)

        result = router.route("what's the weather today?")

        assert result.needs_orchestration
        assert result.direct_response is None

    @patch("src.prompts.modules.router.get_fast_lm")
    @patch("src.prompts.modules.router.dspy.context")
    def test_llm_failure_defaults_to_orchestration(self, mock_context, mock_get_lm):
        """Test that LLM failures default to orchestration (safe fallback)."""
        mock_lm = MagicMock()
        mock_get_lm.return_value = mock_lm
        mock_context.return_value.__enter__ = MagicMock()
        mock_context.return_value.__exit__ = MagicMock(return_value=False)

        from src.query_router import QueryRouter

        router = QueryRouter()

        # Mock the internal router to raise an exception
        router._module.router = MagicMock(side_effect=Exception("LLM error"))

        result = router.route("hello")

        assert result.needs_orchestration
        assert result.direct_response is None
        assert "failed" in result.reason.lower()

    @patch("src.prompts.modules.router.get_fast_lm")
    @patch("src.prompts.modules.router.dspy.context")
    def test_follow_up_questions_query(self, mock_context, mock_get_lm):
        """Test that follow-up question requests are handled directly."""
        mock_lm = MagicMock()
        mock_get_lm.return_value = mock_lm
        mock_context.return_value.__enter__ = MagicMock()
        mock_context.return_value.__exit__ = MagicMock(return_value=False)

        from src.query_router import QueryRouter

        router = QueryRouter()

        mock_result = MagicMock()
        mock_result.needs_tools = False
        mock_result.reasoning = "Can generate follow-up questions from context"
        mock_result.direct_answer = "Here are some follow-up questions you might ask:\n1. What is...\n2. How does..."
        router._module.router = MagicMock(return_value=mock_result)

        result = router.route("suggest follow-up questions")

        assert not result.needs_orchestration
        assert "follow-up" in result.direct_response.lower()

    @patch("src.prompts.modules.router.get_fast_lm")
    @patch("src.prompts.modules.router.dspy.context")
    def test_calculation_query_routes_to_orchestrator(self, mock_context, mock_get_lm):
        """Test that calculation queries route to orchestrator."""
        mock_lm = MagicMock()
        mock_get_lm.return_value = mock_lm
        mock_context.return_value.__enter__ = MagicMock()
        mock_context.return_value.__exit__ = MagicMock(return_value=False)

        from src.query_router import QueryRouter

        router = QueryRouter()

        mock_result = MagicMock()
        mock_result.needs_tools = True
        mock_result.reasoning = "Large calculation requires calculator tool"
        mock_result.direct_answer = ""
        router._module.router = MagicMock(return_value=mock_result)

        result = router.route("calculate 2^100")

        assert result.needs_orchestration

    def test_tools_list_includes_registry_tools(self):
        """Test that tools list includes tools from registry."""
        from src.query_router import QueryRouter

        router = QueryRouter()
        tools_list = router._module._build_tools_list()

        # Should include static tools from registry
        assert "web_search" in tools_list
        assert "calculate" in tools_list
        assert "python_execute" in tools_list

    @patch("src.prompts.modules.router.get_fast_lm")
    def test_router_uses_low_temperature(self, mock_get_lm):
        """Test that router uses low temperature for deterministic routing."""
        mock_lm = MagicMock()
        mock_get_lm.return_value = mock_lm

        from src.query_router import QueryRouter

        router = QueryRouter()

        # The route method should call get_fast_lm with temperature=0.3
        with patch("src.prompts.modules.router.dspy.context") as mock_context:
            mock_context.return_value.__enter__ = MagicMock()
            mock_context.return_value.__exit__ = MagicMock(return_value=False)

            mock_result = MagicMock()
            mock_result.needs_tools = False
            mock_result.reasoning = "Simple"
            mock_result.direct_answer = "Hello!"
            router._module.router = MagicMock(return_value=mock_result)

            router.route("hello")

        # Verify get_fast_lm was called with temperature=0.3
        mock_get_lm.assert_called_once_with(temperature=0.3)


class TestRoutingResult:
    """Tests for the RoutingResult dataclass."""

    def test_routing_result_direct(self):
        """Test RoutingResult for direct responses."""
        from src.prompts.modules.router import RoutingResult

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
        from src.prompts.modules.router import RoutingResult

        result = RoutingResult(
            needs_orchestration=True,
            direct_response=None,
            reason="LLM determined tools required",
        )

        assert result.needs_orchestration
        assert result.direct_response is None
        assert result.reason == "LLM determined tools required"
