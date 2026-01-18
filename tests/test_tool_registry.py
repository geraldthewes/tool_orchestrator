"""
Tests for the Tool Registry.

Tests cover tool registration, retrieval, and summary generation.
"""


class TestToolRegistry:
    """Tests for the ToolRegistry class."""

    def test_registry_has_static_tools(self):
        """Test that static tools are registered on import."""
        from src.tools import ToolRegistry

        tools = ToolRegistry.all_tools()

        assert "web_search" in tools
        assert "calculate" in tools
        assert "python_execute" in tools

    def test_get_existing_tool(self):
        """Test retrieving an existing tool."""
        from src.tools import ToolRegistry

        tool = ToolRegistry.get("calculate")

        assert tool is not None
        assert tool.name == "calculate"
        assert "math" in tool.description.lower()
        assert "expression" in tool.parameters

    def test_get_nonexistent_tool(self):
        """Test retrieving a nonexistent tool returns None."""
        from src.tools import ToolRegistry

        tool = ToolRegistry.get("nonexistent_tool")

        assert tool is None

    def test_tool_handler_works(self):
        """Test that registered tool handlers are callable."""
        from src.tools import ToolRegistry

        tool = ToolRegistry.get("calculate")
        result = tool.handler({"expression": "2 + 2"})

        assert result["success"] is True
        assert result["result"] == 4

    def test_tool_formatter_works(self):
        """Test that registered tool formatters work."""
        from src.tools import ToolRegistry

        tool = ToolRegistry.get("calculate")
        result = tool.handler({"expression": "3 * 3"})
        formatted = tool.formatter(result)

        assert "9" in formatted
        assert "3 * 3" in formatted

    def test_get_tools_summary(self):
        """Test getting a summary of all tools."""
        from src.tools import ToolRegistry

        summary = ToolRegistry.get_tools_summary()

        assert "web_search" in summary
        assert "calculate" in summary
        assert "python_execute" in summary
        # Should have descriptions
        assert "search" in summary.lower() or "web" in summary.lower()

    def test_all_tools_returns_copy(self):
        """Test that all_tools returns a copy, not the original dict."""
        from src.tools import ToolRegistry

        tools1 = ToolRegistry.all_tools()
        tools2 = ToolRegistry.all_tools()

        # Should be equal but not the same object
        assert tools1 == tools2
        assert tools1 is not tools2


class TestToolDefinition:
    """Tests for the ToolDefinition dataclass."""

    def test_tool_definition_attributes(self):
        """Test ToolDefinition has expected attributes."""
        from src.tools import ToolRegistry

        tool = ToolRegistry.get("web_search")

        assert hasattr(tool, "name")
        assert hasattr(tool, "description")
        assert hasattr(tool, "parameters")
        assert hasattr(tool, "handler")
        assert hasattr(tool, "formatter")

    def test_web_search_tool_parameters(self):
        """Test web_search tool has correct parameters."""
        from src.tools import ToolRegistry

        tool = ToolRegistry.get("web_search")

        assert "query" in tool.parameters
        assert "categories" in tool.parameters
        assert "num_results" in tool.parameters

    def test_python_execute_tool_parameters(self):
        """Test python_execute tool has correct parameters."""
        from src.tools import ToolRegistry

        tool = ToolRegistry.get("python_execute")

        assert "code" in tool.parameters
        assert "timeout" in tool.parameters

    def test_calculate_tool_parameters(self):
        """Test calculate tool has correct parameters."""
        from src.tools import ToolRegistry

        tool = ToolRegistry.get("calculate")

        assert "expression" in tool.parameters
