"""Tests for tool definitions and Qwen3 prompt formatting."""

from src.orchestration.tool_defs import (
    build_tool_definitions,
    build_tools_prompt_block,
    ANSWER_TOOL,
)


class TestBuildToolDefinitions:
    """Tests for build_tool_definitions."""

    def test_includes_registry_tools(self):
        """Should include tools from the ToolRegistry."""
        tools = build_tool_definitions()
        names = [t["function"]["name"] for t in tools]
        assert "calculate" in names
        assert "web_search" in names
        assert "python_execute" in names

    def test_includes_answer_tool(self):
        """Should always include the answer tool."""
        tools = build_tool_definitions()
        names = [t["function"]["name"] for t in tools]
        assert "answer" in names

    def test_answer_tool_is_last(self):
        """Answer tool should be the last tool in the list."""
        tools = build_tool_definitions()
        assert tools[-1]["function"]["name"] == "answer"

    def test_openai_format(self):
        """Each tool should follow OpenAI function-calling format."""
        tools = build_tool_definitions()
        for tool in tools:
            assert tool["type"] == "function"
            assert "function" in tool
            func = tool["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func
            params = func["parameters"]
            assert params["type"] == "object"
            assert "properties" in params

    def test_exclude_tools(self):
        """Excluded tools should not appear in the output."""
        tools = build_tool_definitions(exclude_tools={"calculate"})
        names = [t["function"]["name"] for t in tools]
        assert "calculate" not in names
        # Other tools should still be present
        assert "web_search" in names

    def test_answer_never_excluded(self):
        """The answer tool should never be excluded."""
        tools = build_tool_definitions(exclude_tools={"answer"})
        names = [t["function"]["name"] for t in tools]
        assert "answer" in names

    def test_includes_delegate_tools(self):
        """Should include delegate tools when delegates_config is provided."""
        from src.config_loader import get_delegates_from_app_config

        delegates = get_delegates_from_app_config()
        tools = build_tool_definitions(delegates_config=delegates)
        names = [t["function"]["name"] for t in tools]

        # Check that at least one delegate tool is present
        delegate_tools = [n for n in names if n.startswith("ask_")]
        assert len(delegate_tools) > 0

    def test_delegate_tool_format(self):
        """Delegate tools should have a query parameter."""
        from src.config_loader import get_delegates_from_app_config

        delegates = get_delegates_from_app_config()
        tools = build_tool_definitions(delegates_config=delegates)

        delegate_tools = [t for t in tools if t["function"]["name"].startswith("ask_")]
        for tool in delegate_tools:
            params = tool["function"]["parameters"]
            assert "query" in params["properties"]
            assert "query" in params["required"]

    def test_exclude_delegate_tools(self):
        """Delegate tools should be excludable."""
        from src.config_loader import get_delegates_from_app_config

        delegates = get_delegates_from_app_config()
        # Get a delegate tool name to exclude
        first_role = next(iter(delegates.delegates))
        tool_name = delegates.delegates[first_role].tool_name

        tools = build_tool_definitions(
            delegates_config=delegates,
            exclude_tools={tool_name},
        )
        names = [t["function"]["name"] for t in tools]
        assert tool_name not in names


class TestAnswerTool:
    """Tests for the ANSWER_TOOL constant."""

    def test_answer_tool_structure(self):
        """Answer tool should have the correct structure."""
        assert ANSWER_TOOL["type"] == "function"
        func = ANSWER_TOOL["function"]
        assert func["name"] == "answer"
        assert "content" in func["parameters"]["properties"]
        assert "content" in func["parameters"]["required"]


class TestBuildToolsPromptBlock:
    """Tests for the Qwen3 ChatML <tools> prompt block."""

    def test_contains_tools_xml_tags(self):
        """Output should include <tools> and </tools> XML tags."""
        tools = build_tool_definitions()
        block = build_tools_prompt_block(tools)
        assert "<tools>" in block
        assert "</tools>" in block

    def test_contains_tool_call_instruction(self):
        """Output should include <tool_call> usage instruction."""
        tools = build_tool_definitions()
        block = build_tools_prompt_block(tools)
        assert "<tool_call>" in block
        assert "</tool_call>" in block
        assert '"name"' in block

    def test_contains_tools_header(self):
        """Output should include the '# Tools' section header."""
        tools = build_tool_definitions()
        block = build_tools_prompt_block(tools)
        assert "# Tools" in block

    def test_each_tool_is_json_line(self):
        """Each tool should appear as a JSON line inside <tools> block."""
        tools = build_tool_definitions()
        block = build_tools_prompt_block(tools)

        # Extract lines between <tools> and </tools>
        in_block = False
        tool_lines = []
        for line in block.splitlines():
            if line.strip() == "<tools>":
                in_block = True
                continue
            if line.strip() == "</tools>":
                break
            if in_block:
                tool_lines.append(line)

        assert len(tool_lines) == len(tools)
        for line in tool_lines:
            import json

            parsed = json.loads(line)
            assert parsed["type"] == "function"
            assert "function" in parsed

    def test_empty_tools_list(self):
        """Should produce valid block even with no tools."""
        block = build_tools_prompt_block([])
        assert "<tools>" in block
        assert "</tools>" in block
