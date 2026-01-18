"""
Tool Registry - Single source of truth for tool definitions.

Provides a central registry for all tools with their metadata,
handlers, and formatters.
"""

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class ToolDefinition:
    """Metadata for a tool - defined once, used everywhere."""

    name: str
    description: str
    parameters: dict[str, str]  # param_name -> description
    handler: Callable[[dict], dict]
    formatter: Callable[[dict], str]


class ToolRegistry:
    """Central registry for all tools."""

    _tools: dict[str, ToolDefinition] = {}

    @classmethod
    def register(
        cls,
        name: str,
        description: str,
        parameters: dict[str, str],
        handler: Callable[[dict], dict],
        formatter: Callable[[dict], str],
    ) -> None:
        """Register a tool with its metadata."""
        cls._tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
            formatter=formatter,
        )

    @classmethod
    def get(cls, name: str) -> Optional[ToolDefinition]:
        """Get a tool by name."""
        return cls._tools.get(name)

    @classmethod
    def all_tools(cls) -> dict[str, ToolDefinition]:
        """Get a copy of all registered tools."""
        return cls._tools.copy()

    @classmethod
    def get_tools_summary(cls) -> str:
        """Get formatted summary of all tools for prompts."""
        lines = []
        for name, tool in cls._tools.items():
            lines.append(f"- {name}: {tool.description}")
        return "\n".join(lines)

    @classmethod
    def clear(cls) -> None:
        """Clear all registered tools (mainly for testing)."""
        cls._tools.clear()
