"""
OpenAI function-calling tool definitions.

Converts ToolRegistry entries and delegate configurations into
the OpenAI ``tools`` format for vLLM native function calling.
"""

import logging
from typing import Optional

from ..tools.registry import ToolRegistry
from ..models import DelegatesConfiguration

logger = logging.getLogger(__name__)

# Terminal tool: calling this ends the orchestration loop.
ANSWER_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "answer",
        "description": (
            "Provide the final answer to the user's question. "
            "Call this tool when you have gathered enough information "
            "to answer completely."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The final answer to the user's question.",
                }
            },
            "required": ["content"],
        },
    },
}


def build_tool_definitions(
    delegates_config: Optional[DelegatesConfiguration] = None,
    exclude_tools: Optional[set[str]] = None,
) -> list[dict]:
    """
    Build OpenAI function-calling tool definitions from the registry and delegates.

    Args:
        delegates_config: Delegate LLM configuration. If None, only
            registry tools and the answer tool are included.
        exclude_tools: Set of tool names to exclude (for anti-repetition).
            The ``answer`` tool is never excluded.

    Returns:
        List of OpenAI-format tool definitions.
    """
    exclude = exclude_tools or set()
    tools: list[dict] = []

    # Registry tools (static tools: web_search, python_execute, calculate)
    for name, tool_def in ToolRegistry.all_tools().items():
        if name in exclude:
            logger.debug("Excluding tool '%s' (anti-repetition)", name)
            continue

        properties: dict = {}
        required: list[str] = []
        for param_name, param_desc in tool_def.parameters.items():
            properties[param_name] = {
                "type": "string",
                "description": param_desc,
            }
            required.append(param_name)

        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool_def.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }
        )

    # Delegate tools (ask_reasoner, ask_coder, ask_fast, etc.)
    if delegates_config:
        for role, delegate in delegates_config.delegates.items():
            tool_name = delegate.tool_name
            if tool_name in exclude:
                logger.debug(
                    "Excluding delegate tool '%s' (anti-repetition)", tool_name
                )
                continue

            description = (
                f"{delegate.description} "
                f"(Best for: {', '.join(delegate.capabilities.specializations)})"
            )

            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": description,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": (
                                        "The question or task to delegate "
                                        f"to the {delegate.display_name}."
                                    ),
                                }
                            },
                            "required": ["query"],
                        },
                    },
                }
            )

    # The answer tool is always included (never excluded).
    tools.append(ANSWER_TOOL)

    return tools
