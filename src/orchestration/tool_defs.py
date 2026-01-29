"""
Tool definitions for the orchestration loop.

Converts ToolRegistry entries and delegate configurations into
OpenAI-style JSON tool definitions, and formats them into the
Qwen3 ChatML ``<tools>`` prompt block that Nemotron-Orchestrator-8B
was trained to understand.
"""

import json
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

# Context retrieval tool for externalized delegate responses.
RETRIEVE_CONTEXT_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "retrieve_context",
        "description": (
            "Retrieve full content from a previous delegate response that was "
            "externalized due to length. Use this when you need more detail "
            "from a delegate's response than what's shown in the summary."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "context_id": {
                    "type": "string",
                    "description": (
                        "The context ID (e.g., 'ctx_abc123') from an externalized "
                        "delegate response."
                    ),
                },
                "offset": {
                    "type": "integer",
                    "description": (
                        "Character offset to start from (default 0). Use for "
                        "pagination through very long responses."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        "Maximum characters to retrieve (default 4000). Increase "
                        "if you need more context."
                    ),
                },
            },
            "required": ["context_id"],
        },
    },
}


def build_tool_definitions(
    delegates_config: Optional[DelegatesConfiguration] = None,
    exclude_tools: Optional[set[str]] = None,
    include_retrieve_context: bool = False,
) -> list[dict]:
    """
    Build OpenAI function-calling tool definitions from the registry and delegates.

    Args:
        delegates_config: Delegate LLM configuration. If None, only
            registry tools and the answer tool are included.
        exclude_tools: Set of tool names to exclude (for anti-repetition).
            The ``answer`` tool is never excluded.
        include_retrieve_context: Whether to include the retrieve_context tool
            for accessing externalized delegate content.

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

    # Include retrieve_context tool if externalization is active
    if include_retrieve_context and "retrieve_context" not in exclude:
        tools.append(RETRIEVE_CONTEXT_TOOL)

    # The answer tool is always included (never excluded).
    tools.append(ANSWER_TOOL)

    return tools


def build_tools_prompt_block(tools: list[dict]) -> str:
    """
    Format tool definitions into the Qwen3 ChatML ``<tools>`` prompt block.

    This produces the exact format that the Qwen3 chat template generates
    when tools are passed, allowing us to embed tool descriptions directly
    in the system prompt instead of using the OpenAI ``tools`` API parameter.

    Nemotron-Orchestrator-8B was RL-trained with this format and expects
    tools described in ``<tools>`` XML tags with one JSON object per line.

    Args:
        tools: List of OpenAI-format tool definitions (from ``build_tool_definitions``).

    Returns:
        Formatted prompt block string to append to the system prompt.
    """
    lines = [
        "",
        "# Tools",
        "",
        "You may call one or more functions to assist with the user query.",
        "",
        "You are provided with function signatures within <tools></tools> XML tags:",
        "<tools>",
    ]
    for tool in tools:
        lines.append(json.dumps(tool, separators=(",", ":")))
    lines.append("</tools>")
    lines.append("")
    lines.append(
        "For each function call, return a json object with function name and arguments "
        "within <tool_call></tool_call> XML tags:"
    )
    lines.append("<tool_call>")
    lines.append('{"name": <function-name>, "arguments": <args-json-object>}')
    lines.append("</tool_call>")
    return "\n".join(lines)
