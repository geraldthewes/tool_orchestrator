"""
ToolOrchestra Main Orchestrator

Implements the ReAct (Reasoning-Action-Observation) loop using
Nemotron-Orchestrator-8B to coordinate tools and delegate LLMs.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional, Callable

from .llm_call import LLMClient
from .config_loader import load_delegates_config
from .models import DelegateConfig, DelegatesConfiguration
from .tools import (
    search,
    format_search_results,
    execute_python,
    format_python_result,
    calculate,
    format_math_result,
    format_delegate_result,
)
from .tools.llm_delegate import call_delegate

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result from a tool execution."""

    tool_name: str
    success: bool
    result: str
    raw_data: dict = field(default_factory=dict)


@dataclass
class OrchestrationStep:
    """A single step in the orchestration process."""

    step_number: int
    reasoning: Optional[str] = None
    action: Optional[str] = None
    action_input: Optional[dict] = None
    observation: Optional[str] = None
    is_final: bool = False
    final_answer: Optional[str] = None


class ToolOrchestrator:
    """
    Main orchestrator that implements the ReAct loop.

    Uses Nemotron-Orchestrator-8B to:
    1. Reason about the task
    2. Select and call appropriate tools
    3. Process observations
    4. Produce final answers
    """

    # Static tool handlers (non-delegate tools)
    _STATIC_TOOL_HANDLERS: dict[str, Callable] = {
        "web_search": lambda params: search(
            query=params.get("query", ""),
            categories=params.get("categories"),
            num_results=params.get("num_results", 5),
        ),
        "python_execute": lambda params: execute_python(
            code=params.get("code", ""),
            timeout_seconds=params.get("timeout", 30),
        ),
        "calculate": lambda params: calculate(
            expression=params.get("expression", ""),
        ),
    }

    # Static tool formatters
    _STATIC_TOOL_FORMATTERS: dict[str, Callable] = {
        "web_search": format_search_results,
        "python_execute": format_python_result,
        "calculate": format_math_result,
    }

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        max_steps: int = 10,
        verbose: bool = False,
        delegates_config: Optional[DelegatesConfiguration] = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            llm_client: LLM client for calling the orchestrator model
            max_steps: Maximum number of reasoning steps
            verbose: Enable verbose logging
            delegates_config: Configuration for delegate LLMs (loaded from YAML if not provided)
        """
        self.llm_client = llm_client or LLMClient()
        self.max_steps = max_steps
        self.verbose = verbose
        self.steps: list[OrchestrationStep] = []

        # Load delegates configuration
        self.delegates_config = delegates_config or load_delegates_config()

        # Initialize tool registries with static tools
        self.tool_handlers: dict[str, Callable] = dict(self._STATIC_TOOL_HANDLERS)
        self.tool_formatters: dict[str, Callable] = dict(self._STATIC_TOOL_FORMATTERS)

        # Register delegate tools dynamically
        self._register_delegate_tools()

    def _register_delegate_tools(self) -> None:
        """Register tool handlers for all configured delegates."""
        for role, delegate_config in self.delegates_config.delegates.items():
            tool_name = delegate_config.tool_name
            handler = self._create_delegate_handler(delegate_config)
            self.tool_handlers[tool_name] = handler
            self.tool_formatters[tool_name] = format_delegate_result
            logger.debug(f"Registered delegate tool: {tool_name}")

    def _create_delegate_handler(self, config: DelegateConfig) -> Callable:
        """
        Create a handler function for a delegate LLM.

        Args:
            config: Delegate configuration

        Returns:
            Handler function for the delegate
        """
        def handler(params: dict) -> dict:
            prompt = params.get("prompt", "")
            temperature = params.get("temperature", config.defaults.temperature)
            max_tokens = params.get("max_tokens", config.defaults.max_tokens)

            # Clamp max_tokens to capability limit
            if max_tokens > config.capabilities.max_output_tokens:
                logger.warning(
                    f"Requested max_tokens ({max_tokens}) exceeds limit "
                    f"({config.capabilities.max_output_tokens}) for {config.role}, clamping"
                )
                max_tokens = config.capabilities.max_output_tokens

            return call_delegate(
                connection=config.connection,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        return handler

    def _build_system_prompt(self) -> str:
        """Build the system prompt with tool definitions."""
        # Start with static tools
        prompt_parts = [
            "You are an AI assistant with access to the following tools:",
            "",
            "1. **web_search**: Search the web for current information",
            '   - Parameters: {"query": "search query", "categories": "optional category", "num_results": 5}',
            "",
            "2. **python_execute**: Execute Python code safely in a sandbox",
            '   - Parameters: {"code": "python code to execute"}',
            "   - Allowed modules: math, statistics, random, datetime, json, re, collections, itertools, functools",
            "",
            "3. **calculate**: Perform mathematical calculations",
            '   - Parameters: {"expression": "math expression like 2+2 or sqrt(16)"}',
        ]

        # Add delegate tools dynamically
        tool_num = 4
        for role, delegate in self.delegates_config.delegates.items():
            specializations = ", ".join(delegate.capabilities.specializations)
            prompt_parts.extend([
                "",
                f"{tool_num}. **{delegate.tool_name}**: Delegate to {delegate.display_name}",
                f'   - Parameters: {{"prompt": "detailed question or task", "temperature": {delegate.defaults.temperature}, "max_tokens": {delegate.defaults.max_tokens}}}',
                f"   - Context limit: {delegate.capabilities.context_length:,} tokens",
                f"   - Best for: {specializations}",
                f"   - {delegate.description}",
            ])
            tool_num += 1

        # Add response format and examples
        prompt_parts.extend([
            "",
            "## Response Format",
            "",
            "You must respond using the following format:",
            "",
            "**Thought**: [Your reasoning about what to do next]",
            '**Action**: [Tool name to use, or "Final Answer" if done]',
            "**Action Input**: [JSON parameters for the tool, or the final response text]",
            "",
            "## Examples",
            "",
            "Example 1 - Using web search:",
            "**Thought**: I need to find current information about this topic.",
            "**Action**: web_search",
            '**Action Input**: {"query": "latest news about AI"}',
            "",
            "Example 2 - Using calculator:",
            "**Thought**: I need to calculate this mathematical expression.",
            "**Action**: calculate",
            '**Action Input**: {"expression": "sqrt(144) + 10**2"}',
        ])

        # Add a dynamic example for the first delegate if available
        if self.delegates_config.delegates:
            first_delegate = list(self.delegates_config.delegates.values())[0]
            prompt_parts.extend([
                "",
                f"Example 3 - Delegating to {first_delegate.display_name}:",
                f"**Thought**: This requires {first_delegate.capabilities.specializations[0] if first_delegate.capabilities.specializations else 'expert analysis'}.",
                f"**Action**: {first_delegate.tool_name}",
                '**Action Input**: {"prompt": "Analyze the implications of quantum computing on cryptography"}',
            ])

        prompt_parts.extend([
            "",
            "Example 4 - Final answer:",
            "**Thought**: I have gathered all the information needed to answer.",
            "**Action**: Final Answer",
            "**Action Input**: The answer to your question is...",
            "",
            "## Important Rules",
            "",
            "1. Always start with a **Thought** explaining your reasoning",
            "2. Use tools when you need external information or computation",
            "3. Delegate complex tasks to appropriate expert LLMs",
            "4. When you have enough information, provide a **Final Answer**",
            "5. Be concise but thorough in your final answers",
        ])

        return "\n".join(prompt_parts)

    def _parse_response(self, response: str) -> OrchestrationStep:
        """
        Parse the model's response into structured components.

        Args:
            response: Raw model response text

        Returns:
            OrchestrationStep with parsed components
        """
        step = OrchestrationStep(step_number=len(self.steps) + 1)

        # Extract Thought
        thought_match = re.search(
            r"\*\*Thought\*\*:\s*(.+?)(?=\*\*Action\*\*|$)",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if thought_match:
            step.reasoning = thought_match.group(1).strip()

        # Extract Action
        action_match = re.search(
            r"\*\*Action\*\*:\s*(.+?)(?=\*\*Action Input\*\*|$)",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if action_match:
            step.action = action_match.group(1).strip()

        # Extract Action Input
        input_match = re.search(
            r"\*\*Action Input\*\*:\s*(.+?)$",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if input_match:
            input_text = input_match.group(1).strip()

            # Check if it's a final answer
            if step.action and step.action.lower() == "final answer":
                step.is_final = True
                step.final_answer = input_text
            else:
                # Try to parse as JSON
                try:
                    # Handle potential markdown code blocks
                    json_text = input_text
                    if "```" in json_text:
                        json_match = re.search(r"```(?:json)?\s*(.+?)\s*```", json_text, re.DOTALL)
                        if json_match:
                            json_text = json_match.group(1)
                    step.action_input = json.loads(json_text)
                except json.JSONDecodeError:
                    # If JSON parsing fails, treat as raw text
                    step.action_input = {"raw": input_text}
                    logger.warning(f"Failed to parse action input as JSON: {input_text}")

        return step

    def _execute_tool(self, tool_name: str, params: dict) -> ToolResult:
        """
        Execute a tool with the given parameters.

        Args:
            tool_name: Name of the tool to execute
            params: Parameters for the tool

        Returns:
            ToolResult with execution outcome
        """
        if tool_name not in self.tool_handlers:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                result=f"Unknown tool: {tool_name}",
            )

        try:
            handler = self.tool_handlers[tool_name]
            raw_result = handler(params)

            # Format the result for the LLM
            formatter = self.tool_formatters.get(tool_name)
            if formatter:
                formatted_result = formatter(raw_result)
            else:
                formatted_result = str(raw_result)

            return ToolResult(
                tool_name=tool_name,
                success=True,
                result=formatted_result,
                raw_data=raw_result if isinstance(raw_result, dict) else {"value": raw_result},
            )

        except Exception as e:
            logger.error(f"Tool execution failed: {tool_name} - {e}")
            return ToolResult(
                tool_name=tool_name,
                success=False,
                result=f"Tool execution error: {e}",
            )

    def _build_messages(self, query: str) -> list[dict]:
        """
        Build the message history for the LLM.

        Args:
            query: The user's original query

        Returns:
            List of message dictionaries
        """
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": query},
        ]

        # Add previous steps as conversation history
        for step in self.steps:
            # Add assistant's response
            assistant_content = ""
            if step.reasoning:
                assistant_content += f"**Thought**: {step.reasoning}\n"
            if step.action:
                assistant_content += f"**Action**: {step.action}\n"
            if step.action_input:
                if step.is_final:
                    assistant_content += f"**Action Input**: {step.final_answer}"
                else:
                    assistant_content += f"**Action Input**: {json.dumps(step.action_input)}"

            messages.append({"role": "assistant", "content": assistant_content})

            # Add observation as user message (simulating tool result)
            if step.observation and not step.is_final:
                messages.append({
                    "role": "user",
                    "content": f"**Observation**: {step.observation}\n\nContinue with your next thought.",
                })

        return messages

    def run(self, query: str) -> str:
        """
        Run the orchestration loop for a given query.

        Args:
            query: The user's question or task

        Returns:
            The final answer string
        """
        self.steps = []
        logger.info(f"Starting orchestration for: {query}")

        for i in range(self.max_steps):
            # Build messages and call the orchestrator model
            messages = self._build_messages(query)

            if self.verbose:
                logger.debug(f"Step {i + 1}: Calling orchestrator model")

            response = self.llm_client.call_orchestrator(
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
            )

            if not response["success"]:
                logger.error(f"Orchestrator call failed: {response['error']}")
                return f"Error: {response['error']}"

            # Parse the response
            step = self._parse_response(response["response"])

            if self.verbose:
                logger.info(f"Step {step.step_number}:")
                logger.info(f"  Thought: {step.reasoning}")
                logger.info(f"  Action: {step.action}")
                logger.info(f"  Input: {step.action_input}")

            # Check if we have a final answer
            if step.is_final:
                self.steps.append(step)
                logger.info(f"Final answer reached after {len(self.steps)} steps")
                return step.final_answer or "No answer provided"

            # Execute the tool
            if step.action and step.action_input:
                tool_result = self._execute_tool(step.action, step.action_input)
                step.observation = tool_result.result

                if self.verbose:
                    logger.info(f"  Observation: {step.observation[:200]}...")

            self.steps.append(step)

        # Max steps reached without final answer
        logger.warning(f"Max steps ({self.max_steps}) reached without final answer")
        return "I was unable to complete the task within the allowed number of steps. Here's what I found so far:\n\n" + "\n".join(
            f"Step {s.step_number}: {s.reasoning}" for s in self.steps if s.reasoning
        )

    def get_trace(self) -> list[dict]:
        """
        Get a trace of all orchestration steps.

        Returns:
            List of step dictionaries
        """
        return [
            {
                "step": s.step_number,
                "reasoning": s.reasoning,
                "action": s.action,
                "action_input": s.action_input,
                "observation": s.observation,
                "is_final": s.is_final,
                "final_answer": s.final_answer,
            }
            for s in self.steps
        ]


def run_query(query: str, verbose: bool = False) -> str:
    """
    Convenience function to run a single query.

    Args:
        query: The user's question or task
        verbose: Enable verbose output

    Returns:
        The final answer
    """
    orchestrator = ToolOrchestrator(verbose=verbose)
    return orchestrator.run(query)
