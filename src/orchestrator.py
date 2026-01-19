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
from .tools import format_delegate_result, ToolRegistry
from .tools.llm_delegate import call_delegate
from .tracing import TracingContext

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

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        max_steps: int = 10,
        verbose: bool = False,
        delegates_config: Optional[DelegatesConfiguration] = None,
        execution_id: Optional[str] = None,
        tracing_context: Optional[TracingContext] = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            llm_client: LLM client for calling the orchestrator model
            max_steps: Maximum number of reasoning steps
            verbose: Enable verbose logging
            delegates_config: Configuration for delegate LLMs (loaded from YAML if not provided)
            execution_id: Optional ID for correlating logs across the orchestration
            tracing_context: Optional tracing context for Langfuse observability
        """
        self.llm_client = llm_client or LLMClient()
        self.max_steps = max_steps
        self.verbose = verbose
        self.execution_id = execution_id
        self.tracing_context = tracing_context
        self.steps: list[OrchestrationStep] = []

        # Load delegates configuration
        self.delegates_config = delegates_config or load_delegates_config()

        # Build delegate tool handlers (registry handles static tools)
        self.delegate_handlers: dict[str, Callable] = {}
        self._register_delegate_tools()

    def _register_delegate_tools(self) -> None:
        """Register delegate tool handlers (stored locally, not in global registry)."""
        for role, delegate_config in self.delegates_config.delegates.items():
            tool_name = delegate_config.tool_name
            handler = self._create_delegate_handler(delegate_config)
            self.delegate_handlers[tool_name] = handler
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

            return self._traced_delegate_call(
                config=config,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        return handler

    def _traced_delegate_call(
        self,
        config: DelegateConfig,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> dict:
        """
        Call a delegate LLM with tracing.

        Args:
            config: Delegate configuration
            prompt: The prompt to send
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Result dictionary from call_delegate
        """
        # If no tracing context, call directly
        if not self.tracing_context:
            return call_delegate(
                connection=config.connection,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        # Wrap in generation context
        with self.tracing_context.generation(
            name=f"delegate_{config.role}",
            model=config.connection.model,
            input={"prompt": prompt},
            metadata={
                "role": config.role,
                "connection_type": config.connection.type.value,
                "base_url": config.connection.base_url,
            },
            model_parameters={
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        ) as gen:
            result = call_delegate(
                connection=config.connection,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            if result.get("success"):
                gen.set_output(result.get("response", ""))
            else:
                gen.set_status("error")
                gen.set_output(result.get("error", "Unknown error"))

            return result

    def _traced_orchestrator_call(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        step_number: int,
        span_context=None,
    ) -> dict:
        """
        Call the orchestrator LLM with tracing.

        Args:
            messages: Message history to send
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            step_number: Current step number for metadata
            span_context: Parent span context if available

        Returns:
            Result dictionary from llm_client
        """
        # Determine the parent for the generation
        parent = span_context if span_context else self.tracing_context

        # If no tracing context, call directly
        if not parent:
            return self.llm_client.call_orchestrator(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        # Wrap in generation context
        with parent.generation(
            name="orchestrator_llm",
            model=self.llm_client.orchestrator_model,
            input={"messages": messages},
            metadata={
                "step_number": step_number,
                "base_url": self.llm_client.orchestrator_url,
            },
            model_parameters={
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        ) as gen:
            result = self.llm_client.call_orchestrator(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            if result.get("success"):
                gen.set_output(result.get("response", ""))
            else:
                gen.set_status("error")
                gen.set_output(result.get("error", "Unknown error"))

            return result

    def _traced_tool_execution(
        self,
        tool_name: str,
        params: dict,
        step_number: int,
        span_context=None,
    ) -> "ToolResult":
        """
        Execute a tool with tracing.

        Args:
            tool_name: Name of the tool to execute
            params: Parameters for the tool
            step_number: Current step number for metadata
            span_context: Parent span context if available

        Returns:
            ToolResult from tool execution
        """
        parent = span_context if span_context else self.tracing_context

        # If no tracing context, execute directly
        if not parent:
            return self._execute_tool(tool_name, params)

        # Wrap in span context
        with parent.span(
            name=f"tool_{tool_name}",
            metadata={"step_number": step_number},
            input={"tool_name": tool_name, "params": params},
        ) as span:
            result = self._execute_tool(tool_name, params)

            span.set_output({
                "success": result.success,
                "result": result.result[:500] if result.result else None,  # Truncate for tracing
            })
            if not result.success:
                span.set_status("error")

            return result

    def _build_system_prompt(self) -> str:
        """Build the system prompt with tool definitions."""
        prompt_parts = [
            "You are an AI assistant with access to the following tools:",
            "",
        ]

        # Add tools from registry (static tools)
        tool_num = 1
        for name, tool in ToolRegistry.all_tools().items():
            params_str = json.dumps(tool.parameters)
            prompt_parts.extend([
                f"{tool_num}. **{name}**: {tool.description}",
                f"   - Parameters: {params_str}",
                "",
            ])
            tool_num += 1

        # Add delegate tools dynamically
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
        # Check registry for static tools first
        tool = ToolRegistry.get(tool_name)
        if tool:
            try:
                raw_result = tool.handler(params)
                formatted_result = tool.formatter(raw_result)
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

        # Check delegate handlers
        if tool_name in self.delegate_handlers:
            try:
                handler = self.delegate_handlers[tool_name]
                raw_result = handler(params)
                formatted_result = format_delegate_result(raw_result)
                return ToolResult(
                    tool_name=tool_name,
                    success=True,
                    result=formatted_result,
                    raw_data=raw_result if isinstance(raw_result, dict) else {"value": raw_result},
                )
            except Exception as e:
                logger.error(f"Delegate tool execution failed: {tool_name} - {e}")
                return ToolResult(
                    tool_name=tool_name,
                    success=False,
                    result=f"Tool execution error: {e}",
                )

        # Unknown tool
        return ToolResult(
            tool_name=tool_name,
            success=False,
            result=f"Unknown tool: {tool_name}",
        )

    def _log_trace_summary(self) -> None:
        """Log a compact trace summary."""
        id_prefix = f"[{self.execution_id}] " if self.execution_id else ""
        logger.info(f"{id_prefix}{'─' * 50}")
        logger.info(f"{id_prefix}TRACE SUMMARY")
        logger.info(f"{id_prefix}{'─' * 50}")
        for step in self.steps:
            if step.is_final:
                logger.info(f"{id_prefix}Step {step.step_number} [FINAL]: {step.action}")
            else:
                obs_preview = (step.observation[:80] + "...") if step.observation and len(step.observation) > 80 else step.observation
                logger.info(f"{id_prefix}Step {step.step_number}: {step.action} -> {obs_preview}")

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
        logger.debug(f"Starting orchestration for: {query}")

        # Use tracing span if available
        if self.tracing_context:
            return self._run_with_tracing(query)
        else:
            return self._run_orchestration_loop(query)

    def _run_with_tracing(self, query: str) -> str:
        """Run orchestration loop with tracing span."""
        with self.tracing_context.span(
            name="orchestration",
            metadata={
                "max_steps": self.max_steps,
                "execution_id": self.execution_id,
            },
            input={"query": query},
        ) as orch_span:
            result = self._run_orchestration_loop(query, span_context=orch_span)
            orch_span.set_output({
                "steps_taken": len(self.steps),
                "final_answer": result[:500] if result else None,
            })
            return result

    def _run_orchestration_loop(self, query: str, span_context=None) -> str:
        """
        Execute the orchestration loop.

        Args:
            query: The user's question or task
            span_context: Optional span context for tracing

        Returns:
            The final answer string
        """
        for i in range(self.max_steps):
            step_number = i + 1

            # Create step span if tracing is enabled
            if span_context:
                with span_context.child_span(
                    name=f"step_{step_number}",
                    metadata={"step_number": step_number},
                ) as step_span:
                    result = self._execute_step(query, step_number, step_span)
                    if result is not None:
                        return result
            else:
                result = self._execute_step(query, step_number, None)
                if result is not None:
                    return result

        # Max steps reached without final answer
        id_prefix = f"[{self.execution_id}] " if self.execution_id else ""
        query_preview = query[:100] + "..." if len(query) > 100 else query
        logger.warning(
            f"{id_prefix}Max steps ({self.max_steps}) reached without final answer. "
            f"Query: {query_preview}"
        )
        self._log_trace_summary()
        return "I was unable to complete the task within the allowed number of steps. Here's what I found so far:\n\n" + "\n".join(
            f"Step {s.step_number}: {s.reasoning}" for s in self.steps if s.reasoning
        )

    def _execute_step(self, query: str, step_number: int, span_context=None) -> str | None:
        """
        Execute a single orchestration step.

        Args:
            query: The user's original query
            step_number: Current step number
            span_context: Optional span context for tracing

        Returns:
            Final answer if reached, None otherwise
        """
        # Build messages and call the orchestrator model
        messages = self._build_messages(query)

        if self.verbose:
            logger.debug(f"Step {step_number}: Calling orchestrator model")

        response = self._traced_orchestrator_call(
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
            step_number=step_number,
            span_context=span_context,
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
            logger.info(f"Completed in {len(self.steps)} steps")
            self._log_trace_summary()
            return step.final_answer or "No answer provided"

        # Execute the tool
        if step.action and step.action_input:
            logger.info(f"Tool: {step.action}({json.dumps(step.action_input)})")
            tool_result = self._traced_tool_execution(
                tool_name=step.action,
                params=step.action_input,
                step_number=step_number,
                span_context=span_context,
            )
            step.observation = tool_result.result

            if self.verbose:
                logger.info(f"  Observation: {step.observation[:200]}...")

        self.steps.append(step)
        return None

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
