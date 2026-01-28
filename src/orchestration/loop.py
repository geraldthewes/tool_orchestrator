"""
Core orchestration loop matching NVIDIA's ToolOrchestra reference architecture.

Implements stateless prompt reconstruction with structured observation buffers,
OpenAI function-calling format, and anti-repetition guards. Each turn
reconstructs fresh [system, user] messages from the buffers rather than
appending to a growing conversation history.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI
from openai.types.chat import ChatCompletionMessage

from ..config import config
from ..config_loader import get_delegates_from_app_config
from ..models import DelegatesConfiguration
from ..models.delegate import DelegateConfig
from ..tools.registry import ToolDefinition, ToolRegistry
from ..tools.llm_delegate import call_delegate, format_result_for_llm
from ..tracing import TracingContext
from .buffers import ObservationBuffers, TokenBudgets
from .tool_defs import build_tool_definitions

logger = logging.getLogger(__name__)

# System prompt matching NVIDIA's RL training environment.
SYSTEM_PROMPT = "You are good at using tools."

# Anti-repetition: exclude a tool after this many consecutive calls.
MAX_CONSECUTIVE_CALLS = 2


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


@dataclass
class OrchestrationResult:
    """Result from a complete orchestration run."""

    answer: str
    steps: list[OrchestrationStep] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)


class OrchestrationLoop:
    """
    Custom orchestration loop using OpenAI function-calling.

    Per-step flow:
        1. Build fresh [system, user] messages from observation buffers
        2. Determine excluded tools (anti-repetition)
        3. Build OpenAI function-calling tool definitions
        4. Call LLM via OpenAI SDK with ``tools`` parameter
        5. Parse response: extract <think> block, handle tool_calls
        6. If ``answer`` tool: extract final answer, break
        7. If other tool: execute, add observation to buffers
        8. Update repetition tracking
    """

    def __init__(
        self,
        max_steps: int = 10,
        verbose: bool = False,
        delegates_config: Optional[DelegatesConfiguration] = None,
        execution_id: Optional[str] = None,
        tracing_context: Optional[TracingContext] = None,
        base_url: Optional[str] = None,
    ):
        self.max_steps = max_steps
        self.verbose = verbose
        self.execution_id = execution_id
        self.tracing_context = tracing_context
        self.delegates_config = delegates_config or get_delegates_from_app_config()

        # OpenAI client pointing at the vLLM endpoint
        resolved_url = base_url or config.orchestrator.base_url
        self._client = OpenAI(
            base_url=resolved_url,
            api_key="not-needed",  # vLLM does not require auth
        )
        self._model = config.orchestrator.model

        # Observation buffers with token budgets from config
        self._buffers = ObservationBuffers(
            budgets=TokenBudgets(
                attempts=config.orchestrator.attempts_budget,
                code=config.orchestrator.code_budget,
                delegates=config.orchestrator.code_budget,
            ),
            max_observation_chars=config.orchestrator.max_observation_tokens * 4,
        )

        # State
        self.steps: list[OrchestrationStep] = []
        self._call_history: list[str] = []  # Track tool call sequence

    def run(self, query: str) -> OrchestrationResult:
        """
        Run the orchestration loop for a query.

        Args:
            query: The user's question or task.

        Returns:
            OrchestrationResult with the final answer and step trace.
        """
        self.steps = []
        self._call_history = []

        logger.debug("Starting orchestration for: %s", query)

        if self.tracing_context:
            return self._run_with_tracing(query)
        return self._run_loop(query)

    def _run_with_tracing(self, query: str) -> OrchestrationResult:
        """Run orchestration with a tracing span."""
        if self.tracing_context is None:
            return self._run_loop(query)
        with self.tracing_context.span(
            name="orchestration",
            metadata={
                "max_steps": self.max_steps,
                "execution_id": self.execution_id,
            },
            input={"query": query},
        ) as orch_span:
            result = self._run_loop(query)
            orch_span.set_output(
                {
                    "steps_taken": len(self.steps),
                    "final_answer": (result.answer[:500] if result.answer else ""),
                }
            )
            return result

    def _run_loop(self, query: str) -> OrchestrationResult:
        """Core orchestration loop."""
        for step_num in range(1, self.max_steps + 1):
            step = OrchestrationStep(step_number=step_num)

            # 1. Build fresh messages from buffers
            messages = self._build_messages(query)

            # 2. Get excluded tools (anti-repetition)
            excluded = self._get_excluded_tools()

            # 3. Build tool definitions
            tools = build_tool_definitions(
                delegates_config=self.delegates_config,
                exclude_tools=excluded,
            )

            # 4. Call LLM
            response_message = self._call_llm(messages, tools, step_num)
            if response_message is None:
                step.reasoning = "LLM call failed"
                step.is_final = True
                step.final_answer = "Error: orchestration LLM call failed."
                self.steps.append(step)
                self._log_trace_summary()
                return OrchestrationResult(
                    answer=step.final_answer,
                    steps=list(self.steps),
                    tools_used=self._unique_tools_used(),
                )

            # 5. Extract <think> block as reasoning
            content = response_message.content or ""
            step.reasoning = self._extract_think_block(content)

            # 6. Handle tool calls
            tool_calls = response_message.tool_calls
            if not tool_calls:
                # No tool call: treat content as the answer
                clean_content = self._strip_think_block(content).strip()
                step.is_final = True
                step.final_answer = clean_content or content
                step.action = "answer"
                self.steps.append(step)
                self._log_trace_summary()
                return OrchestrationResult(
                    answer=step.final_answer,
                    steps=list(self.steps),
                    tools_used=self._unique_tools_used(),
                )

            # Process the first tool call (always function-type from vLLM)
            tc = tool_calls[0]
            func = tc.function  # type: ignore[union-attr]
            tool_name = func.name
            try:
                tool_args = json.loads(func.arguments)
            except (json.JSONDecodeError, TypeError):
                tool_args = {"raw": func.arguments}

            step.action = tool_name
            step.action_input = tool_args

            # 7. Check if this is the terminal "answer" tool
            if tool_name == "answer":
                answer_text = tool_args.get("content", "")
                step.is_final = True
                step.final_answer = answer_text
                self.steps.append(step)
                self._log_trace_summary()
                return OrchestrationResult(
                    answer=answer_text,
                    steps=list(self.steps),
                    tools_used=self._unique_tools_used(),
                )

            # 8. Execute tool and record observation
            observation = self._execute_tool(tool_name, tool_args, step_num)
            step.observation = observation
            self._buffers.add_observation(tool_name, observation, step_num)

            # 9. Update repetition tracking
            self._call_history.append(tool_name)

            self.steps.append(step)

        # Max steps reached: force an answer
        logger.warning("Max steps (%d) reached, forcing answer", self.max_steps)
        final_step = OrchestrationStep(
            step_number=len(self.steps) + 1,
            reasoning="Maximum steps reached",
            action="answer",
            is_final=True,
            final_answer=self._synthesize_forced_answer(),
        )
        self.steps.append(final_step)
        self._log_trace_summary()
        return OrchestrationResult(
            answer=final_step.final_answer or "",
            steps=list(self.steps),
            tools_used=self._unique_tools_used(),
        )

    def _build_messages(self, query: str) -> list[dict]:
        """
        Build fresh [system, user] messages from observation buffers.

        This is stateless prompt reconstruction: each turn starts from
        the system prompt + user query + accumulated observations,
        rather than appending to a growing conversation history.
        """
        system_msg = {"role": "system", "content": SYSTEM_PROMPT}

        # Build user message with observations context
        if self._buffers.has_observations:
            context = self._buffers.build_context_string()
            user_content = f"{query}\n\n" f"# Previous observations\n{context}"
        else:
            user_content = query

        user_msg = {"role": "user", "content": user_content}
        return [system_msg, user_msg]

    def _get_excluded_tools(self) -> set[str]:
        """
        Determine which tools to exclude based on anti-repetition tracking.

        A tool is excluded if it has been called MAX_CONSECUTIVE_CALLS times
        consecutively. The ``answer`` tool is never excluded.
        """
        if len(self._call_history) < MAX_CONSECUTIVE_CALLS:
            return set()

        excluded: set[str] = set()
        # Check last N calls
        recent = self._call_history[-MAX_CONSECUTIVE_CALLS:]
        if len(set(recent)) == 1:
            tool_name = recent[0]
            if tool_name != "answer":
                excluded.add(tool_name)
                logger.info(
                    "Excluding tool '%s' after %d consecutive calls",
                    tool_name,
                    MAX_CONSECUTIVE_CALLS,
                )
        return excluded

    def _call_llm(
        self,
        messages: list[dict],
        tools: list[dict],
        step_num: int,
    ) -> Optional[ChatCompletionMessage]:
        """
        Call the orchestrator LLM via the OpenAI SDK.

        Args:
            messages: Chat messages.
            tools: OpenAI function-calling tool definitions.
            step_num: Current step number (for tracing).

        Returns:
            The response message object, or None on failure.
        """
        id_prefix = f"[{self.execution_id}] " if self.execution_id else ""

        create_kwargs: dict = {
            "model": self._model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": config.orchestrator.temperature,
            "max_tokens": config.max_tokens,
        }

        # Add optional parameters
        if config.orchestrator.stop:
            create_kwargs["stop"] = config.orchestrator.stop
        if config.orchestrator.frequency_penalty:
            create_kwargs["frequency_penalty"] = config.orchestrator.frequency_penalty
        if config.orchestrator.presence_penalty:
            create_kwargs["presence_penalty"] = config.orchestrator.presence_penalty

        # Tracing: wrap in a generation span
        if self.tracing_context:
            return self._call_llm_with_tracing(create_kwargs, step_num, id_prefix)

        try:
            logger.debug("%sStep %d: calling LLM", id_prefix, step_num)
            response = self._client.chat.completions.create(**create_kwargs)
            return response.choices[0].message
        except Exception as e:
            logger.error("%sLLM call failed at step %d: %s", id_prefix, step_num, e)
            return None

    def _call_llm_with_tracing(
        self, create_kwargs: dict, step_num: int, id_prefix: str
    ) -> Optional[ChatCompletionMessage]:
        """Call LLM with Langfuse generation tracing."""
        if self.tracing_context is None:
            return None
        with self.tracing_context.generation(
            name=f"orchestrator_step_{step_num}",
            model=self._model,
            input=create_kwargs.get("messages"),
            model_parameters={
                "temperature": create_kwargs.get("temperature"),
                "max_tokens": create_kwargs.get("max_tokens"),
            },
        ) as gen:
            try:
                logger.debug("%sStep %d: calling LLM (traced)", id_prefix, step_num)
                response = self._client.chat.completions.create(**create_kwargs)
                msg = response.choices[0].message

                # Record output
                output_text = msg.content or ""
                if msg.tool_calls:
                    tc_summary = [
                        f"{tc.function.name}({tc.function.arguments})"
                        for tc in msg.tool_calls
                    ]
                    output_text += f" [tool_calls: {', '.join(tc_summary)}]"
                gen.set_output(output_text[:2000])

                # Record usage if available
                if response.usage:
                    gen.set_usage(
                        prompt_tokens=response.usage.prompt_tokens,
                        completion_tokens=response.usage.completion_tokens,
                        total_tokens=response.usage.total_tokens,
                    )

                return msg
            except Exception as e:
                logger.error("%sLLM call failed at step %d: %s", id_prefix, step_num, e)
                gen.set_status("error")
                return None

    def _execute_tool(self, tool_name: str, tool_args: dict, step_num: int) -> str:
        """
        Execute a tool and return the formatted observation.

        Args:
            tool_name: Name of the tool to execute.
            tool_args: Arguments for the tool.
            step_num: Current step number (for tracing).

        Returns:
            Formatted observation string.
        """
        id_prefix = f"[{self.execution_id}] " if self.execution_id else ""

        # Check if it's a delegate tool
        if tool_name.startswith("ask_"):
            return self._execute_delegate(tool_name, tool_args, step_num)

        # Look up in registry
        tool_def = ToolRegistry.get(tool_name)
        if not tool_def:
            logger.warning("%sUnknown tool: %s", id_prefix, tool_name)
            return f"Error: Unknown tool '{tool_name}'"

        # Execute with tracing
        if self.tracing_context:
            return self._execute_tool_with_tracing(
                tool_name, tool_def, tool_args, step_num
            )

        try:
            logger.debug(
                "%sStep %d: executing tool '%s'", id_prefix, step_num, tool_name
            )
            raw_result = tool_def.handler(tool_args)
            return tool_def.formatter(raw_result)
        except Exception as e:
            logger.error("%sTool '%s' execution failed: %s", id_prefix, tool_name, e)
            error_msg = str(e)
            if len(error_msg) > 500:
                error_msg = error_msg[:500] + "..."
            return f"Tool '{tool_name}' execution error: {error_msg}"

    def _execute_tool_with_tracing(
        self,
        tool_name: str,
        tool_def: ToolDefinition,
        tool_args: dict,
        step_num: int,
    ) -> str:
        """Execute a registry tool with tracing."""
        if self.tracing_context is None:
            return f"Tool '{tool_name}' error: tracing context not available"
        with self.tracing_context.span(
            name=f"tool:{tool_name}",
            input=tool_args,
        ) as span:
            try:
                raw_result = tool_def.handler(tool_args)
                formatted = tool_def.formatter(raw_result)
                span.set_output(
                    {"result": formatted[:500] if len(formatted) > 500 else formatted}
                )
                return formatted
            except Exception as e:
                logger.error("Tool '%s' execution failed: %s", tool_name, e)
                span.set_status("error")
                error_msg = str(e)
                if len(error_msg) > 500:
                    error_msg = error_msg[:500] + "..."
                return f"Tool '{tool_name}' execution error: {error_msg}"

    def _execute_delegate(self, tool_name: str, tool_args: dict, step_num: int) -> str:
        """Execute a delegate LLM tool."""
        id_prefix = f"[{self.execution_id}] " if self.execution_id else ""

        # Extract role from tool_name (ask_reasoner -> reasoner)
        role = tool_name.removeprefix("ask_")
        if role not in self.delegates_config.delegates:
            logger.warning("%sUnknown delegate role: %s", id_prefix, role)
            return f"Error: Unknown delegate '{tool_name}'"

        delegate = self.delegates_config.delegates[role]
        prompt = tool_args.get("query") or tool_args.get("prompt", "")
        if not prompt or not str(prompt).strip():
            return "Error: Query is empty. Please provide a query."

        temp = delegate.defaults.temperature
        max_tokens = delegate.defaults.max_tokens
        if max_tokens > delegate.capabilities.max_output_tokens:
            max_tokens = delegate.capabilities.max_output_tokens

        if self.tracing_context:
            return self._execute_delegate_with_tracing(
                tool_name, delegate, prompt, temp, max_tokens, step_num
            )

        try:
            logger.debug(
                "%sStep %d: delegating to '%s'", id_prefix, step_num, tool_name
            )
            result = call_delegate(
                connection=delegate.connection,
                prompt=prompt,
                temperature=temp,
                max_tokens=max_tokens,
                timeout=delegate.defaults.timeout,
            )
            return format_result_for_llm(result)
        except Exception as e:
            logger.error("Delegate '%s' call failed: %s", tool_name, e)
            error_msg = str(e)
            if len(error_msg) > 500:
                error_msg = error_msg[:500] + "..."
            return f"Delegate '{tool_name}' error: {error_msg}"

    def _execute_delegate_with_tracing(
        self,
        tool_name: str,
        delegate: DelegateConfig,
        prompt: str,
        temperature: float,
        max_tokens: int,
        step_num: int,
    ) -> str:
        """Execute a delegate tool with tracing."""
        if self.tracing_context is None:
            return f"Delegate '{tool_name}' error: tracing context not available"
        with self.tracing_context.span(
            name=f"tool:{tool_name}",
            input={
                "prompt": prompt[:500] if len(prompt) > 500 else prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        ) as span:
            try:
                result = call_delegate(
                    connection=delegate.connection,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=delegate.defaults.timeout,
                )
                formatted = format_result_for_llm(result)
                span.set_output(
                    {"result": formatted[:500] if len(formatted) > 500 else formatted}
                )
                if not result.get("success", False):
                    span.set_status("error")
                return formatted
            except Exception as e:
                logger.error("Delegate '%s' call failed: %s", tool_name, e)
                span.set_status("error")
                error_msg = str(e)
                if len(error_msg) > 500:
                    error_msg = error_msg[:500] + "..."
                return f"Delegate '{tool_name}' error: {error_msg}"

    @staticmethod
    def _extract_think_block(content: str) -> Optional[str]:
        """Extract text inside <think>...</think> tags."""
        match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        return match.group(1).strip() if match else None

    @staticmethod
    def _strip_think_block(content: str) -> str:
        """Remove <think>...</think> tags from content."""
        return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)

    def _synthesize_forced_answer(self) -> str:
        """Synthesize a best-effort answer from accumulated observations."""
        if not self.steps:
            return "Unable to determine an answer within the step limit."

        # Use the last observation that isn't an error
        for step in reversed(self.steps):
            if step.observation and "error" not in step.observation.lower():
                return f"Based on available information: {step.observation[:1000]}"

        return "Unable to determine a complete answer within the step limit."

    def _unique_tools_used(self) -> list[str]:
        """Get unique list of tools used (excluding 'answer')."""
        seen: set[str] = set()
        result: list[str] = []
        for step in self.steps:
            if step.action and step.action != "answer" and step.action not in seen:
                seen.add(step.action)
                result.append(step.action)
        return result

    def _log_trace_summary(self) -> None:
        """Log a compact trace summary."""
        id_prefix = f"[{self.execution_id}] " if self.execution_id else ""
        logger.info("%s%s", id_prefix, "\u2500" * 50)
        logger.info("%sTRACE SUMMARY", id_prefix)
        logger.info("%s%s", id_prefix, "\u2500" * 50)
        for step in self.steps:
            if step.is_final:
                logger.info(
                    "%sStep %d [FINAL]: %s", id_prefix, step.step_number, step.action
                )
            else:
                if step.observation and "Execution error" in step.observation:
                    logger.error(
                        "%sStep %d execution failed:\n%s",
                        id_prefix,
                        step.step_number,
                        step.observation,
                    )
                obs_preview = (
                    (step.observation[:80] + "...")
                    if step.observation and len(step.observation) > 80
                    else step.observation
                )
                logger.info(
                    "%sStep %d: %s -> %s",
                    id_prefix,
                    step.step_number,
                    step.action,
                    obs_preview,
                )

    def get_trace(self) -> list[dict]:
        """
        Get a trace of all orchestration steps.

        Returns:
            List of step dictionaries.
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

    def close(self) -> None:
        """Close the underlying OpenAI client."""
        try:
            self._client.close()
        except Exception as e:
            logger.debug("Error closing OpenAI client: %s", e)
