"""
Core orchestration loop matching NVIDIA's ToolOrchestra reference architecture.

Implements stateless prompt reconstruction with structured observation buffers,
Qwen3 ChatML tool format (``<tools>``/``<tool_call>`` XML tags), and
anti-repetition guards.  Each turn reconstructs fresh [system, user] messages
from the buffers rather than appending to a growing conversation history.

The model outputs ``<tool_call>`` XML in its text response; we parse that
rather than relying on the OpenAI ``tools`` API parameter which requires
vLLM ``--enable-auto-tool-choice``.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

from ..config import config
from ..config_loader import get_delegates_from_app_config
from ..models import DelegatesConfiguration
from ..models.delegate import DelegateConfig
from ..tools.registry import ToolDefinition, ToolRegistry
from ..tools.llm_delegate import call_delegate, call_delegate_by_role, format_result_for_llm
from ..tracing import TracingContext
from .buffers import ObservationBuffers, TokenBudgets
from .content_store import ContentStore
from .summarizer import Summarizer
from .tool_defs import build_tool_definitions, build_tools_prompt_block

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
    Custom orchestration loop using Qwen3 ChatML tool format.

    Tools are embedded in the system prompt as ``<tools>`` XML, and the
    model outputs ``<tool_call>`` XML in its text response.  This avoids
    the vLLM ``--enable-auto-tool-choice`` requirement.

    Per-step flow:
        1. Determine excluded tools (anti-repetition)
        2. Build tool definitions and system prompt with ``<tools>`` block
        3. Build fresh [system, user] messages from observation buffers
        4. Call LLM via OpenAI SDK (plain text completion, no ``tools`` param)
        5. Parse response: extract ``<think>`` block, parse ``<tool_call>`` XML
        6. If ``answer`` tool: extract final answer, return
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

        # Context externalization components
        self._content_store = ContentStore()
        self._summarizer = self._create_summarizer()

        # Observation buffers with token budgets from config
        self._buffers = ObservationBuffers(
            budgets=TokenBudgets(
                attempts=config.orchestrator.attempts_budget,
                code=config.orchestrator.code_budget,
                delegates=config.orchestrator.delegates_budget,
            ),
            max_observation_chars=config.orchestrator.max_observation_tokens * 4,
            externalize_threshold=config.orchestrator.externalize_threshold,
            keep_recent_delegate_full=config.orchestrator.keep_recent_delegate_full,
        )

        # Wire up externalization dependencies
        self._buffers.set_externalization_deps(
            content_store=self._content_store,
            summarizer=self._summarizer,
        )

        # State
        self.steps: list[OrchestrationStep] = []
        self._call_history: list[str] = []  # Track tool call sequence

    def _create_summarizer(self) -> Summarizer:
        """Create a Summarizer with fast delegate caller if available."""

        def fast_delegate_caller(prompt: str) -> Optional[str]:
            """Call the fast delegate for summary generation."""
            if "fast" not in self.delegates_config.delegates:
                return None
            try:
                result = call_delegate_by_role("fast", prompt, max_tokens=500)
                if result.get("success"):
                    return result.get("response")
            except Exception as e:
                logger.debug("Fast delegate call for summary failed: %s", e)
            return None

        return Summarizer(fast_delegate_caller=fast_delegate_caller)

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

            # 1. Get excluded tools (anti-repetition)
            excluded = self._get_excluded_tools()

            # 2. Build tool definitions and system prompt with <tools> block
            # Include retrieve_context tool if we have externalized content
            has_externalized = len(self._content_store) > 0
            tool_defs = build_tool_definitions(
                delegates_config=self.delegates_config,
                exclude_tools=excluded,
                include_retrieve_context=has_externalized,
            )
            system_prompt = SYSTEM_PROMPT + build_tools_prompt_block(tool_defs)

            # 3. Build fresh messages from buffers
            messages = self._build_messages(query, system_prompt)

            # 4. Call LLM (plain text completion, no tools API param)
            content = self._call_llm(messages, step_num)
            if content is None:
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
            step.reasoning = self._extract_think_block(content)

            # 6. Parse <tool_call> XML from text
            tool_call = self._parse_tool_call(content)
            if tool_call is None:
                # No valid tool call parsed - may be truncated output
                # Try to extract partial answer from truncated tool_call
                partial_answer = self._extract_partial_answer(content)
                if partial_answer:
                    logger.warning("Extracted partial answer from truncated tool_call")
                    clean_content = partial_answer
                else:
                    # Strip any XML tags (complete or incomplete) from content
                    clean_content = self._strip_tags(content).strip()

                step.is_final = True
                # Never return raw content with markers - use placeholder if empty
                step.final_answer = (
                    clean_content
                    if clean_content
                    else "[Response generation incomplete]"
                )
                step.action = "answer"
                self.steps.append(step)
                self._log_trace_summary()
                return OrchestrationResult(
                    answer=step.final_answer,
                    steps=list(self.steps),
                    tools_used=self._unique_tools_used(),
                )

            tool_name, tool_args = tool_call
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

    def _build_messages(
        self, query: str, system_prompt: Optional[str] = None
    ) -> list[dict]:
        """
        Build fresh [system, user] messages from observation buffers.

        This is stateless prompt reconstruction: each turn starts from
        the system prompt (with ``<tools>`` block) + user query +
        accumulated observations, rather than appending to a growing
        conversation history.

        Args:
            query: The user's question.
            system_prompt: Full system prompt including tool definitions.
                Falls back to the base SYSTEM_PROMPT if not provided.
        """
        prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
        system_msg = {"role": "system", "content": prompt}

        # Build user message with observations context
        if self._buffers.has_observations:
            context = self._buffers.build_context_string()
            user_content = f"{query}\n\n# Previous observations\n{context}"
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
        step_num: int,
    ) -> Optional[str]:
        """
        Call the orchestrator LLM via the OpenAI SDK.

        Tool definitions are already embedded in the system prompt as
        ``<tools>`` XML; we do not pass ``tools`` or ``tool_choice``
        API parameters.

        Args:
            messages: Chat messages (system prompt includes tools block).
            step_num: Current step number (for tracing).

        Returns:
            The response text content, or None on failure.
        """
        id_prefix = f"[{self.execution_id}] " if self.execution_id else ""

        create_kwargs: dict = {
            "model": self._model,
            "messages": messages,
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
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error("%sLLM call failed at step %d: %s", id_prefix, step_num, e)
            return None

    def _call_llm_with_tracing(
        self, create_kwargs: dict, step_num: int, id_prefix: str
    ) -> Optional[str]:
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
                content = response.choices[0].message.content or ""

                gen.set_output(content[:2000])

                # Record usage if available
                if response.usage:
                    gen.set_usage(
                        prompt_tokens=response.usage.prompt_tokens,
                        completion_tokens=response.usage.completion_tokens,
                        total_tokens=response.usage.total_tokens,
                    )

                return content
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

        # Check if it's the retrieve_context tool
        if tool_name == "retrieve_context":
            return self._execute_retrieve_context(tool_args, step_num)

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

    def _execute_retrieve_context(self, tool_args: dict, step_num: int) -> str:
        """
        Execute the retrieve_context tool to fetch externalized content.

        Args:
            tool_args: Arguments containing context_id, optional offset and limit.
            step_num: Current step number (for logging).

        Returns:
            Retrieved content or error message.
        """
        id_prefix = f"[{self.execution_id}] " if self.execution_id else ""

        context_id = tool_args.get("context_id", "")
        if not context_id:
            return "Error: context_id is required"

        # Parse offset and limit with defaults
        try:
            offset = int(tool_args.get("offset", 0))
        except (TypeError, ValueError):
            offset = 0

        try:
            limit = int(tool_args.get("limit", 4000))
        except (TypeError, ValueError):
            limit = 4000

        logger.debug(
            "%sStep %d: retrieving context %s (offset=%d, limit=%d)",
            id_prefix,
            step_num,
            context_id,
            offset,
            limit,
        )

        # Execute with tracing if available
        if self.tracing_context:
            return self._execute_retrieve_context_with_tracing(
                context_id, offset, limit, step_num
            )

        content = self._content_store.retrieve(context_id, offset=offset, limit=limit)
        if content is None:
            return f"Error: Context ID '{context_id}' not found"

        return content

    def _execute_retrieve_context_with_tracing(
        self,
        context_id: str,
        offset: int,
        limit: int,
        step_num: int,
    ) -> str:
        """Execute retrieve_context with tracing."""
        if self.tracing_context is None:
            return "Error: tracing context not available"

        with self.tracing_context.span(
            name="tool:retrieve_context",
            input={"context_id": context_id, "offset": offset, "limit": limit},
        ) as span:
            content = self._content_store.retrieve(
                context_id, offset=offset, limit=limit
            )
            if content is None:
                span.set_status("error")
                return f"Error: Context ID '{context_id}' not found"

            span.set_output(
                {"chars_retrieved": len(content), "preview": content[:200]}
            )
            return content

    @staticmethod
    def _extract_think_block(content: str) -> Optional[str]:
        """Extract text inside <think>...</think> tags."""
        match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        return match.group(1).strip() if match else None

    @staticmethod
    def _strip_think_block(content: str) -> str:
        """Remove <think>...</think> tags from content."""
        return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)

    @staticmethod
    def _parse_tool_call(content: str) -> Optional[tuple[str, dict]]:
        """
        Parse a ``<tool_call>`` XML block from model output.

        The Qwen3 ChatML format outputs tool calls as::

            <tool_call>
            {"name": "web_search", "arguments": {"query": "test"}}
            </tool_call>

        Returns:
            Tuple of (tool_name, tool_args) or None if no tool call found.
        """
        match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", content, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(1))
            name = data.get("name", "")
            args = data.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            if not name:
                return None
            return (name, args)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse <tool_call> JSON: %s", match.group(1)[:200])
            return None

    @staticmethod
    def _strip_tags(content: str) -> str:
        """Remove ``<think>`` and ``<tool_call>`` blocks from content.

        Handles both complete blocks (with closing tags) and incomplete/truncated
        blocks (where output was cut off mid-tag due to max_tokens).
        """
        # Remove complete blocks first
        result = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        result = re.sub(r"<tool_call>.*?</tool_call>", "", result, flags=re.DOTALL)
        result = re.sub(r"<message>(.*?)</message>", r"\1", result, flags=re.DOTALL)

        # Remove incomplete/unclosed blocks (truncated output)
        result = re.sub(r"<think>.*$", "", result, flags=re.DOTALL)
        result = re.sub(r"<tool_call>.*$", "", result, flags=re.DOTALL)

        return result

    @staticmethod
    def _extract_partial_answer(content: str) -> Optional[str]:
        """Extract answer content from a potentially truncated tool_call.

        When the LLM output is truncated (hits max_tokens), the JSON inside
        ``<tool_call>`` may be incomplete. This method attempts to extract
        the answer content even from malformed JSON.

        Args:
            content: Raw LLM output that may contain truncated tool_call.

        Returns:
            Extracted answer text, or None if not an answer tool_call or
            content cannot be extracted.
        """
        # Only attempt extraction for answer tool calls
        if '"name": "answer"' not in content and '"name":"answer"' not in content:
            return None

        # Try to find content value even if JSON is incomplete
        # Matches: "content": "some text that may be cut off
        match = re.search(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)', content, re.DOTALL)
        if match:
            # Unescape JSON string escapes
            answer = match.group(1)
            answer = answer.replace('\\"', '"').replace("\\n", "\n")
            # Clean up trailing backslash from truncation and strip whitespace
            answer = answer.rstrip("\\").rstrip()
            if answer:
                return answer
        return None

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
