"""
DSPy Module for Tool Orchestration.

Implements ReAct-style reasoning using DSPy's built-in ReAct module
with adapters for existing ToolOrchestra tools.
"""

import contextvars
import inspect
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, Any

import dspy

from ..signatures import ToolOrchestrationTask
from ..adapters import get_orchestrator_lm, NemotronJSONAdapter
from ..adapters.nemotron_adapter import set_current_query, reset_current_query
from ..adapters.lm_factory import TracedLM
from ..optimization.checkpoint import CheckpointManager
from ...tools.registry import ToolRegistry
from ...config import config
from ...config_loader import get_delegates_from_app_config
from ...models import DelegatesConfiguration
from ...tracing import TracingContext

logger = logging.getLogger(__name__)

# Thread-local storage for active tracing context during concurrent execution.
# This is needed because GEPA runs multiple examples concurrently on the same
# ToolOrchestratorModule instance. Using contextvars ensures each thread/context
# has its own tracing context that doesn't get overwritten by concurrent executions.
_active_tracing_context_var: contextvars.ContextVar[Optional[TracingContext]] = (
    contextvars.ContextVar("active_tracing_context", default=None)
)


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


def create_dspy_tool(
    name: str,
    description: str,
    parameters: dict[str, str],
    handler: Callable[[dict], Any],
    formatter: Callable[[Any], str],
    tracing_context: Optional[TracingContext] = None,
    tracing_context_getter: Optional[Callable[[], Optional[TracingContext]]] = None,
) -> Callable:
    """
    Create a DSPy-compatible tool function from handler and formatter.

    Args:
        name: Tool name
        description: Tool description
        parameters: Dict mapping parameter names to descriptions
        handler: Function that executes the tool
        formatter: Function that formats the result
        tracing_context: Optional tracing context for observability
        tracing_context_getter: Optional callable that returns the active tracing context.
            When provided, this is called at tool execution time to get the current
            context, enabling per-call tracing during DSPy optimization.

    Returns:
        DSPy-compatible tool function with explicit parameter signature
    """

    def tool_func(**kwargs) -> str:
        """Execute the tool and return formatted result."""
        logger.debug(f"Tool '{name}' called with kwargs: {list(kwargs.keys())}")

        # Check for malformed JSON input (DSPy sets "raw" key when parsing fails)
        if "raw" in kwargs:
            expected_params = ", ".join(parameters.keys())
            return f"Error: Invalid input format. Expected JSON with parameters: {expected_params}"

        # Get tracing context - prefer getter for per-call tracing, fallback to static context
        ctx = tracing_context_getter() if tracing_context_getter else tracing_context
        logger.debug(
            f"Tool '{name}' tracing context: "
            f"getter={tracing_context_getter is not None}, "
            f"ctx={ctx is not None}"
        )

        if ctx:
            logger.debug(f"Tool '{name}' creating span, trace_id={ctx._trace_id}")
            with ctx.span(
                name=f"tool:{name}",
                input=kwargs,
            ) as span:
                try:
                    raw_result = handler(kwargs)
                    formatted = formatter(raw_result)
                    span.set_output(
                        {
                            "result": (
                                formatted[:500] if len(formatted) > 500 else formatted
                            )
                        }
                    )
                    return formatted
                except Exception as e:
                    logger.error(f"Tool execution failed: {name} - {e}")
                    span.set_status("error")
                    # Return clean error message without full stack trace
                    error_msg = str(e)
                    if len(error_msg) > 500:
                        error_msg = error_msg[:500] + "..."
                    return f"Tool '{name}' execution error: {error_msg}"
        else:
            # Original logic without tracing
            try:
                raw_result = handler(kwargs)
                formatted = formatter(raw_result)
                return formatted
            except Exception as e:
                logger.error(f"Tool execution failed: {name} - {e}")
                # Return clean error message without full stack trace
                error_msg = str(e)
                if len(error_msg) > 500:
                    error_msg = error_msg[:500] + "..."
                return f"Tool '{name}' execution error: {error_msg}"

    # Set function metadata for DSPy
    tool_func.__name__ = name
    tool_func.__doc__ = description

    # Build explicit signature from parameters dict so DSPy sees named params
    # instead of **kwargs. All params are optional with None default.
    params = [
        inspect.Parameter(
            param_name,
            inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation=str,
        )
        for param_name in parameters.keys()
    ]
    tool_func.__signature__ = inspect.Signature(
        parameters=params,
        return_annotation=str,
    )

    # Add annotations dict for DSPy's get_type_hints()
    tool_func.__annotations__ = {param_name: str for param_name in parameters.keys()}
    tool_func.__annotations__["return"] = str

    return tool_func


def create_delegate_tool(
    role: str,
    display_name: str,
    description: str,
    tool_name: str,
    delegates_config: DelegatesConfiguration,
    tracing_context: Optional[TracingContext] = None,
    tracing_context_getter: Optional[Callable[[], Optional[TracingContext]]] = None,
    parameters: Optional[dict[str, str]] = None,
) -> Callable:
    """
    Create a DSPy-compatible tool function for a delegate LLM.

    Args:
        role: Delegate role (e.g., "reasoner")
        display_name: Human-readable name
        description: Tool description
        tool_name: Tool name (e.g., "ask_reasoner")
        delegates_config: Delegates configuration
        tracing_context: Optional tracing context for observability
        tracing_context_getter: Optional callable that returns the active tracing context.
            When provided, this is called at tool execution time to get the current
            context, enabling per-call tracing during DSPy optimization.
        parameters: Dict mapping parameter names to descriptions. Defaults to
            {"query": "the question or task to delegate"}

    Returns:
        DSPy-compatible tool function with explicit parameter signature
    """
    from ...tools.llm_delegate import call_delegate, format_result_for_llm

    delegate = delegates_config.delegates[role]

    # Default parameters for delegate tools
    if parameters is None:
        parameters = {"query": "the question or task to delegate"}

    def delegate_tool(**kwargs) -> str:
        """Delegate to specialized LLM."""
        logger.debug(
            f"Delegate tool '{tool_name}' called with kwargs: {list(kwargs.keys())}"
        )

        # Check for malformed JSON input (DSPy sets "raw" key when parsing fails)
        if "raw" in kwargs and "query" not in kwargs:
            return 'Error: Invalid input format. Expected JSON: {"query": "your question or task"}'

        # Accept both 'query' and 'prompt' for compatibility
        prompt = kwargs.get("query") or kwargs.get("prompt")
        if not prompt or not str(prompt).strip():
            return "Error: Query is empty. Please provide a query."

        temperature = kwargs.get("temperature")
        max_tokens = kwargs.get("max_tokens")

        temp = temperature if temperature is not None else delegate.defaults.temperature
        tokens = max_tokens if max_tokens is not None else delegate.defaults.max_tokens

        # Clamp max_tokens
        if tokens > delegate.capabilities.max_output_tokens:
            tokens = delegate.capabilities.max_output_tokens

        # Get tracing context - prefer getter for per-call tracing, fallback to static context
        ctx = tracing_context_getter() if tracing_context_getter else tracing_context
        logger.debug(
            f"Delegate tool '{tool_name}' tracing context: "
            f"getter={tracing_context_getter is not None}, "
            f"ctx={ctx is not None}"
        )

        if ctx:
            logger.debug(
                f"Delegate tool '{tool_name}' creating span, trace_id={ctx._trace_id}"
            )
            with ctx.span(
                name=f"tool:{tool_name}",
                input={
                    "prompt": prompt[:500] if len(prompt) > 500 else prompt,
                    "temperature": temp,
                    "max_tokens": tokens,
                },
            ) as span:
                result = call_delegate(
                    connection=delegate.connection,
                    prompt=prompt,
                    temperature=temp,
                    max_tokens=tokens,
                    timeout=delegate.defaults.timeout,
                )
                formatted = format_result_for_llm(result)
                span.set_output(
                    {"result": formatted[:500] if len(formatted) > 500 else formatted}
                )
                if not result.get("success", False):
                    span.set_status("error")
                return formatted
        else:
            result = call_delegate(
                connection=delegate.connection,
                prompt=prompt,
                temperature=temp,
                max_tokens=tokens,
                timeout=delegate.defaults.timeout,
            )
            return format_result_for_llm(result)

    # Set function metadata for DSPy
    delegate_tool.__name__ = tool_name
    delegate_tool.__doc__ = (
        f"{description} (Best for: {', '.join(delegate.capabilities.specializations)})"
    )

    # Build explicit signature from parameters dict so DSPy sees named params
    # instead of **kwargs. All params are optional with None default.
    params = [
        inspect.Parameter(
            param_name,
            inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation=str,
        )
        for param_name in parameters.keys()
    ]
    delegate_tool.__signature__ = inspect.Signature(
        parameters=params,
        return_annotation=str,
    )

    # Add annotations dict for DSPy's get_type_hints()
    delegate_tool.__annotations__ = {
        param_name: str for param_name in parameters.keys()
    }
    delegate_tool.__annotations__["return"] = str

    return delegate_tool


class ToolOrchestratorModule(dspy.Module):
    """
    DSPy module for tool orchestration using ReAct pattern.

    Uses DSPy's built-in ReAct module with adapted tools from ToolOrchestra.
    """

    def __init__(
        self,
        max_steps: int = 10,
        verbose: bool = False,
        delegates_config: Optional[DelegatesConfiguration] = None,
        execution_id: Optional[str] = None,
        tracing_context: Optional[TracingContext] = None,
        trace_per_call: bool = False,
        trace_name: str = "orchestration_eval",
        trace_metadata: Optional[dict] = None,
    ):
        """
        Initialize the orchestrator module.

        Args:
            max_steps: Maximum number of reasoning steps
            verbose: Enable verbose logging
            delegates_config: Configuration for delegate LLMs
            execution_id: Optional ID for correlating logs
            tracing_context: Optional tracing context for observability (used when
                trace_per_call=False, e.g., for API request tracing)
            trace_per_call: If True, create a separate Langfuse trace for each
                forward() call. Useful for DSPy optimization where each example
                should have its own trace. Default False preserves existing behavior.
            trace_name: Name for per-call traces when trace_per_call=True
            trace_metadata: Additional metadata to include in per-call traces
        """
        super().__init__()

        self.max_steps = max_steps
        self.verbose = verbose
        self.execution_id = execution_id
        self.tracing_context = tracing_context
        self.trace_per_call = trace_per_call
        self.trace_name = trace_name
        self.trace_metadata = trace_metadata or {}
        self.steps: list[OrchestrationStep] = []

        # Note: Active tracing context is stored in the module-level ContextVar
        # _active_tracing_context_var to support concurrent execution (GEPA).

        # Load delegates configuration
        self.delegates_config = delegates_config or get_delegates_from_app_config()

        # Build DSPy tools list (uses _get_active_tracing_context for deferred resolution)
        self._tools = self._build_dspy_tools()

        # Configure adapter for native function calling (Nemotron compatibility)
        # Use NemotronJSONAdapter to handle "final" wrapper in responses
        # Note: adapter uses static tracing_context since it's not per-call
        dspy.settings.adapter = NemotronJSONAdapter(
            use_native_function_calling=True,
            tracing_context=self.tracing_context,
        )

        # Create ReAct module with tools
        self.react = dspy.ReAct(
            ToolOrchestrationTask,
            tools=self._tools,
            max_iters=max_steps,
        )

        # Load optimized checkpoint if configured
        self._load_optimized_checkpoint()

    def _get_active_tracing_context(self) -> Optional[TracingContext]:
        """
        Get the active tracing context for the current forward() call.

        When trace_per_call=True, this returns the per-call context stored
        in the thread-local ContextVar. Otherwise, it falls back to the shared
        tracing_context passed to __init__ (or None if not provided).

        This method is used as a getter by tools to defer context resolution
        until tool execution time, enabling proper tracing in per-call mode.

        The context is stored in a ContextVar to support concurrent execution
        (e.g., GEPA running multiple examples in parallel on the same module).

        Returns:
            The active TracingContext, or None if tracing is not enabled.
        """
        active_ctx = _active_tracing_context_var.get()
        ctx = active_ctx or self.tracing_context
        logger.debug(
            f"_get_active_tracing_context called: "
            f"_active={active_ctx is not None}, "
            f"shared={self.tracing_context is not None}, "
            f"returning={ctx is not None}"
        )
        return ctx

    def _build_dspy_tools(self) -> list[Callable]:
        """Build list of DSPy-compatible tool functions."""
        tools = []

        # Add tools from registry (static tools)
        # Use tracing_context_getter for deferred resolution to support per-call tracing
        for name, tool_def in ToolRegistry.all_tools().items():
            dspy_tool = create_dspy_tool(
                name=name,
                description=tool_def.description,
                parameters=tool_def.parameters,
                handler=tool_def.handler,
                formatter=tool_def.formatter,
                tracing_context_getter=self._get_active_tracing_context,
            )
            tools.append(dspy_tool)
            logger.debug(f"Registered DSPy tool: {name}")

        # Add delegate tools
        # Use tracing_context_getter for deferred resolution to support per-call tracing
        for role, delegate in self.delegates_config.delegates.items():
            delegate_tool = create_delegate_tool(
                role=role,
                display_name=delegate.display_name,
                description=delegate.description,
                tool_name=delegate.tool_name,
                delegates_config=self.delegates_config,
                tracing_context_getter=self._get_active_tracing_context,
            )
            tools.append(delegate_tool)
            logger.debug(f"Registered DSPy delegate tool: {delegate.tool_name}")

        return tools

    def _load_optimized_checkpoint(self) -> None:
        """Load the best checkpoint if optimized_prompts_path is configured."""
        checkpoint_base = config.dspy.optimized_prompts_path
        if not checkpoint_base:
            logger.debug("No optimized_prompts_path configured, using default prompts")
            return

        checkpoint_dir = Path(checkpoint_base) / "orchestrator"
        if not checkpoint_dir.exists():
            logger.warning(f"Checkpoint directory not found: {checkpoint_dir}")
            return

        best_checkpoint = CheckpointManager.get_best_checkpoint(checkpoint_dir)
        if best_checkpoint:
            self.load(str(best_checkpoint))
            logger.info(f"Loaded optimized checkpoint: {best_checkpoint}")
        else:
            logger.debug(f"No best checkpoint found in {checkpoint_dir}")

    def forward(self, question: str) -> dspy.Prediction:
        """
        Run orchestration for a question using DSPy ReAct.

        When trace_per_call=True, creates a fresh Langfuse trace for this call,
        enabling separate traces for each example during DSPy optimization.

        Args:
            question: The user's question or task

        Returns:
            dspy.Prediction with answer and tools fields
        """
        self.steps = []
        logger.debug(f"Starting orchestration for: {question}")

        # Per-call tracing: create and manage a trace for this specific forward() call
        local_tracing_context: Optional[TracingContext] = None
        if self.trace_per_call:
            # Generate unique execution_id using timestamp and question hash
            execution_id = (
                f"eval-{int(time.time() * 1000)}-{abs(hash(question)) % 10000}"
            )
            local_tracing_context = TracingContext(execution_id=execution_id)
            local_tracing_context.start_trace(
                name=self.trace_name,
                query=question[:200] if len(question) > 200 else question,
                metadata={
                    **self.trace_metadata,
                    "question": question[:500] if len(question) > 500 else question,
                },
            )
            # Set as active context so tools use this trace (thread-local)
            _active_tracing_context_var.set(local_tracing_context)
            logger.debug(
                f"Created per-call trace: execution_id={execution_id}, "
                f"trace_id={local_tracing_context._trace_id}"
            )
        else:
            # Use shared tracing context (original behavior)
            _active_tracing_context_var.set(self.tracing_context)

        try:
            # Always use the orchestrator LM for ReAct tool-calling
            # This ensures the correct model is used even during DSPy optimization
            # (where a teacher LM may be set in dspy.settings.lm for prompt generation)
            orchestrator_lm = get_orchestrator_lm()

            # Get the active context for tracing
            active_ctx = self._get_active_tracing_context()
            if active_ctx:
                logger.debug(
                    f"Tracing context available: _trace_id={active_ctx._trace_id}, "
                    f"_enabled={active_ctx._enabled}"
                )
                orchestrator_lm = TracedLM(orchestrator_lm, active_ctx, "orchestrator")
            else:
                logger.debug("No tracing context available in forward()")

            # Set query context for adapter logging
            query_token = set_current_query(question)
            try:
                with dspy.context(lm=orchestrator_lm):
                    result = self.react(question=question)

                # Extract steps from trajectory
                if hasattr(result, "trajectory") and result.trajectory:
                    self.steps = self._extract_steps_from_trajectory(result.trajectory)
                    # Set final_answer on the last step if it's marked as final
                    if self.steps and self.steps[-1].is_final:
                        self.steps[-1].final_answer = (
                            result.answer if hasattr(result, "answer") else None
                        )

                # Extract answer from result
                answer = result.answer if hasattr(result, "answer") else str(result)

                # Extract tools used from steps (excluding "finish" action)
                tools_used = [
                    step.action
                    for step in self.steps
                    if step.action and step.action != "finish"
                ]

                # Log completion
                logger.info("Completed orchestration")
                self._log_trace_summary()

                # Return dspy.Prediction with both answer and tools for metric evaluation
                return dspy.Prediction(answer=answer, tools=tools_used)

            except Exception as e:
                logger.error(
                    f"Orchestration failed: {e} "
                    f"(endpoint={config.orchestrator.base_url}, model={config.orchestrator.model})"
                )
                return dspy.Prediction(
                    answer=f"Error during orchestration: {e}", tools=[]
                )
            finally:
                # Always reset query context
                reset_current_query(query_token)

        finally:
            # End per-call trace if we created one
            if local_tracing_context:
                # Get the answer for the trace output
                answer_for_trace = ""
                if self.steps:
                    final_step = self.steps[-1]
                    if final_step.final_answer:
                        answer_for_trace = final_step.final_answer[:500]
                local_tracing_context.end_trace(
                    output=answer_for_trace,
                    status="completed",
                    metadata={"steps_taken": len(self.steps)},
                )
                logger.debug(
                    f"Ended per-call trace: trace_id={local_tracing_context._trace_id}"
                )
            # Clear the active context (thread-local)
            _active_tracing_context_var.set(None)

    def run(self, query: str) -> str:
        """
        Run the orchestration loop for a given query.

        Alias for forward() to match the original ToolOrchestrator interface.

        Args:
            query: The user's question or task

        Returns:
            The final answer string
        """
        if self.tracing_context:
            return self._run_with_tracing(query)
        result = self.forward(query)
        return result.answer

    def _run_with_tracing(self, query: str) -> str:
        """Run orchestration with tracing span."""
        assert self.tracing_context is not None  # Guaranteed by caller check
        with self.tracing_context.span(
            name="orchestration",
            metadata={
                "max_steps": self.max_steps,
                "execution_id": self.execution_id,
            },
            input={"query": query},
        ) as orch_span:
            result = self.forward(query)
            answer = result.answer
            orch_span.set_output(
                {
                    "steps_taken": len(self.steps),
                    "final_answer": answer[:500] if answer else "",
                }
            )
            return answer

    def _extract_steps_from_trajectory(
        self, trajectory: dict
    ) -> list[OrchestrationStep]:
        """
        Convert DSPy ReAct trajectory dict to OrchestrationStep list.

        The trajectory dict contains keys like:
        - thought_{idx}: The reasoning for this step
        - tool_name_{idx}: The tool name (or "finish" for final answer)
        - tool_args_{idx}: The tool arguments (dict or other)
        - observation_{idx}: The tool output

        Args:
            trajectory: DSPy ReAct trajectory dictionary

        Returns:
            List of OrchestrationStep objects
        """
        if not trajectory or not isinstance(trajectory, dict):
            return []

        steps = []
        idx = 0

        while True:
            thought_key = f"thought_{idx}"
            tool_name_key = f"tool_name_{idx}"
            tool_args_key = f"tool_args_{idx}"
            observation_key = f"observation_{idx}"

            # Check if this step exists
            if thought_key not in trajectory and tool_name_key not in trajectory:
                break

            thought = trajectory.get(thought_key)
            tool_name = trajectory.get(tool_name_key)
            tool_args = trajectory.get(tool_args_key)
            observation = trajectory.get(observation_key)

            # Normalize tool_args to dict
            if tool_args is None:
                tool_args = {}
            elif not isinstance(tool_args, dict):
                tool_args = {"value": tool_args}

            is_final = tool_name == "finish" if tool_name else False

            step = OrchestrationStep(
                step_number=idx + 1,
                reasoning=thought,
                action=tool_name,
                action_input=tool_args,
                observation=observation,
                is_final=is_final,
            )
            steps.append(step)
            idx += 1

        return steps

    def _log_trace_summary(self) -> None:
        """Log a compact trace summary."""
        id_prefix = f"[{self.execution_id}] " if self.execution_id else ""
        logger.info(f"{id_prefix}{'─' * 50}")
        logger.info(f"{id_prefix}TRACE SUMMARY")
        logger.info(f"{id_prefix}{'─' * 50}")
        for step in self.steps:
            if step.is_final:
                logger.info(
                    f"{id_prefix}Step {step.step_number} [FINAL]: {step.action}"
                )
            else:
                # Log full error at ERROR level for debugging
                if step.observation and "Execution error" in step.observation:
                    logger.error(
                        f"{id_prefix}Step {step.step_number} execution failed:\n"
                        f"{step.observation}"
                    )

                obs_preview = (
                    (step.observation[:80] + "...")
                    if step.observation and len(step.observation) > 80
                    else step.observation
                )
                logger.info(
                    f"{id_prefix}Step {step.step_number}: {step.action} -> {obs_preview}"
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
