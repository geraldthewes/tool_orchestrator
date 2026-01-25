"""
DSPy Module for Tool Orchestration.

Implements ReAct-style reasoning using DSPy's built-in ReAct module
with adapters for existing ToolOrchestra tools.
"""

import inspect
import logging
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

    Returns:
        DSPy-compatible tool function with explicit parameter signature
    """

    def tool_func(**kwargs) -> str:
        """Execute the tool and return formatted result."""
        if tracing_context:
            with tracing_context.span(
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
                    return f"Tool execution error: {e}"
        else:
            # Original logic without tracing
            try:
                raw_result = handler(kwargs)
                formatted = formatter(raw_result)
                return formatted
            except Exception as e:
                logger.error(f"Tool execution failed: {name} - {e}")
                return f"Tool execution error: {e}"

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
    tool_func.__annotations__ = {
        param_name: str for param_name in parameters.keys()
    }
    tool_func.__annotations__["return"] = str

    return tool_func


def create_delegate_tool(
    role: str,
    display_name: str,
    description: str,
    tool_name: str,
    delegates_config: DelegatesConfiguration,
    tracing_context: Optional[TracingContext] = None,
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

    Returns:
        DSPy-compatible tool function
    """
    from ...tools.llm_delegate import call_delegate, format_result_for_llm

    delegate = delegates_config.delegates[role]

    def delegate_tool(
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Delegate to specialized LLM."""
        if not prompt or not prompt.strip():
            return "Error: Prompt is empty. Please provide a prompt."

        temp = temperature if temperature is not None else delegate.defaults.temperature
        tokens = max_tokens if max_tokens is not None else delegate.defaults.max_tokens

        # Clamp max_tokens
        if tokens > delegate.capabilities.max_output_tokens:
            tokens = delegate.capabilities.max_output_tokens

        if tracing_context:
            with tracing_context.span(
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
    ):
        """
        Initialize the orchestrator module.

        Args:
            max_steps: Maximum number of reasoning steps
            verbose: Enable verbose logging
            delegates_config: Configuration for delegate LLMs
            execution_id: Optional ID for correlating logs
            tracing_context: Optional tracing context for observability
        """
        super().__init__()

        self.max_steps = max_steps
        self.verbose = verbose
        self.execution_id = execution_id
        self.tracing_context = tracing_context
        self.steps: list[OrchestrationStep] = []

        # Load delegates configuration
        self.delegates_config = delegates_config or get_delegates_from_app_config()

        # Build DSPy tools list
        self._tools = self._build_dspy_tools()

        # Configure adapter for native function calling (Nemotron compatibility)
        # Use NemotronJSONAdapter to handle "final" wrapper in responses
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

    def _build_dspy_tools(self) -> list[Callable]:
        """Build list of DSPy-compatible tool functions."""
        tools = []

        # Add tools from registry (static tools)
        for name, tool_def in ToolRegistry.all_tools().items():
            dspy_tool = create_dspy_tool(
                name=name,
                description=tool_def.description,
                parameters=tool_def.parameters,
                handler=tool_def.handler,
                formatter=tool_def.formatter,
                tracing_context=self.tracing_context,
            )
            tools.append(dspy_tool)
            logger.debug(f"Registered DSPy tool: {name}")

        # Add delegate tools
        for role, delegate in self.delegates_config.delegates.items():
            delegate_tool = create_delegate_tool(
                role=role,
                display_name=delegate.display_name,
                description=delegate.description,
                tool_name=delegate.tool_name,
                delegates_config=self.delegates_config,
                tracing_context=self.tracing_context,
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

        Args:
            question: The user's question or task

        Returns:
            The final answer string
        """
        self.steps = []
        logger.debug(f"Starting orchestration for: {question}")

        # Always use the orchestrator LM for ReAct tool-calling
        # This ensures the correct model is used even during DSPy optimization
        # (where a teacher LM may be set in dspy.settings.lm for prompt generation)
        orchestrator_lm = get_orchestrator_lm()
        if self.tracing_context:
            orchestrator_lm = TracedLM(
                orchestrator_lm, self.tracing_context, "orchestrator"
            )

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

            # Log completion
            logger.info("Completed orchestration")
            self._log_trace_summary()

            # Return dspy.Prediction for proper metric evaluation during optimization
            return dspy.Prediction(answer=answer)

        except Exception as e:
            logger.error(
                f"Orchestration failed: {e} "
                f"(endpoint={config.orchestrator.base_url}, model={config.orchestrator.model})"
            )
            return dspy.Prediction(answer=f"Error during orchestration: {e}")
        finally:
            # Always reset query context
            reset_current_query(query_token)

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
