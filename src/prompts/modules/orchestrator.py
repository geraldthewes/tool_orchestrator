"""
DSPy Module for Tool Orchestration.

Implements ReAct-style reasoning using DSPy's built-in ReAct module
with adapters for existing ToolOrchestra tools.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, Any

import dspy

from ..signatures import ToolOrchestrationTask
from ..adapters import get_orchestrator_lm
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
    handler: Callable[[dict], Any],
    formatter: Callable[[Any], str],
    tracing_context: Optional[TracingContext] = None,
) -> Callable:
    """
    Create a DSPy-compatible tool function from handler and formatter.

    Args:
        name: Tool name
        description: Tool description
        handler: Function that executes the tool
        formatter: Function that formats the result
        tracing_context: Optional tracing context for observability

    Returns:
        DSPy-compatible tool function
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
        dspy.settings.adapter = dspy.JSONAdapter(use_native_function_calling=True)

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

        # Use existing LM from context if available (e.g., during optimization)
        # Otherwise, create the default orchestrator LM
        current_lm = dspy.settings.lm
        if current_lm is not None:
            logger.debug("Using LM from DSPy context")
            orchestrator_lm = current_lm
        else:
            orchestrator_lm = get_orchestrator_lm()
            # Wrap with tracing if available (only for newly created LM)
            if self.tracing_context:
                orchestrator_lm = TracedLM(
                    orchestrator_lm, self.tracing_context, "orchestrator"
                )

        try:
            with dspy.context(lm=orchestrator_lm):
                result = self.react(question=question)

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
