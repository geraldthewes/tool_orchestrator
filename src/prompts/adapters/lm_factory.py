"""
DSPy Language Model Factory for ToolOrchestra.

Provides factory functions for creating DSPy LM instances from existing
configuration for the orchestrator and delegate LLMs.
"""

import logging
from typing import Any, Optional

import dspy

from ...config import config
from ...config_loader import load_delegates_config
from ...models import DelegatesConfiguration
from ...tracing import TracingContext

logger = logging.getLogger(__name__)

# Cache for delegates configuration
_delegates_config: Optional[DelegatesConfiguration] = None


def _get_delegates_config() -> DelegatesConfiguration:
    """Get or load the delegates configuration (cached)."""
    global _delegates_config
    if _delegates_config is None:
        _delegates_config = load_delegates_config()
    return _delegates_config


def get_orchestrator_lm(
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> dspy.LM:
    """
    Create a DSPy LM instance for the orchestrator model.

    Uses configuration from environment variables via config module.

    Args:
        temperature: Override temperature (uses config default if None)
        max_tokens: Override max tokens (uses 1024 if None)

    Returns:
        Configured DSPy LM instance for orchestrator
    """
    base_url = config.orchestrator.base_url
    model = config.orchestrator.model
    temp = temperature if temperature is not None else config.orchestrator.temperature

    # Build the full model identifier for DSPy
    # DSPy uses "openai/<model>" format for OpenAI-compatible endpoints
    model_id = f"openai/{model}"

    logger.debug(
        f"Creating orchestrator LM: model={model_id}, "
        f"base_url={base_url}, temperature={temp}"
    )

    lm = dspy.LM(
        model=model_id,
        api_base=base_url,
        api_key="not-needed",  # Most local deployments don't need keys
        temperature=temp,
        max_tokens=max_tokens or 1024,
    )

    return lm


def get_delegate_lm(
    role: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    delegates_config: Optional[DelegatesConfiguration] = None,
) -> dspy.LM:
    """
    Create a DSPy LM instance for a delegate LLM by role.

    Args:
        role: The delegate role (e.g., "reasoner", "coder", "fast")
        temperature: Override temperature (uses delegate default if None)
        max_tokens: Override max tokens (uses delegate default if None)
        delegates_config: Optional pre-loaded delegates config

    Returns:
        Configured DSPy LM instance for the delegate

    Raises:
        KeyError: If the role is not found in configuration
    """
    config_to_use = delegates_config or _get_delegates_config()

    if role not in config_to_use.delegates:
        available = list(config_to_use.delegates.keys())
        raise KeyError(
            f"Delegate role '{role}' not found. Available roles: {available}"
        )

    delegate = config_to_use.delegates[role]
    conn = delegate.connection
    defaults = delegate.defaults

    # Use provided values or fall back to defaults
    temp = temperature if temperature is not None else defaults.temperature
    tokens = max_tokens if max_tokens is not None else defaults.max_tokens

    # Clamp to capability limit
    if tokens > delegate.capabilities.max_output_tokens:
        logger.warning(
            f"Requested max_tokens ({tokens}) exceeds limit "
            f"({delegate.capabilities.max_output_tokens}) for {role}, clamping"
        )
        tokens = delegate.capabilities.max_output_tokens

    # Build model identifier
    model_id = f"openai/{conn.model}"

    logger.debug(
        f"Creating delegate LM for '{role}': model={model_id}, "
        f"base_url={conn.base_url}, temperature={temp}"
    )

    lm = dspy.LM(
        model=model_id,
        api_base=conn.base_url,
        api_key=conn.api_key or "not-needed",
        temperature=temp,
        max_tokens=tokens,
    )

    return lm


def get_fast_lm(
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> dspy.LM:
    """
    Convenience function to get the fast delegate LM.

    Args:
        temperature: Override temperature
        max_tokens: Override max tokens

    Returns:
        Configured DSPy LM instance for the fast delegate
    """
    return get_delegate_lm("fast", temperature=temperature, max_tokens=max_tokens)


def configure_dspy_default(lm: Optional[dspy.LM] = None) -> None:
    """
    Configure DSPy's default language model.

    Args:
        lm: LM to set as default. If None, uses the orchestrator LM.
    """
    if lm is None:
        lm = get_orchestrator_lm()
    dspy.configure(lm=lm)
    logger.debug(f"Configured DSPy default LM: {lm}")


class TracedLM:
    """
    Wrapper that adds Langfuse tracing to DSPy LM calls.

    Intercepts DSPy's LM calls and creates Langfuse generation spans for
    observability, including prompts, completions, and token usage.
    """

    def __init__(
        self,
        lm: dspy.LM,
        tracing_context: TracingContext,
        name: str = "orchestrator",
    ):
        """
        Initialize the traced LM wrapper.

        Args:
            lm: The underlying DSPy LM instance to wrap
            tracing_context: Tracing context for creating generation spans
            name: Name prefix for generation spans (e.g., "orchestrator")
        """
        self._lm = lm
        self._tracing_context = tracing_context
        self._name = name

    def __call__(
        self,
        prompt: Optional[str] = None,
        messages: Optional[list] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Call the underlying LM with tracing.

        Args:
            prompt: Optional prompt string
            messages: Optional list of messages
            **kwargs: Additional arguments to pass to the LM

        Returns:
            The LM response
        """
        # Prepare input for tracing
        if messages:
            input_data = {"messages": messages}
        elif prompt:
            input_data = {"prompt": prompt[:1000] if len(prompt) > 1000 else prompt}
        else:
            input_data = {}

        # Extract model parameters for tracing
        model_params = {}
        if "temperature" in kwargs:
            model_params["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            model_params["max_tokens"] = kwargs["max_tokens"]

        with self._tracing_context.generation(
            name=f"llm:{self._name}",
            model=self._lm.model,
            input=input_data,
            model_parameters=model_params if model_params else None,
        ) as gen:
            result = self._lm(prompt=prompt, messages=messages, **kwargs)

            # Extract and record output
            output_str = str(result)[:1000] if result else ""
            gen.set_output(output_str)

            # Try to extract usage from result if available
            if hasattr(result, "usage"):
                usage = result.usage
                if hasattr(usage, "prompt_tokens"):
                    gen.set_usage(
                        prompt_tokens=getattr(usage, "prompt_tokens", None),
                        completion_tokens=getattr(usage, "completion_tokens", None),
                        total_tokens=getattr(usage, "total_tokens", None),
                    )

            return result

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying LM."""
        return getattr(self._lm, name)
