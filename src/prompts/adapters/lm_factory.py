"""
DSPy Language Model Factory for ToolOrchestra.

Provides factory functions for creating DSPy LM instances from existing
configuration for the orchestrator and delegate LLMs.
"""

import logging
import os
from typing import Any, Optional

import dspy

from ...config import config
from ...config_loader import get_delegates_from_app_config
from ...models import DelegatesConfiguration
from ...tracing import TracingContext

logger = logging.getLogger(__name__)

# Cache for delegates configuration
_delegates_config: Optional[DelegatesConfiguration] = None


def _estimate_tokens(text: str) -> int:
    """
    Estimate token count using character-based heuristic.

    Most LLMs average ~4 characters per token for English text.
    We use 3.5 to be conservative (overestimate tokens).
    """
    if not text:
        return 0
    return int(len(text) / 3.5)


def _estimate_messages_tokens(messages: list) -> int:
    """Estimate tokens for a list of chat messages."""
    total = 0
    for msg in messages:
        # Role token overhead (~4 tokens per message for role, formatting)
        total += 4
        content = (
            msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
        )
        if isinstance(content, str):
            total += _estimate_tokens(content)
        elif isinstance(content, list):
            # Handle multi-part content (text, images, etc.)
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    total += _estimate_tokens(part["text"])
    return total


class TokenAwareLM(dspy.LM):
    """
    LM wrapper that dynamically adjusts max_tokens based on input size.

    Prevents output truncation when input is large by calculating
    available output tokens = context_length - estimated_input - buffer.
    """

    def __init__(
        self,
        lm: dspy.LM,
        context_length: int,
        min_output_tokens: int = 256,
        safety_buffer: int = 100,
    ):
        self._lm = lm
        self._context_length = context_length
        self._min_output_tokens = min_output_tokens
        self._safety_buffer = safety_buffer

        # Copy required attributes from wrapped LM
        self.model = lm.model
        self.cache = lm.cache
        self.history = lm.history
        self.callbacks = lm.callbacks
        self.kwargs = lm.kwargs

    def forward(
        self,
        prompt: Optional[str] = None,
        messages: Optional[list] = None,
        **kwargs: Any,
    ) -> Any:
        # Estimate input tokens
        if messages:
            estimated_input = _estimate_messages_tokens(messages)
        elif prompt:
            estimated_input = _estimate_tokens(prompt)
        else:
            estimated_input = 0

        # Calculate available output tokens
        available = self._context_length - estimated_input - self._safety_buffer

        logger.debug(
            f"TokenAwareLM.forward: input ~{estimated_input} tokens, "
            f"available ~{available} tokens, context {self._context_length}"
        )

        # Check if input exceeds context length
        if available <= 0:
            logger.error(
                f"Input too large for context window: ~{estimated_input} tokens "
                f"exceeds context_length {self._context_length} - {self._safety_buffer} buffer. "
                f"Response will likely be truncated or fail."
            )

        # Get requested max_tokens
        requested = kwargs.get("max_tokens") or self._lm.kwargs.get(
            "max_tokens", config.max_tokens
        )

        # Clamp if needed
        if available < requested:
            adjusted = max(self._min_output_tokens, available)
            logger.warning(
                f"Adjusting max_tokens: {requested} -> {adjusted} "
                f"(input ~{estimated_input} tokens, context {self._context_length})"
            )
            kwargs["max_tokens"] = adjusted

        result = self._lm.forward(prompt=prompt, messages=messages, **kwargs)

        # Ensure usage is preserved (handle caching side effects)
        # DSPy's caching mechanism can replace the result with a cached version
        # that has an empty usage dict. Try to recover from underlying LM history.
        if hasattr(result, "usage"):
            usage = result.usage
            is_empty_usage = (
                isinstance(usage, dict)
                and not usage.get("prompt_tokens")
                and not usage.get("completion_tokens")
            )
            if is_empty_usage and hasattr(self._lm, "history") and self._lm.history:
                last_entry = self._lm.history[-1]
                if hasattr(last_entry, "usage") and last_entry.usage:
                    logger.debug("TokenAwareLM: Recovering usage from LM history")
                    result.usage = last_entry.usage

        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_lm"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in (
            "_lm",
            "_context_length",
            "_min_output_tokens",
            "_safety_buffer",
            "model",
            "cache",
            "history",
            "callbacks",
            "kwargs",
        ):
            object.__setattr__(self, name, value)
        else:
            setattr(self._lm, name, value)


def _get_delegates_config() -> DelegatesConfiguration:
    """Get or load the delegates configuration (cached)."""
    global _delegates_config
    if _delegates_config is None:
        _delegates_config = get_delegates_from_app_config()
    return _delegates_config


def get_teacher_lm(
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> dspy.LM:
    """
    Create a DSPy LM instance for use as a teacher model in DSPy optimization.

    Configuration priority (highest to lowest):
    1. Function parameters (base_url, model, max_tokens)
    2. Environment variables (TEACHER_BASE_URL, TEACHER_MODEL)
    3. Config file (max_tokens)

    Args:
        base_url: Override base URL (uses TEACHER_BASE_URL env var if None)
        model: Override model name (uses TEACHER_MODEL env var if None)
        temperature: Override temperature (defaults to 0.7 if None)
        max_tokens: Override max tokens (uses config.max_tokens if None)

    Returns:
        Configured DSPy LM instance for teacher

    Raises:
        ValueError: If teacher config is missing (no base_url or model)
    """
    resolved_base_url = base_url or os.environ.get("TEACHER_BASE_URL")
    resolved_model = model or os.environ.get("TEACHER_MODEL")

    if not resolved_base_url:
        raise ValueError(
            "Teacher LLM base URL not configured. "
            "Set TEACHER_BASE_URL environment variable or pass --teacher-base-url"
        )
    if not resolved_model:
        raise ValueError(
            "Teacher LLM model not configured. "
            "Set TEACHER_MODEL environment variable or pass --teacher-model"
        )

    temp = temperature if temperature is not None else 0.7
    resolved_max_tokens = max_tokens if max_tokens is not None else config.max_tokens

    # Build the full model identifier for DSPy
    # DSPy uses "openai/<model>" format for OpenAI-compatible endpoints
    model_id = f"openai/{resolved_model}"

    full_url = f"{resolved_base_url.rstrip('/')}/chat/completions"
    logger.info(
        f"Creating teacher LM: model={model_id}, "
        f"endpoint={full_url}, temperature={temp}"
    )

    lm = dspy.LM(
        model=model_id,
        api_base=resolved_base_url,
        api_key="not-needed",  # Most local deployments don't need keys
        temperature=temp,
        max_tokens=resolved_max_tokens,
    )

    return lm


def get_orchestrator_lm(
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> dspy.LM:
    """
    Create a DSPy LM instance for the orchestrator model.

    Uses configuration from environment variables via config module.

    Args:
        temperature: Override temperature (uses config default if None)
        max_tokens: Override max tokens (uses config.max_tokens if None)

    Returns:
        Configured DSPy LM instance for orchestrator
    """
    base_url = config.orchestrator.base_url
    model = config.orchestrator.model
    temp = temperature if temperature is not None else config.orchestrator.temperature
    tokens = max_tokens if max_tokens is not None else config.max_tokens

    # Build the full model identifier for DSPy
    # DSPy uses "openai/<model>" format for OpenAI-compatible endpoints
    model_id = f"openai/{model}"

    full_url = f"{base_url.rstrip('/')}/chat/completions"
    stop_info = f", stop={config.orchestrator.stop}" if config.orchestrator.stop else ""
    logger.info(
        f"Creating orchestrator LM: model={model_id}, "
        f"endpoint={full_url}, temperature={temp}, max_tokens={tokens}, "
        f"frequency_penalty={config.orchestrator.frequency_penalty}, "
        f"presence_penalty={config.orchestrator.presence_penalty}{stop_info}"
    )

    # Build LM kwargs, only include stop if configured
    lm_kwargs: dict[str, Any] = {
        "model": model_id,
        "api_base": base_url,
        "api_key": "not-needed",  # Most local deployments don't need keys
        "temperature": temp,
        "max_tokens": tokens,
        "frequency_penalty": config.orchestrator.frequency_penalty,
        "presence_penalty": config.orchestrator.presence_penalty,
    }
    if config.orchestrator.stop:
        lm_kwargs["stop"] = config.orchestrator.stop

    lm = dspy.LM(**lm_kwargs)

    # Wrap with TokenAwareLM to prevent output truncation on large inputs
    context_length = config.orchestrator.context_length
    lm = TokenAwareLM(lm, context_length=context_length)

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
        max_tokens: Override max tokens (uses config.max_tokens if None)
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

    # Use provided values or fall back to global config
    temp = temperature if temperature is not None else defaults.temperature
    tokens = max_tokens if max_tokens is not None else config.max_tokens

    # Clamp to capability limit
    if tokens > delegate.capabilities.max_output_tokens:
        logger.warning(
            f"Requested max_tokens ({tokens}) exceeds limit "
            f"({delegate.capabilities.max_output_tokens}) for {role}, clamping"
        )
        tokens = delegate.capabilities.max_output_tokens

    # Build model identifier
    model_id = f"openai/{conn.model}"

    full_url = f"{conn.base_url.rstrip('/')}/chat/completions"
    logger.info(
        f"Creating delegate LM for '{role}': model={model_id}, "
        f"endpoint={full_url}, temperature={temp}"
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


class TracedLM(dspy.LM):
    """
    Wrapper that adds Langfuse tracing to DSPy LM calls.

    Inherits from dspy.LM to pass isinstance checks while delegating
    actual LLM calls to a wrapped LM instance.
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
        # Don't call super().__init__() - we're wrapping an existing LM
        # Instead, copy required attributes from the wrapped LM
        object.__setattr__(self, "_lm", lm)
        object.__setattr__(self, "_tracing_context", tracing_context)
        object.__setattr__(self, "_name", name)

        # Copy core attributes from wrapped LM to satisfy BaseLM interface
        # These are checked by DSPy internals
        self.model = lm.model
        self.cache = lm.cache
        self.history = lm.history
        self.callbacks = lm.callbacks
        self.kwargs = lm.kwargs

    def forward(
        self,
        prompt: Optional[str] = None,
        messages: Optional[list] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Override forward() to add tracing around the actual LM call.

        DSPy's BaseLM.__call__ invokes forward() internally, so this is
        the correct place to add tracing.

        Args:
            prompt: Optional prompt string
            messages: Optional list of messages
            **kwargs: Additional arguments to pass to the LM

        Returns:
            The LM response
        """
        # Prepare input for tracing
        input_data: dict[str, Any]
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
            # Call the underlying LM's forward method
            result = self._lm.forward(prompt=prompt, messages=messages, **kwargs)

            # Extract and record output
            # Use configurable max length (0 = unlimited)
            max_len = config.langfuse.output_max_length
            if result:
                output_str = str(result)
                if max_len > 0:
                    output_str = output_str[:max_len]
            else:
                output_str = ""
            gen.set_output(output_str)

            # Try to extract usage from result if available
            if hasattr(result, "usage"):
                usage = result.usage
                prompt_tokens = None
                completion_tokens = None
                total_tokens = None

                if isinstance(usage, dict):
                    # Handle dict-style usage (e.g., from cache or some providers)
                    prompt_tokens = usage.get("prompt_tokens")
                    completion_tokens = usage.get("completion_tokens")
                    total_tokens = usage.get("total_tokens")
                elif hasattr(usage, "prompt_tokens"):
                    # Handle object-style usage (LiteLLM Usage class)
                    prompt_tokens = getattr(usage, "prompt_tokens", None)
                    completion_tokens = getattr(usage, "completion_tokens", None)
                    total_tokens = getattr(usage, "total_tokens", None)

                # Only set usage if we have at least one valid token count
                if any(
                    v is not None
                    for v in [prompt_tokens, completion_tokens, total_tokens]
                ):
                    gen.set_usage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                    )

            return result

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying LM."""
        # Use object.__getattribute__ to avoid recursion
        lm = object.__getattribute__(self, "_lm")
        return getattr(lm, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Handle attribute setting - core attrs go to self, others to wrapped LM."""
        if name in (
            "_lm",
            "_tracing_context",
            "_name",
            "model",
            "cache",
            "history",
            "callbacks",
            "kwargs",
        ):
            object.__setattr__(self, name, value)
        else:
            setattr(self._lm, name, value)
