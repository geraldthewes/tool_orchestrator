"""
LLM Delegation Tools

Provides tools for delegating tasks to specialized LLMs.
Supports both OpenAI-compatible endpoints and Ollama.
"""

import logging
from openai import OpenAI
import requests

from ..config import config
from ..models import ConnectionType, DelegateConnection

logger = logging.getLogger(__name__)


def call_delegate(
    connection: DelegateConnection,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    timeout: int = 120,
) -> dict:
    """
    Call a delegate LLM using the appropriate protocol.

    Args:
        connection: Connection details for the delegate LLM
        prompt: The task or question
        temperature: Sampling temperature
        max_tokens: Maximum response tokens (uses config.max_tokens if None)
        timeout: Request timeout in seconds

    Returns:
        Dictionary with response or error
    """
    resolved_max_tokens = max_tokens if max_tokens is not None else config.max_tokens
    if connection.type == ConnectionType.OPENAI_COMPATIBLE:
        return _call_openai_compatible(
            connection, prompt, temperature, resolved_max_tokens, timeout
        )
    elif connection.type == ConnectionType.OLLAMA:
        return _call_ollama(connection, prompt, temperature, timeout)
    else:
        return {
            "success": False,
            "model": "unknown",
            "response": None,
            "error": f"Unknown connection type: {connection.type}",
        }


def _call_openai_compatible(
    connection: DelegateConnection,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> dict:
    """Call an OpenAI-compatible endpoint."""
    try:
        api_key = connection.api_key or "dummy"
        client = OpenAI(base_url=connection.base_url, api_key=api_key, timeout=timeout)
        response = client.chat.completions.create(
            model=connection.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return {
            "success": True,
            "model": connection.model,
            "response": response.choices[0].message.content,
            "error": None,
        }
    except Exception as e:
        logger.error(f"OpenAI-compatible call failed ({connection.base_url}): {e}")
        return {
            "success": False,
            "model": connection.model,
            "response": None,
            "error": str(e),
        }


def _call_ollama(
    connection: DelegateConnection,
    prompt: str,
    temperature: float,
    timeout: int,
) -> dict:
    """Call an Ollama endpoint."""
    try:
        response = requests.post(
            connection.base_url,
            json={
                "model": connection.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        return {
            "success": True,
            "model": connection.model,
            "response": data["message"]["content"],
            "error": None,
        }
    except Exception as e:
        logger.error(f"Ollama call failed ({connection.base_url}): {e}")
        return {
            "success": False,
            "model": connection.model,
            "response": None,
            "error": str(e),
        }


def format_result_for_llm(delegate_result: dict) -> str:
    """
    Format delegation result for the orchestrator.

    Args:
        delegate_result: Result from a delegate call

    Returns:
        Formatted string
    """
    if not delegate_result["success"]:
        return f"Delegation to {delegate_result['model']} failed: {delegate_result['error']}"

    return f"Response from {delegate_result['model']}:\n\n{delegate_result['response']}"


def get_delegate(role: str):
    """
    Get a delegate configuration by role name.

    Args:
        role: The delegate role (e.g., 'fast', 'reasoner', 'coder')

    Returns:
        DelegateConfig or None if not found
    """
    from ..config import config

    return config.delegates.get(role)


def call_delegate_by_role(
    role: str,
    prompt: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> dict:
    """
    Call a delegate LLM by role name.

    Args:
        role: The delegate role (e.g., 'fast', 'reasoner', 'coder')
        prompt: The task or question
        temperature: Sampling temperature (uses delegate default if None)
        max_tokens: Maximum response tokens (uses config.max_tokens if None)

    Returns:
        Dictionary with response or error
    """
    delegate = get_delegate(role)
    if delegate is None:
        return {
            "success": False,
            "model": "unknown",
            "response": None,
            "error": f"Unknown delegate role: {role}",
        }
    return call_delegate(
        connection=delegate.connection,
        prompt=prompt,
        temperature=(
            temperature if temperature is not None else delegate.defaults.temperature
        ),
        max_tokens=max_tokens,  # Falls back to config.max_tokens in call_delegate
        timeout=delegate.defaults.timeout,
    )
