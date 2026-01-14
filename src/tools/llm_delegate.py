"""
LLM Delegation Tools

Provides tools for delegating tasks to specialized LLMs:
- Reasoning LLM - Complex reasoning tasks
- Coding LLM - Code generation tasks
- Fast LLM - Quick reasoning tasks
"""

import logging
from typing import Optional
from openai import OpenAI
import requests

from ..config import config

logger = logging.getLogger(__name__)


def call_reasoning_llm(
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> dict:
    """
    Call the reasoning LLM for complex reasoning tasks.

    Args:
        prompt: The task or question
        temperature: Sampling temperature
        max_tokens: Maximum response tokens

    Returns:
        Dictionary with response or error
    """
    try:
        client = OpenAI(base_url=config.delegates.reasoning_llm_url, api_key="dummy")
        response = client.chat.completions.create(
            model=config.delegates.reasoning_llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return {
            "success": True,
            "model": "reasoning-llm",
            "response": response.choices[0].message.content,
            "error": None,
        }
    except Exception as e:
        logger.error(f"Reasoning LLM call failed: {e}")
        return {
            "success": False,
            "model": "reasoning-llm",
            "response": None,
            "error": str(e),
        }


# Alias for backwards compatibility
call_gpt_oss = call_reasoning_llm


def call_coding_llm(
    prompt: str,
    temperature: float = 0.3,  # Lower temp for code generation
    max_tokens: int = 2048,
) -> dict:
    """
    Call the coding LLM for code generation tasks.

    Args:
        prompt: The coding task or question
        temperature: Sampling temperature
        max_tokens: Maximum response tokens

    Returns:
        Dictionary with response or error
    """
    try:
        client = OpenAI(base_url=config.delegates.coding_llm_url, api_key="dummy")
        response = client.chat.completions.create(
            model=config.delegates.coding_llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return {
            "success": True,
            "model": "coding-llm",
            "response": response.choices[0].message.content,
            "error": None,
        }
    except Exception as e:
        logger.error(f"Coding LLM call failed: {e}")
        return {
            "success": False,
            "model": "coding-llm",
            "response": None,
            "error": str(e),
        }


# Alias for backwards compatibility
call_qwen_coder = call_coding_llm


def call_fast_llm(
    prompt: str,
    temperature: float = 0.7,
) -> dict:
    """
    Call the fast LLM for quick reasoning tasks (Ollama endpoint).

    Args:
        prompt: The task or question
        temperature: Sampling temperature

    Returns:
        Dictionary with response or error
    """
    try:
        response = requests.post(
            config.delegates.fast_llm_url,
            json={
                "model": config.delegates.fast_llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return {
            "success": True,
            "model": "fast-llm",
            "response": data["message"]["content"],
            "error": None,
        }
    except Exception as e:
        logger.error(f"Fast LLM call failed: {e}")
        return {
            "success": False,
            "model": "fast-llm",
            "response": None,
            "error": str(e),
        }


# Alias for backwards compatibility
call_nemotron_nano = call_fast_llm


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
