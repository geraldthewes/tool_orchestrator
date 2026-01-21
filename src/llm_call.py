"""
LLM Call Interface for ToolOrchestra

Provides a unified interface for calling different LLM backends:
- vLLM (OpenAI-compatible)
- Ollama
- SGLang
"""

import logging
from typing import Optional
from openai import OpenAI
import requests

from .config import config

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified LLM client supporting multiple backends."""

    def __init__(
        self,
        orchestrator_url: Optional[str] = None,
        orchestrator_model: Optional[str] = None,
    ):
        self.orchestrator_url = orchestrator_url or config.orchestrator.base_url
        self.orchestrator_model = orchestrator_model or config.orchestrator.model
        self.orchestrator_client = OpenAI(
            base_url=self.orchestrator_url,
            api_key="dummy",  # vLLM doesn't require auth
        )

    def call_orchestrator(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> dict:
        """Call the main orchestrator model (Nemotron-Orchestrator-8B)."""
        try:
            response = self.orchestrator_client.chat.completions.create(
                model=self.orchestrator_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return {
                "success": True,
                "response": response.choices[0].message.content,
                "error": None,
            }
        except Exception as e:
            logger.error(f"Orchestrator call failed: {e}")
            return {
                "success": False,
                "response": None,
                "error": str(e),
            }

    def call_openai_compatible(
        self,
        base_url: str,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Call an OpenAI-compatible endpoint (vLLM, SGLang)."""
        client = OpenAI(base_url=base_url, api_key="dummy")
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI-compatible call to {base_url} failed: {e}")
            raise

    def call_ollama(
        self,
        endpoint: str,
        model: str,
        prompt: str,
    ) -> str:
        """Call an Ollama endpoint."""
        try:
            response = requests.post(
                endpoint,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                },
                timeout=120,
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
        except Exception as e:
            logger.error(f"Ollama call to {endpoint} failed: {e}")
            raise
