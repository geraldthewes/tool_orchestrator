"""
Configuration management for ToolOrchestra.

Loads all configuration from environment variables with sensible defaults
for local development.
"""

import os
from dataclasses import dataclass

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class OrchestratorConfig:
    """Configuration for the main orchestrator model."""
    base_url: str = os.getenv("ORCHESTRATOR_BASE_URL", "http://localhost:8001/v1")
    model: str = os.getenv("ORCHESTRATOR_MODEL", "nvidia/Nemotron-Orchestrator-8B")
    temperature: float = float(os.getenv("ORCHESTRATOR_TEMPERATURE", "0.7"))
    max_steps: int = int(os.getenv("MAX_ORCHESTRATION_STEPS", "10"))


@dataclass
class DelegateLLMConfig:
    """Configuration for delegate LLM endpoints."""
    # Reasoning LLM (for complex reasoning tasks)
    reasoning_llm_url: str = os.getenv("REASONING_LLM_BASE_URL", "http://localhost:30000/v1")
    reasoning_llm_model: str = os.getenv("REASONING_LLM_MODEL", "openai/gpt-oss-120b")

    # Coding LLM (for code generation tasks)
    coding_llm_url: str = os.getenv("CODING_LLM_BASE_URL", "http://localhost:8000/v1")
    coding_llm_model: str = os.getenv("CODING_LLM_MODEL", "qwen3-coder")

    # Fast LLM (for quick reasoning, Ollama endpoint)
    fast_llm_url: str = os.getenv("FAST_LLM_URL", "http://localhost:11434/api/chat")
    fast_llm_model: str = os.getenv("FAST_LLM_MODEL", "nemotron-3-nano")


@dataclass
class ToolConfig:
    """Configuration for tool endpoints."""
    searxng_endpoint: str = os.getenv("SEARXNG_ENDPOINT", "http://localhost:8080/search")
    python_timeout: int = int(os.getenv("PYTHON_EXECUTOR_TIMEOUT", "30"))


@dataclass
class Config:
    """Main configuration container."""
    orchestrator: OrchestratorConfig
    delegates: DelegateLLMConfig
    tools: ToolConfig
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


def get_config() -> Config:
    """Get the application configuration."""
    return Config(
        orchestrator=OrchestratorConfig(),
        delegates=DelegateLLMConfig(),
        tools=ToolConfig(),
    )


# Global config instance
config = get_config()
