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
    """Configuration for delegate LLM endpoints.

    Note: Delegate LLMs are now configured via config/delegates.yaml.
    The settings below are only used by legacy code paths.
    """
    # Path to delegates YAML configuration
    delegates_config_path: str = os.getenv("DELEGATES_CONFIG_PATH", "")

    # Legacy settings (kept for backward compatibility with llm_delegate.py functions)
    reasoning_llm_url: str = os.getenv("REASONING_LLM_BASE_URL", "http://gx10-d8ce.cluster:8000/v1")
    reasoning_llm_model: str = os.getenv("REASONING_LLM_MODEL", "glm-reap")
    coding_llm_url: str = os.getenv("CODING_LLM_BASE_URL", "http://localhost:8000/v1")
    coding_llm_model: str = os.getenv("CODING_LLM_MODEL", "qwen3-coder")
    fast_llm_url: str = os.getenv("FAST_LLM_URL", "http://localhost:11434/api/chat")
    fast_llm_model: str = os.getenv("FAST_LLM_MODEL", "nemotron-3-nano")


@dataclass
class ToolConfig:
    """Configuration for tool endpoints."""
    searxng_endpoint: str = os.getenv("SEARXNG_ENDPOINT", "http://localhost:8080/search")
    python_executor_url: str = os.getenv("PYTHON_EXECUTOR_URL", "http://pyexec.cluster:9999/")
    python_timeout: int = int(os.getenv("PYTHON_EXECUTOR_TIMEOUT", "30"))


@dataclass
class ServerConfig:
    """Configuration for the FastAPI server."""
    host: str = os.getenv("SERVER_HOST", "0.0.0.0")
    port: int = int(os.getenv("SERVER_PORT", "8000"))
    workers: int = int(os.getenv("SERVER_WORKERS", "1"))
    reload: bool = os.getenv("SERVER_RELOAD", "false").lower() == "true"


@dataclass
class Config:
    """Main configuration container."""
    orchestrator: OrchestratorConfig
    delegates: DelegateLLMConfig
    tools: ToolConfig
    server: ServerConfig
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


def get_config() -> Config:
    """Get the application configuration."""
    return Config(
        orchestrator=OrchestratorConfig(),
        delegates=DelegateLLMConfig(),
        tools=ToolConfig(),
        server=ServerConfig(),
    )


# Global config instance
config = get_config()
