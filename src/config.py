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

    Delegate LLMs are configured via config/delegates.yaml.
    """
    # Path to delegates YAML configuration
    delegates_config_path: str = os.getenv("DELEGATES_CONFIG_PATH", "")


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
class FastPathConfig:
    """Configuration for fast-path routing of simple queries."""
    enabled: bool = os.getenv("FAST_PATH_ENABLED", "true").lower() == "true"


@dataclass
class DSPyConfig:
    """Configuration for DSPy prompt optimization."""
    optimized_prompts_path: str = os.getenv("DSPY_OPTIMIZED_PROMPTS_PATH", "")


@dataclass
class LangfuseConfig:
    """Configuration for Langfuse observability.

    Tracing auto-enables when both public_key and secret_key are provided.
    """
    public_key: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    secret_key: str = os.getenv("LANGFUSE_SECRET_KEY", "")
    host: str = os.getenv("LANGFUSE_HOST", "")
    flush_at: int = int(os.getenv("LANGFUSE_FLUSH_AT", "10"))
    flush_interval: float = float(os.getenv("LANGFUSE_FLUSH_INTERVAL", "1.0"))
    debug: bool = os.getenv("LANGFUSE_DEBUG", "false").lower() == "true"
    output_max_length: int = int(os.getenv("LANGFUSE_OUTPUT_MAX_LENGTH", "0"))

    @property
    def enabled(self) -> bool:
        """Auto-enable when both keys are configured."""
        return bool(self.public_key and self.secret_key)


@dataclass
class Config:
    """Main configuration container."""
    orchestrator: OrchestratorConfig
    delegates: DelegateLLMConfig
    tools: ToolConfig
    server: ServerConfig
    fast_path: FastPathConfig
    langfuse: LangfuseConfig
    dspy: DSPyConfig
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


def get_config() -> Config:
    """Get the application configuration."""
    return Config(
        orchestrator=OrchestratorConfig(),
        delegates=DelegateLLMConfig(),
        tools=ToolConfig(),
        server=ServerConfig(),
        fast_path=FastPathConfig(),
        langfuse=LangfuseConfig(),
        dspy=DSPyConfig(),
    )


# Global config instance
config = get_config()
