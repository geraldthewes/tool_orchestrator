"""
Configuration models for ToolOrchestra.

Defines dataclasses for the unified YAML configuration file.
"""

from dataclasses import dataclass, field
from typing import Optional

from .delegate import DelegateConfig


@dataclass
class OrchestratorConfig:
    """Configuration for the main orchestrator model."""
    base_url: str = "http://localhost:8001/v1"
    model: str = "nvidia/Nemotron-Orchestrator-8B"
    temperature: float = 0.7
    max_steps: int = 10


@dataclass
class ServerConfig:
    """Configuration for the FastAPI server."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False


@dataclass
class PythonExecutorConfig:
    """Configuration for the Python executor tool."""
    url: str = "http://pyexec.cluster:9999/"
    timeout: int = 30


@dataclass
class ToolsConfig:
    """Configuration for tool endpoints."""
    searxng_endpoint: str = "http://localhost:8080/search"
    python_executor: PythonExecutorConfig = field(default_factory=PythonExecutorConfig)

    # Backward compatibility properties
    @property
    def python_executor_url(self) -> str:
        """Backward compatibility alias for python_executor.url."""
        return self.python_executor.url

    @property
    def python_timeout(self) -> int:
        """Backward compatibility alias for python_executor.timeout."""
        return self.python_executor.timeout


@dataclass
class FastPathConfig:
    """Configuration for fast-path routing of simple queries."""
    enabled: bool = True


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"


@dataclass
class LangfuseConfig:
    """Configuration for Langfuse observability.

    Tracing auto-enables when both public_key and secret_key are provided.
    """
    enabled: bool = False
    public_key: str = ""
    secret_key: str = ""
    host: str = "https://cloud.langfuse.com"
    flush_at: int = 10
    flush_interval: float = 1.0
    debug: bool = False
    output_max_length: int = 0

    @property
    def is_configured(self) -> bool:
        """Check if Langfuse is configured (both keys present)."""
        return bool(self.public_key and self.secret_key)


@dataclass
class DSPyConfig:
    """Configuration for DSPy prompt optimization."""
    optimized_prompts_path: str = ""


@dataclass
class AppConfig:
    """
    Unified application configuration container.

    Holds all configuration sections loaded from config/config.yaml.
    """
    version: str = "1.0"
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    fast_path: FastPathConfig = field(default_factory=FastPathConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    langfuse: LangfuseConfig = field(default_factory=LangfuseConfig)
    dspy: DSPyConfig = field(default_factory=DSPyConfig)
    delegates: dict[str, DelegateConfig] = field(default_factory=dict)

    @property
    def log_level(self) -> str:
        """Backward compatibility alias for logging.level."""
        return self.logging.level
