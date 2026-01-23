"""
Configuration management for ToolOrchestra.

Loads all configuration from config/config.yaml with environment variable
interpolation support. Values can use ${VAR} or ${VAR:-default} syntax.
"""

from .config_loader import load_app_config
from .models import (
    AppConfig,
    OrchestratorConfig,
    ServerConfig,
    ToolsConfig,
    PythonExecutorConfig,
    FastPathConfig,
    LoggingConfig,
    LangfuseConfig,
    DSPyConfig,
)

# Re-export config models for backward compatibility
__all__ = [
    "config",
    "get_config",
    "AppConfig",
    "OrchestratorConfig",
    "ServerConfig",
    "ToolsConfig",
    "PythonExecutorConfig",
    "FastPathConfig",
    "LoggingConfig",
    "LangfuseConfig",
    "DSPyConfig",
]


# Type alias for backward compatibility
Config = AppConfig


def get_config() -> AppConfig:
    """Get the application configuration."""
    return load_app_config()


# Global config instance
config = get_config()
