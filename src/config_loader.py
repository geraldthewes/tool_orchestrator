"""
Configuration loader for ToolOrchestra.

Loads configuration from YAML files with support for
environment variable interpolation.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

import yaml

from .models import (
    ConnectionType,
    DelegateConnection,
    DelegateCapabilities,
    DelegateDefaults,
    DelegateConfig,
    DelegatesConfiguration,
    OrchestratorConfig,
    ServerConfig,
    PythonExecutorConfig,
    SearxngConfig,
    ToolsConfig,
    FastPathConfig,
    LoggingConfig,
    LangfuseConfig,
    DSPyConfig,
    AppConfig,
)

logger = logging.getLogger(__name__)

# Default config paths relative to project root
DEFAULT_DELEGATES_CONFIG_PATH = (
    Path(__file__).parent.parent / "config" / "delegates.yaml"
)
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"

# Regex for environment variable interpolation: ${VAR} or ${VAR:-default}
ENV_VAR_PATTERN = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")

# Singleton cache for app config
_app_config: Optional[AppConfig] = None


def resolve_env_vars(value: str) -> str:
    """
    Resolve environment variable references in a string.

    Supports ${VAR} and ${VAR:-default} syntax.

    Args:
        value: String potentially containing env var references

    Returns:
        String with env vars resolved
    """

    def replace_match(match: re.Match) -> str:
        var_name = match.group(1)
        default_value = match.group(2) if match.group(2) is not None else ""
        return os.environ.get(var_name, default_value)

    return ENV_VAR_PATTERN.sub(replace_match, value)


def _parse_connection(data: dict) -> DelegateConnection:
    """Parse connection configuration from dict."""
    conn_type_str = data.get("type", "openai_compatible")
    try:
        conn_type = ConnectionType(conn_type_str)
    except ValueError:
        raise ValueError(f"Unknown connection type: {conn_type_str}")

    return DelegateConnection(
        type=conn_type,
        base_url=resolve_env_vars(data.get("base_url", "")),
        model=resolve_env_vars(data.get("model", "")),
        api_key=(
            resolve_env_vars(data.get("api_key", "")) if data.get("api_key") else None
        ),
    )


def _parse_capabilities(data: dict) -> DelegateCapabilities:
    """Parse capabilities configuration from dict."""
    return DelegateCapabilities(
        context_length=data.get("context_length", 4096),
        max_output_tokens=data.get("max_output_tokens", 2048),
        specializations=data.get("specializations", []),
    )


def _parse_defaults(data: dict) -> DelegateDefaults:
    """Parse defaults configuration from dict."""
    timeout_value = data.get("timeout", 120)
    if isinstance(timeout_value, str):
        timeout_value = int(resolve_env_vars(timeout_value))

    return DelegateDefaults(
        temperature=data.get("temperature", 0.7),
        max_tokens=data.get("max_tokens", 2048),
        timeout=timeout_value,
    )


def _parse_delegate(role: str, data: dict) -> DelegateConfig:
    """Parse a single delegate configuration from dict."""
    connection_data = data.get("connection", {})
    capabilities_data = data.get("capabilities", {})
    defaults_data = data.get("defaults", {})

    return DelegateConfig(
        role=role,
        display_name=data.get("display_name", role.title()),
        connection=_parse_connection(connection_data),
        capabilities=_parse_capabilities(capabilities_data),
        defaults=_parse_defaults(defaults_data),
        description=data.get("description", f"Delegate to {role} LLM."),
    )


def _substitute_env_vars_recursive(data: Any) -> Any:
    """
    Recursively substitute environment variables in a data structure.

    Args:
        data: Any data structure (dict, list, str, etc.)

    Returns:
        Data structure with env vars resolved
    """
    if isinstance(data, dict):
        return {k: _substitute_env_vars_recursive(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_substitute_env_vars_recursive(item) for item in data]
    elif isinstance(data, str):
        return resolve_env_vars(data)
    return data


def load_delegates_config(path: Optional[str] = None) -> DelegatesConfiguration:
    """
    Load delegate LLM configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file. If None, uses
              DELEGATES_CONFIG_PATH env var or the default path.

    Returns:
        DelegatesConfiguration with all delegates loaded

    Raises:
        FileNotFoundError: If the config file doesn't exist
        ValueError: If the config is invalid
    """
    # Determine config path
    if path is None:
        path = os.environ.get(
            "DELEGATES_CONFIG_PATH", str(DEFAULT_DELEGATES_CONFIG_PATH)
        )

    config_path = Path(path)

    if not config_path.exists():
        logger.warning(
            f"Delegates config not found at {config_path}, using empty config"
        )
        return DelegatesConfiguration(version="1.0", delegates={})

    logger.debug(f"Loading delegates config from {config_path}")

    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)

    if raw_config is None:
        return DelegatesConfiguration(version="1.0", delegates={})

    version = raw_config.get("version", "1.0")
    delegates_data = raw_config.get("delegates", {})

    delegates = {}
    for role, delegate_data in delegates_data.items():
        try:
            delegates[role] = _parse_delegate(role, delegate_data)
            logger.debug(f"Loaded delegate: {role} -> {delegates[role].tool_name}")
        except Exception as e:
            logger.error(f"Failed to parse delegate '{role}': {e}")
            raise ValueError(f"Invalid delegate configuration for '{role}': {e}") from e

    return DelegatesConfiguration(version=version, delegates=delegates)


def validate_delegates_config(config: DelegatesConfiguration) -> list[str]:
    """
    Validate a delegates configuration.

    Args:
        config: Configuration to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    for role, delegate in config.delegates.items():
        if not delegate.connection.base_url:
            errors.append(f"Delegate '{role}': missing connection.base_url")
        if not delegate.connection.model:
            errors.append(f"Delegate '{role}': missing connection.model")
        if delegate.capabilities.context_length <= 0:
            errors.append(f"Delegate '{role}': context_length must be positive")
        if delegate.capabilities.max_output_tokens <= 0:
            errors.append(f"Delegate '{role}': max_output_tokens must be positive")

    return errors


# =============================================================================
# Unified Configuration Loading
# =============================================================================


def _parse_orchestrator_config(data: dict) -> OrchestratorConfig:
    """Parse orchestrator configuration from dict."""
    return OrchestratorConfig(
        base_url=data.get("base_url", "http://localhost:8001/v1"),
        model=data.get("model", "nvidia/Nemotron-Orchestrator-8B"),
        temperature=float(data.get("temperature", 0.7)),
        max_steps=int(data.get("max_steps", 10)),
    )


def _parse_server_config(data: dict) -> ServerConfig:
    """Parse server configuration from dict."""
    reload_value = data.get("reload", False)
    if isinstance(reload_value, str):
        reload_value = reload_value.lower() == "true"

    return ServerConfig(
        host=data.get("host", "0.0.0.0"),
        port=int(data.get("port", 8000)),
        workers=int(data.get("workers", 1)),
        reload=reload_value,
    )


def _parse_python_executor_config(data: dict) -> PythonExecutorConfig:
    """Parse Python executor configuration from dict."""
    return PythonExecutorConfig(
        url=data.get("url", "http://pyexec.cluster:9999/"),
        timeout=int(data.get("timeout", 30)),
    )


def _parse_searxng_config(data: dict) -> SearxngConfig:
    """Parse SearXNG configuration from dict."""
    return SearxngConfig(
        url=data.get("url", "http://searxng.cluster:9999/search"),
        timeout=int(data.get("timeout", 30)),
    )


def _parse_tools_config(data: dict) -> ToolsConfig:
    """Parse tools configuration from dict."""
    python_executor_data = data.get("python_executor", {})
    searxng_data = data.get("searxng", {})

    return ToolsConfig(
        searxng=_parse_searxng_config(searxng_data),
        python_executor=_parse_python_executor_config(python_executor_data),
    )


def _parse_fast_path_config(data: dict) -> FastPathConfig:
    """Parse fast-path configuration from dict."""
    enabled_value = data.get("enabled", True)
    if isinstance(enabled_value, str):
        enabled_value = enabled_value.lower() == "true"

    return FastPathConfig(enabled=enabled_value)


def _parse_logging_config(data: dict) -> LoggingConfig:
    """Parse logging configuration from dict."""
    return LoggingConfig(
        level=data.get("level", "INFO"),
    )


def _parse_langfuse_config(data: dict) -> LangfuseConfig:
    """Parse Langfuse configuration from dict."""
    enabled_value = data.get("enabled", False)
    if isinstance(enabled_value, str):
        enabled_value = enabled_value.lower() == "true"

    debug_value = data.get("debug", False)
    if isinstance(debug_value, str):
        debug_value = debug_value.lower() == "true"

    return LangfuseConfig(
        enabled=enabled_value,
        public_key=data.get("public_key", ""),
        secret_key=data.get("secret_key", ""),
        host=data.get("host", "https://cloud.langfuse.com"),
        flush_at=int(data.get("flush_at", 10)),
        flush_interval=float(data.get("flush_interval", 1.0)),
        debug=debug_value,
        output_max_length=int(data.get("output_max_length", 0)),
    )


def _parse_dspy_config(data: dict) -> DSPyConfig:
    """Parse DSPy configuration from dict."""
    return DSPyConfig(
        optimized_prompts_path=data.get("optimized_prompts_path", ""),
    )


def _parse_delegates_from_unified(data: dict) -> dict[str, DelegateConfig]:
    """Parse delegates from unified config format."""
    delegates = {}
    for role, delegate_data in data.items():
        try:
            delegates[role] = _parse_delegate(role, delegate_data)
            logger.debug(f"Loaded delegate: {role} -> {delegates[role].tool_name}")
        except Exception as e:
            logger.error(f"Failed to parse delegate '{role}': {e}")
            raise ValueError(f"Invalid delegate configuration for '{role}': {e}") from e
    return delegates


def load_app_config(path: Optional[str] = None, reload: bool = False) -> AppConfig:
    """
    Load unified application configuration from a YAML file.

    Uses a singleton pattern - subsequent calls return the cached config
    unless reload=True is specified.

    Args:
        path: Path to the YAML configuration file. If None, uses
              CONFIG_PATH env var or the default path (config/config.yaml).
        reload: If True, force reload from disk instead of using cache.

    Returns:
        AppConfig with all configuration loaded

    Raises:
        FileNotFoundError: If the config file doesn't exist
        ValueError: If the config is invalid
    """
    global _app_config

    # Return cached config if available and not reloading
    if _app_config is not None and not reload:
        return _app_config

    # Determine config path
    if path is None:
        path = os.environ.get("CONFIG_PATH", str(DEFAULT_CONFIG_PATH))

    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found at {config_path}. "
            f"Create one from config/config.yaml.template or set CONFIG_PATH env var."
        )

    logger.info(f"Loading configuration from {config_path}")

    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)

    if raw_config is None:
        raise ValueError(f"Configuration file {config_path} is empty")

    # Substitute environment variables throughout the config
    raw_config = _substitute_env_vars_recursive(raw_config)

    # Parse each section
    version = raw_config.get("version", "1.0")

    orchestrator_data = raw_config.get("orchestrator", {})
    server_data = raw_config.get("server", {})
    tools_data = raw_config.get("tools", {})
    fast_path_data = raw_config.get("fast_path", {})
    logging_data = raw_config.get("logging", {})
    langfuse_data = raw_config.get("langfuse", {})
    dspy_data = raw_config.get("dspy", {})
    delegates_data = raw_config.get("delegates", {})

    # Build the config
    app_config = AppConfig(
        version=version,
        orchestrator=_parse_orchestrator_config(orchestrator_data),
        server=_parse_server_config(server_data),
        tools=_parse_tools_config(tools_data),
        fast_path=_parse_fast_path_config(fast_path_data),
        logging=_parse_logging_config(logging_data),
        langfuse=_parse_langfuse_config(langfuse_data),
        dspy=_parse_dspy_config(dspy_data),
        delegates=_parse_delegates_from_unified(delegates_data),
    )

    # Validate delegates
    delegates_config = DelegatesConfiguration(
        version=version,
        delegates=app_config.delegates,
    )
    errors = validate_delegates_config(delegates_config)
    if errors:
        for error in errors:
            logger.warning(f"Config validation warning: {error}")

    # Cache the config
    _app_config = app_config

    logger.debug(
        f"Configuration loaded: version={version}, delegates={list(app_config.delegates.keys())}"
    )

    return app_config


def get_delegates_from_app_config(
    app_config: Optional[AppConfig] = None,
) -> DelegatesConfiguration:
    """
    Get a DelegatesConfiguration from an AppConfig.

    This provides backward compatibility for code that expects DelegatesConfiguration.

    Args:
        app_config: AppConfig to extract delegates from. If None, loads from default.

    Returns:
        DelegatesConfiguration
    """
    if app_config is None:
        app_config = load_app_config()

    return DelegatesConfiguration(
        version=app_config.version,
        delegates=app_config.delegates,
    )


def reset_config_cache() -> None:
    """Reset the configuration cache, forcing a reload on next access."""
    global _app_config
    _app_config = None
    logger.debug("Configuration cache reset")
