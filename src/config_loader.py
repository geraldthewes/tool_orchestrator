"""
Configuration loader for delegate LLMs.

Loads delegate configuration from YAML files with support for
environment variable interpolation.
"""

import logging
import os
import re
from pathlib import Path
from typing import Optional

import yaml

from .models import (
    ConnectionType,
    DelegateConnection,
    DelegateCapabilities,
    DelegateDefaults,
    DelegateConfig,
    DelegatesConfiguration,
)

logger = logging.getLogger(__name__)

# Default config path relative to project root
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "delegates.yaml"

# Regex for environment variable interpolation: ${VAR} or ${VAR:-default}
ENV_VAR_PATTERN = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")


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
        api_key=resolve_env_vars(data.get("api_key", "")) if data.get("api_key") else None,
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
    return DelegateDefaults(
        temperature=data.get("temperature", 0.7),
        max_tokens=data.get("max_tokens", 2048),
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
        path = os.environ.get("DELEGATES_CONFIG_PATH", str(DEFAULT_CONFIG_PATH))

    config_path = Path(path)

    if not config_path.exists():
        logger.warning(f"Delegates config not found at {config_path}, using empty config")
        return DelegatesConfiguration(version="1.0", delegates={})

    logger.info(f"Loading delegates config from {config_path}")

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
