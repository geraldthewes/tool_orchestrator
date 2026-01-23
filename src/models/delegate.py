"""
Data models for LLM delegate configuration.

Defines the structure for configuring delegate LLMs that the orchestrator
can route tasks to based on their capabilities.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ConnectionType(Enum):
    """Supported LLM connection types."""

    OPENAI_COMPATIBLE = "openai_compatible"
    OLLAMA = "ollama"


@dataclass
class DelegateConnection:
    """Connection details for a delegate LLM."""

    type: ConnectionType
    base_url: str
    model: str
    api_key: Optional[str] = None


@dataclass
class DelegateCapabilities:
    """Capability metadata for a delegate LLM."""

    context_length: int
    max_output_tokens: int
    specializations: list[str] = field(default_factory=list)


@dataclass
class DelegateDefaults:
    """Default generation parameters for a delegate LLM."""

    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 120  # Request timeout in seconds


@dataclass
class DelegateConfig:
    """Complete configuration for a single delegate LLM."""

    role: str
    display_name: str
    connection: DelegateConnection
    capabilities: DelegateCapabilities
    defaults: DelegateDefaults
    description: str

    @property
    def tool_name(self) -> str:
        """Generate the tool name from the role."""
        return f"ask_{self.role}"


@dataclass
class DelegatesConfiguration:
    """Top-level configuration containing all delegate LLMs."""

    version: str
    delegates: dict[str, DelegateConfig] = field(default_factory=dict)
