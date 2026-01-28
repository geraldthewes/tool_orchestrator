"""
Nemotron-native orchestration loop.

Replaces the DSPy ReAct loop with a custom loop matching NVIDIA's
ToolOrchestra reference architecture: stateless prompt reconstruction,
structured observation buffers, and OpenAI function-calling format.
"""

from .buffers import ObservationBuffers, TokenBudgets
from .tool_defs import build_tool_definitions, build_tools_prompt_block
from .loop import OrchestrationLoop, OrchestrationStep, OrchestrationResult

__all__ = [
    "ObservationBuffers",
    "TokenBudgets",
    "build_tool_definitions",
    "build_tools_prompt_block",
    "OrchestrationLoop",
    "OrchestrationStep",
    "OrchestrationResult",
]
