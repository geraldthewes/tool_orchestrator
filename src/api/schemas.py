"""
OpenAI-compatible Pydantic schemas for the API.

These schemas match the OpenAI Chat API format to enable compatibility
with tools like OpenWebUI, LiteLLM, and other OpenAI-compatible clients.
"""

import time
import uuid
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


class ContentPart(BaseModel):
    """A single part of multimodal content."""

    type: Literal["text", "image_url"] = Field(
        ..., description="The type of content part"
    )
    text: Optional[str] = Field(default=None, description="Text content (for type='text')")
    image_url: Optional[dict] = Field(
        default=None, description="Image URL object (for type='image_url')"
    )


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    role: Literal["system", "user", "assistant"] = Field(
        ..., description="The role of the message author"
    )
    content: Union[str, list[ContentPart]] = Field(
        ..., description="The content of the message (string or list of content parts)"
    )

    @field_validator("content", mode="before")
    @classmethod
    def normalize_content(cls, v):
        """Normalize content to handle both string and list formats."""
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            # Validate and convert list format
            return [
                ContentPart(**item) if isinstance(item, dict) else item for item in v
            ]
        return v

    def get_text_content(self) -> str:
        """Extract text content regardless of format."""
        if isinstance(self.content, str):
            return self.content
        # Extract text from list of content parts
        text_parts = []
        for part in self.content:
            if part.type == "text" and part.text:
                text_parts.append(part.text)
        return "\n".join(text_parts)


class ChatCompletionRequest(BaseModel):
    """Request body for /v1/chat/completions endpoint."""

    model: str = Field(
        default="tool-orchestrator",
        description="Model ID to use (always tool-orchestrator)",
    )
    messages: list[ChatMessage] = Field(
        ..., description="List of messages in the conversation", min_length=1
    )
    temperature: Optional[float] = Field(
        default=0.7, ge=0.0, le=2.0, description="Sampling temperature"
    )
    max_tokens: Optional[int] = Field(
        default=2048, ge=1, le=4096, description="Maximum tokens in response"
    )
    stream: Optional[bool] = Field(
        default=False, description="Whether to stream responses (not supported)"
    )
    include_trace: Optional[bool] = Field(
        default=False, description="Include orchestration trace in response"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "model": "tool-orchestrator",
                "messages": [{"role": "user", "content": "What is 2 + 2?"}],
                "temperature": 0.7,
            }
        }
    }


class TraceStep(BaseModel):
    """A single step in the orchestration trace."""

    step: int = Field(..., description="Step number in the orchestration")
    reasoning: Optional[str] = Field(default=None, description="The reasoning/thought for this step")
    action: Optional[str] = Field(default=None, description="Tool name that was invoked")
    action_input: Optional[str] = Field(default=None, description="Input passed to the tool")
    observation: Optional[str] = Field(default=None, description="Result from the tool")
    is_final: bool = Field(default=False, description="Whether this was the final step")


class ChatCompletionMessage(BaseModel):
    """Message in a chat completion response."""

    role: Literal["assistant"] = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    """A single choice in a chat completion response."""

    index: int = 0
    message: ChatCompletionMessage
    finish_reason: Literal["stop", "length", "error"] = "stop"


class UsageInfo(BaseModel):
    """Token usage information."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """Response body for /v1/chat/completions endpoint."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "tool-orchestrator"
    choices: list[ChatCompletionChoice]
    usage: UsageInfo = Field(default_factory=UsageInfo)
    trace: Optional[list[TraceStep]] = Field(
        default=None, description="Orchestration trace (when include_trace=True)"
    )


class ModelInfo(BaseModel):
    """Information about an available model."""

    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "tool-orchestrator"


class ModelListResponse(BaseModel):
    """Response body for /v1/models endpoint."""

    object: Literal["list"] = "list"
    data: list[ModelInfo]


class HealthResponse(BaseModel):
    """Response body for /health endpoint."""

    status: Literal["healthy", "unhealthy"]
    version: str
    model: str


class ErrorDetail(BaseModel):
    """Error detail in OpenAI format."""

    message: str
    type: str = "server_error"
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response in OpenAI format."""

    error: ErrorDetail
