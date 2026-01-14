"""
OpenAI-compatible chat completion endpoints.

Implements /v1/chat/completions and /v1/models to enable compatibility
with OpenAI client libraries and tools like OpenWebUI.
"""

import logging
import time

from fastapi import APIRouter, HTTPException

from ..schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatCompletionMessage,
    UsageInfo,
    ModelInfo,
    ModelListResponse,
    ErrorResponse,
    ErrorDetail,
)
from ...orchestrator import ToolOrchestrator
from ...llm_call import LLMClient

logger = logging.getLogger(__name__)

router = APIRouter()

# Model information
MODEL_ID = "tool-orchestrator"
MODEL_CREATED = int(time.time())


@router.get(
    "/v1/models",
    response_model=ModelListResponse,
    summary="List models",
    description="List available models. Returns the tool-orchestrator as the available model.",
)
def list_models() -> ModelListResponse:
    """Return list of available models (just tool-orchestrator)."""
    return ModelListResponse(
        data=[
            ModelInfo(
                id=MODEL_ID,
                created=MODEL_CREATED,
                owned_by="tool-orchestrator",
            )
        ]
    )


@router.get(
    "/v1/models/{model_id}",
    response_model=ModelInfo,
    summary="Get model",
    description="Get information about a specific model.",
)
def get_model(model_id: str) -> ModelInfo:
    """Return model information."""
    if model_id != MODEL_ID:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found. Available model: {MODEL_ID}",
        )
    return ModelInfo(
        id=MODEL_ID,
        created=MODEL_CREATED,
        owned_by="tool-orchestrator",
    )


@router.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Create chat completion",
    description=(
        "Create a chat completion using the ToolOrchestrator. "
        "The orchestrator processes the user message through a ReAct loop, "
        "using available tools and delegate LLMs to generate a response."
    ),
)
def create_chat_completion(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """
    Process a chat completion request through the orchestrator.

    Extracts the last user message from the conversation and runs it through
    the ReAct orchestration loop to generate a response.
    """
    # Check for streaming (not supported)
    if request.stream:
        raise HTTPException(
            status_code=400,
            detail="Streaming is not supported. Set stream=false.",
        )

    # Extract the last user message
    user_messages = [msg for msg in request.messages if msg.role == "user"]
    if not user_messages:
        raise HTTPException(
            status_code=400,
            detail="No user message found in the request.",
        )

    query = user_messages[-1].content
    logger.info(f"Processing chat completion request: {query[:100]}...")

    try:
        # Create orchestrator and run query
        llm_client = LLMClient()
        orchestrator = ToolOrchestrator(
            llm_client=llm_client,
            verbose=False,
        )

        answer = orchestrator.run(query)

        # Estimate token counts (rough approximation)
        prompt_tokens = sum(len(msg.content.split()) * 2 for msg in request.messages)
        completion_tokens = len(answer.split()) * 2

        return ChatCompletionResponse(
            model=MODEL_ID,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(content=answer),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    except Exception as e:
        logger.exception(f"Chat completion failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )
