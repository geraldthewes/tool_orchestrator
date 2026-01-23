"""
OpenAI-compatible chat completion endpoints.

Implements /v1/chat/completions and /v1/models to enable compatibility
with OpenAI client libraries and tools like OpenWebUI.
"""

import json
import logging
import time
import uuid
from typing import Generator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatCompletionMessage,
    UsageInfo,
    ModelInfo,
    ModelListResponse,
    ErrorResponse,
    TraceStep,
)
from ...orchestrator import ToolOrchestrator
from ...llm_call import LLMClient
from ...config import config
from ...query_router import QueryRouter
from ...tracing import TracingContext, get_tracing_client

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


def _create_sse_chunk(
    content: str,
    model: str,
    completion_id: str,
    finish_reason: str | None = None,
) -> str:
    """Create a Server-Sent Events formatted chunk for streaming responses."""
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content} if content else {},
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(chunk)}\n\n"


def _generate_streaming_response(
    answer: str,
    model: str,
) -> Generator[str, None, None]:
    """Generate SSE chunks for a streaming response.

    This is a 'fake' streaming implementation that returns the complete
    response as a single chunk, satisfying clients that require streaming.
    """
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    # Send the complete response as a single content chunk
    yield _create_sse_chunk(answer, model, completion_id)

    # Send final chunk with finish_reason
    yield _create_sse_chunk("", model, completion_id, finish_reason="stop")

    # Send the done marker
    yield "data: [DONE]\n\n"


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
def create_chat_completion(
    request: ChatCompletionRequest,
) -> ChatCompletionResponse | StreamingResponse:
    """
    Process a chat completion request through the orchestrator.

    Extracts the last user message from the conversation and runs it through
    the ReAct orchestration loop to generate a response.
    """
    logger.debug(f"Received chat completion request: {request.model_dump_json()}")

    # Extract the last user message
    user_messages = [msg for msg in request.messages if msg.role == "user"]
    if not user_messages:
        logger.warning("No user message found in request")
        raise HTTPException(
            status_code=400,
            detail="No user message found in the request.",
        )

    # Extract text content (handles both string and list formats)
    query = user_messages[-1].get_text_content()
    execution_id = f"exec-{uuid.uuid4().hex[:8]}"
    logger.info(
        f"[{execution_id}] Processing chat completion request: {query[:100]}..."
    )
    logger.debug(f"[{execution_id}] Full query: {query}")

    # Initialize tracing context
    tracing_context = TracingContext(execution_id=execution_id)
    tracing_context.start_trace(
        name="chat_completion",
        query=query,
        metadata={"model": request.model, "stream": request.stream},
    )

    answer = None

    try:
        # Check if fast-path routing is enabled
        if config.fast_path.enabled:
            query_router = QueryRouter()

            # Trace the routing decision
            with tracing_context.span(
                name="query_router",
                input={"query": query},
            ) as router_span:
                routing = query_router.route(query)
                router_span.set_output(
                    {
                        "needs_orchestration": routing.needs_orchestration,
                        "reason": routing.reason,
                    }
                )

            if not routing.needs_orchestration:
                logger.info(f"[{execution_id}] Fast-path response: {routing.reason}")
                answer = routing.direct_response or ""

                # Return streaming response if requested
                if request.stream:
                    tracing_context.end_trace(output=answer, status="success")
                    _flush_tracing()
                    return StreamingResponse(
                        _generate_streaming_response(answer, MODEL_ID),
                        media_type="text/event-stream",
                    )

                # Estimate token counts
                prompt_tokens = sum(
                    len(msg.get_text_content().split()) * 2 for msg in request.messages
                )
                completion_tokens = len(answer.split()) * 2

                tracing_context.end_trace(
                    output=answer,
                    status="success",
                    metadata={"route": "fast_path"},
                )
                _flush_tracing()

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
                    trace=None,
                )

            logger.info(f"[{execution_id}] Routing to orchestrator: {routing.reason}")

        # Create orchestrator and run query
        # Enable verbose output when LOG_LEVEL is DEBUG
        verbose = config.log_level.upper() == "DEBUG"
        llm_client = LLMClient()
        orchestrator = ToolOrchestrator(
            llm_client=llm_client,
            verbose=verbose,
            execution_id=execution_id,
            tracing_context=tracing_context,
        )

        answer = orchestrator.run(query)
        logger.debug(f"[{execution_id}] Orchestrator response: {answer[:200]}...")

        # Return streaming response if requested
        if request.stream:
            logger.debug("Returning streaming response")
            tracing_context.end_trace(output=answer, status="success")
            _flush_tracing()
            return StreamingResponse(
                _generate_streaming_response(answer, MODEL_ID),
                media_type="text/event-stream",
            )

        # Estimate token counts (rough approximation)
        prompt_tokens = sum(
            len(msg.get_text_content().split()) * 2 for msg in request.messages
        )
        completion_tokens = len(answer.split()) * 2

        # Build trace if requested
        trace = None
        if request.include_trace:
            trace = [
                TraceStep(
                    step=step["step"],
                    reasoning=step.get("reasoning"),
                    action=step.get("action"),
                    action_input=step.get("action_input"),
                    observation=step.get("observation"),
                    is_final=step.get("is_final", False),
                )
                for step in orchestrator.get_trace()
            ]

        tracing_context.end_trace(
            output=answer,
            status="success",
            metadata={"route": "orchestrator", "steps": len(orchestrator.steps)},
        )
        _flush_tracing()

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
            trace=trace,
        )

    except Exception as e:
        logger.exception(f"[{execution_id}] Chat completion failed: {e}")
        tracing_context.end_trace(output=str(e), status="error")
        _flush_tracing()
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


def _flush_tracing() -> None:
    """Flush tracing client if available."""
    client = get_tracing_client()
    if client:
        client.flush()
