"""
FastAPI application for ToolOrchestrator.

Provides an OpenAI-compatible REST API that exposes the orchestration
engine to tools like OpenWebUI.

Usage:
    # Development server with auto-reload
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

    # Production server
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4

    # Debug mode (verbose logging)
    LOG_LEVEL=DEBUG uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..config import config
from ..tools.registry import ToolRegistry
from ..tracing import init_tracing_client, shutdown_tracing
from .routes import health, chat


def configure_logging():
    """Configure logging based on LOG_LEVEL environment variable."""
    log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Set level for our modules
    logging.getLogger("src").setLevel(log_level)


configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown events."""
    logger.info("Starting ToolOrchestrator API server")

    # Log orchestrator configuration
    logger.info("=" * 60)
    logger.info("ORCHESTRATOR CONFIGURATION")
    logger.info(f"  Base URL: {config.orchestrator.base_url}")
    logger.info(f"  Model: {config.orchestrator.model}")
    logger.info(f"  Temperature: {config.orchestrator.temperature}")
    logger.info(f"  Max Steps: {config.orchestrator.max_steps}")

    # Log delegate LLMs
    logger.info("-" * 60)
    logger.info("DELEGATE LLMS")
    for role, delegate in config.delegates.items():
        logger.info(f"  [{delegate.tool_name}] {delegate.display_name}")
        logger.info(f"    URL: {delegate.connection.base_url}")
        logger.info(f"    Model: {delegate.connection.model}")
        logger.info(f"    Context: {delegate.capabilities.context_length:,} tokens")
        logger.info(f"    Type: {delegate.connection.type.value}")
        logger.info(f"    Timeout: {delegate.defaults.timeout}s")

    # Log tool endpoints
    logger.info("-" * 60)
    logger.info("TOOL ENDPOINTS")
    logger.info(f"  SearXNG: {config.tools.searxng_endpoint}")
    logger.info(f"  Python Executor: {config.tools.python_executor_url}")

    # Log registered tools
    logger.info("-" * 60)
    logger.info("REGISTERED TOOLS")
    for name, tool in ToolRegistry.all_tools().items():
        logger.info(f"  - {name}: {tool.description[:60]}...")

    # Log fast-path status
    logger.info("-" * 60)
    logger.info(f"FAST-PATH ROUTING: {'ENABLED' if config.fast_path.enabled else 'DISABLED'}")

    # Initialize Langfuse tracing
    logger.info("-" * 60)
    logger.info("LANGFUSE OBSERVABILITY")
    tracing_client = init_tracing_client(
        public_key=config.langfuse.public_key,
        secret_key=config.langfuse.secret_key,
        host=config.langfuse.host,
        debug=config.langfuse.debug,
    )
    if tracing_client.enabled:
        logger.info(f"  Status: ENABLED")
        logger.info(f"  Host: {config.langfuse.host or 'https://cloud.langfuse.com'}")
    else:
        logger.info(f"  Status: DISABLED")
        if tracing_client.error:
            logger.info(f"  Reason: {tracing_client.error}")

    logger.info("=" * 60)

    yield

    # Shutdown tracing
    logger.info("Shutting down ToolOrchestrator API server")
    shutdown_tracing()
    logger.info("Tracing client shutdown complete")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="ToolOrchestrator API",
        description=(
            "OpenAI-compatible REST API for LLM orchestration with tools and delegates. "
            "Use this server with OpenWebUI or any OpenAI-compatible client by pointing "
            "to the /v1 endpoint."
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS middleware - allow all origins for development
    # In production, restrict to specific origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(chat.router, tags=["Chat"])

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Log validation errors before returning 400 response."""
        logger.warning(
            f"Validation error on {request.method} {request.url.path}: {exc.errors()}"
        )
        # Log request body for debugging (truncated to avoid log spam)
        body = await request.body()
        logger.debug(f"Request body: {body.decode('utf-8', errors='replace')[:1000]}")
        return JSONResponse(
            status_code=400,
            content={"detail": exc.errors()},
        )

    return app


# Create the application instance
app = create_app()


def run_server():
    """
    Run the server using uvicorn.

    This is the entry point for running the server programmatically.
    """
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=config.server.host,
        port=config.server.port,
        reload=config.server.reload,
        workers=1 if config.server.reload else config.server.workers,
    )


if __name__ == "__main__":
    run_server()
