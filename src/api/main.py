"""
FastAPI application for ToolOrchestrator.

Provides an OpenAI-compatible REST API that exposes the orchestration
engine to tools like OpenWebUI.

Usage:
    # Development server with auto-reload
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

    # Production server
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import health, chat

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown events."""
    logger.info("Starting ToolOrchestrator API server")
    yield
    logger.info("Shutting down ToolOrchestrator API server")


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

    return app


# Create the application instance
app = create_app()


def run_server():
    """
    Run the server using uvicorn.

    This is the entry point for running the server programmatically.
    """
    import uvicorn
    from ..config import config

    uvicorn.run(
        "src.api.main:app",
        host=config.server.host,
        port=config.server.port,
        reload=config.server.reload,
        workers=1 if config.server.reload else config.server.workers,
    )


if __name__ == "__main__":
    run_server()
