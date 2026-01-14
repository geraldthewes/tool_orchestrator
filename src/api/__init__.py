"""
FastAPI server module for ToolOrchestrator.

Provides OpenAI-compatible REST API endpoints.
"""

from .main import app, create_app

__all__ = ["app", "create_app"]
