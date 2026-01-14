"""Health check endpoints."""

from fastapi import APIRouter

from ..schemas import HealthResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check if the API server is running and healthy.",
)
def health_check() -> HealthResponse:
    """Return health status of the API server."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        model="tool-orchestrator",
    )
