"""Tests for health check endpoints."""

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_200(self):
        """Health check should return 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_healthy_status(self):
        """Health check should return healthy status."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_includes_version(self):
        """Health check should include version."""
        response = client.get("/health")
        data = response.json()
        assert "version" in data
        assert data["version"] == "0.1.0"

    def test_health_includes_model(self):
        """Health check should include model name."""
        response = client.get("/health")
        data = response.json()
        assert "model" in data
        assert data["model"] == "tool-orchestrator"
