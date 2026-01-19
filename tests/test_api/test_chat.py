"""Tests for OpenAI-compatible chat endpoints."""

from unittest.mock import patch, Mock

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


class TestModelsEndpoint:
    """Tests for /v1/models endpoint."""

    def test_list_models_returns_200(self):
        """Models list should return 200 OK."""
        response = client.get("/v1/models")
        assert response.status_code == 200

    def test_list_models_returns_tool_orchestrator(self):
        """Models list should include tool-orchestrator."""
        response = client.get("/v1/models")
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "tool-orchestrator"

    def test_list_models_format(self):
        """Models list should follow OpenAI format."""
        response = client.get("/v1/models")
        data = response.json()
        model = data["data"][0]
        assert model["object"] == "model"
        assert "created" in model
        assert "owned_by" in model

    def test_get_model_returns_200(self):
        """Get specific model should return 200 OK."""
        response = client.get("/v1/models/tool-orchestrator")
        assert response.status_code == 200

    def test_get_model_not_found(self):
        """Get unknown model should return 404."""
        response = client.get("/v1/models/unknown-model")
        assert response.status_code == 404


class TestChatCompletionsEndpoint:
    """Tests for /v1/chat/completions endpoint."""

    @patch("src.api.routes.chat.ToolOrchestrator")
    @patch("src.api.routes.chat.LLMClient")
    def test_chat_completion_success(self, mock_llm_client, mock_orchestrator_class):
        """Chat completion should return successful response."""
        mock_instance = Mock()
        mock_instance.run.return_value = "The answer is 42"
        mock_instance.steps = []  # Required for tracing metadata
        mock_orchestrator_class.return_value = mock_instance

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "tool-orchestrator",
                "messages": [{"role": "user", "content": "What is 6 * 7?"}],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert "42" in data["choices"][0]["message"]["content"]

    @patch("src.api.routes.chat.ToolOrchestrator")
    @patch("src.api.routes.chat.LLMClient")
    def test_chat_completion_format(self, mock_llm_client, mock_orchestrator_class):
        """Chat completion should follow OpenAI response format."""
        mock_instance = Mock()
        mock_instance.run.return_value = "Test response"
        mock_instance.steps = []  # Required for tracing metadata
        mock_orchestrator_class.return_value = mock_instance

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "tool-orchestrator",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        data = response.json()
        assert "id" in data
        assert data["id"].startswith("chatcmpl-")
        assert "created" in data
        assert data["model"] == "tool-orchestrator"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["finish_reason"] == "stop"
        assert "usage" in data

    def test_chat_completion_empty_messages(self):
        """Chat completion with empty messages should return 400."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "tool-orchestrator",
                "messages": [],
            },
        )
        assert response.status_code == 400

    def test_chat_completion_no_user_message(self):
        """Chat completion without user message should return 400."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "tool-orchestrator",
                "messages": [{"role": "system", "content": "You are helpful"}],
            },
        )
        assert response.status_code == 400

    @patch("src.api.routes.chat.ToolOrchestrator")
    @patch("src.api.routes.chat.LLMClient")
    def test_chat_completion_streaming(self, mock_llm_client, mock_orchestrator_class):
        """Chat completion with stream=true should return SSE response."""
        mock_instance = Mock()
        mock_instance.run.return_value = "Streamed response"
        mock_instance.steps = []  # Required for tracing metadata
        mock_orchestrator_class.return_value = mock_instance

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "tool-orchestrator",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

        # Parse SSE chunks
        content = response.text
        lines = [line for line in content.split("\n") if line.startswith("data: ")]

        # Should have content chunk, finish chunk, and [DONE]
        assert len(lines) == 3
        assert lines[2] == "data: [DONE]"

        # Verify first chunk contains the response
        import json
        chunk = json.loads(lines[0].replace("data: ", ""))
        assert chunk["object"] == "chat.completion.chunk"
        assert chunk["model"] == "tool-orchestrator"
        assert chunk["choices"][0]["delta"]["content"] == "Streamed response"

    @patch("src.api.routes.chat.ToolOrchestrator")
    @patch("src.api.routes.chat.LLMClient")
    def test_chat_completion_uses_last_user_message(
        self, mock_llm_client, mock_orchestrator_class
    ):
        """Chat completion should use the last user message."""
        mock_instance = Mock()
        mock_instance.run.return_value = "Response"
        mock_instance.steps = []  # Required for tracing metadata
        mock_orchestrator_class.return_value = mock_instance

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "tool-orchestrator",
                "messages": [
                    {"role": "user", "content": "First message"},
                    {"role": "assistant", "content": "First response"},
                    {"role": "user", "content": "Second message"},
                ],
            },
        )

        assert response.status_code == 200
        mock_instance.run.assert_called_once_with("Second message")

    @patch("src.api.routes.chat.ToolOrchestrator")
    @patch("src.api.routes.chat.LLMClient")
    def test_chat_completion_error_handling(
        self, mock_llm_client, mock_orchestrator_class
    ):
        """Chat completion should handle orchestrator errors."""
        mock_instance = Mock()
        mock_instance.run.side_effect = Exception("LLM connection failed")
        mock_instance.steps = []  # Required for tracing metadata
        mock_orchestrator_class.return_value = mock_instance

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "tool-orchestrator",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 500
