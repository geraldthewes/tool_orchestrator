# ToolOrchestrator API Server
# OpenAI-compatible REST API for LLM orchestration

FROM python:3.11-slim

# =============================================================================
# ENVIRONMENT VARIABLES
# =============================================================================
# ORCHESTRATOR_BASE_URL   - URL for the orchestrator LLM endpoint
#                           Default: http://localhost:8001/v1
# ORCHESTRATOR_MODEL      - Model name for orchestrator
#                           Default: nvidia/Nemotron-Orchestrator-8B
# ORCHESTRATOR_TEMPERATURE- Temperature for orchestrator (0.0-2.0)
#                           Default: 0.7
# MAX_ORCHESTRATION_STEPS - Maximum ReAct loop iterations
#                           Default: 10
#
# DELEGATES_CONFIG_PATH   - Path to delegates.yaml configuration
#                           Default: config/delegates.yaml
#
# SEARXNG_ENDPOINT        - URL for SearXNG web search
#                           Default: http://localhost:8080/search
# PYTHON_EXECUTOR_TIMEOUT - Timeout for Python execution (seconds)
#                           Default: 30
#
# SERVER_HOST             - Server bind host
#                           Default: 0.0.0.0
# SERVER_PORT             - Server port
#                           Default: 8000
# SERVER_WORKERS          - Number of uvicorn workers
#                           Default: 1
#
# LOG_LEVEL               - Logging level (DEBUG, INFO, WARNING, ERROR)
#                           Default: INFO
# =============================================================================

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Set default environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV SERVER_HOST=0.0.0.0
ENV SERVER_PORT=8000
ENV LOG_LEVEL=INFO

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
