# ToolOrchestrator API Server
# OpenAI-compatible REST API for LLM orchestration

FROM python:3.11-slim

# =============================================================================
# CONFIGURATION
# =============================================================================
# Configuration is loaded from config/config.yaml (or path in CONFIG_PATH env var).
# The config file supports ${VAR:-default} syntax for environment variable interpolation.
#
# To override the config file path:
#   CONFIG_PATH=/path/to/config.yaml
#
# Common environment variables that can be interpolated in config:
#   ORCHESTRATOR_BASE_URL, ORCHESTRATOR_MODEL
#   REASONING_LLM_BASE_URL, REASONING_LLM_MODEL
#   CODING_LLM_BASE_URL, CODING_LLM_MODEL
#   FAST_LLM_URL, FAST_LLM_MODEL
#
# See config/config.yaml.template for full configuration options.
# =============================================================================

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Install curl for Consul config fetch
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy entrypoint script
COPY docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

# Set default environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV SERVER_HOST=0.0.0.0
ENV SERVER_PORT=8000
ENV LOG_LEVEL=INFO

# Expose port
EXPOSE 8000

# Health check (using Python since curl is not in slim image)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Use entrypoint to fetch config before starting
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Run the server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
