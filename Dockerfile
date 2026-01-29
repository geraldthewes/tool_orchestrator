# ToolOrchestrator API Server
# OpenAI-compatible REST API for LLM orchestration

FROM python:3.11-slim

# =============================================================================
# CONFIGURATION
# =============================================================================
# Configuration is fetched from Consul KV at container startup.
# Push config to Consul via `make push-config` from config/config.yaml.
#
# Required environment variable:
#   CONSUL_HTTP_ADDR - Consul HTTP address (e.g., http://consul.service.consul:8500)
#
# The container will fail to start if Consul is unavailable or config is missing.
# See config/config.yaml.template for configuration options.
# =============================================================================

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY deploy/checkpoints/ ./deploy/checkpoints/

# Install curl for Consul config fetch
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy entrypoint script
COPY docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

# Set Python environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check (using Python since curl is not in slim image)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Use entrypoint to fetch config before starting
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Run the server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
