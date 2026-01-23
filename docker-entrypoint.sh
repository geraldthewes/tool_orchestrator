#!/bin/bash
set -e

# Fetch config from Consul if CONSUL_HTTP_ADDR is set and config doesn't exist
if [ -n "$CONSUL_HTTP_ADDR" ] && [ ! -f /app/config/config.yaml ]; then
    echo "Fetching config from Consul: $CONSUL_HTTP_ADDR"
    if ! curl -sf "$CONSUL_HTTP_ADDR/v1/kv/config/tool-orchestrator/config.yaml?raw" -o /app/config/config.yaml; then
        echo "ERROR: Failed to fetch config from Consul" >&2
        exit 1
    fi
    echo "Config fetched successfully"
fi

exec "$@"
