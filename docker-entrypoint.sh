#!/bin/bash
set -e

CONFIG_FILE="/app/config/config.yaml"

# Fetch config from Consul KV
if [ -z "$CONSUL_HTTP_ADDR" ]; then
    echo "ERROR: CONSUL_HTTP_ADDR not set" >&2
    exit 1
fi

echo "Fetching config from Consul: $CONSUL_HTTP_ADDR"
if ! curl -sf "$CONSUL_HTTP_ADDR/v1/kv/config/tool-orchestrator/config.yaml?raw" -o "$CONFIG_FILE"; then
    echo "ERROR: Failed to fetch config from Consul" >&2
    echo "Make sure config is pushed via 'make push-config'" >&2
    exit 1
fi
echo "Config fetched successfully"

exec "$@"
