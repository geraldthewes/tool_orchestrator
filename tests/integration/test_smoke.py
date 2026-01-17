#!/usr/bin/env python
"""
Smoke tests for verifying the container build is functional.

These tests run against the live container. The service URL can be passed
as a command-line argument or via SERVICE_HOST/SERVICE_PORT env vars.

Usage:
    python test_smoke.py http://host:port
    python test_smoke.py  # Uses SERVICE_HOST/SERVICE_PORT env vars
"""

import os
import sys
import urllib.request
import json


def get_service_url():
    """Get the service URL from file, command line args, or environment variables."""
    # Try reading from service_url.txt (written by JobForge before running python-executor)
    # python-executor places files in /work/ directory
    for url_file in ["/work/service_url.txt", "service_url.txt"]:
        if os.path.exists(url_file):
            with open(url_file) as f:
                return f.read().strip().rstrip("/")

    # Fall back to command line args
    if len(sys.argv) > 1:
        return sys.argv[1].rstrip("/")

    # Fall back to environment variables
    host = os.environ.get("SERVICE_HOST", "localhost")
    port = os.environ.get("SERVICE_PORT", "8000")
    return f"http://{host}:{port}"


SERVICE_URL = get_service_url()


def test_health_endpoint():
    """Verify the health endpoint returns healthy status."""
    url = f"{SERVICE_URL}/health"

    with urllib.request.urlopen(url, timeout=10) as response:
        assert response.status == 200
        data = json.loads(response.read().decode())
        assert data["status"] == "healthy"
        print(f"Health check passed: {data}")


def test_openapi_docs_available():
    """Verify the OpenAPI docs endpoint is accessible."""
    url = f"{SERVICE_URL}/openapi.json"

    with urllib.request.urlopen(url, timeout=10) as response:
        assert response.status == 200
        data = json.loads(response.read().decode())
        assert "openapi" in data
        assert "paths" in data
        print("OpenAPI docs available")


if __name__ == "__main__":
    print(f"Running smoke tests against {SERVICE_URL}")

    test_health_endpoint()
    test_openapi_docs_available()

    print("All smoke tests passed!")
