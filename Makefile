# ToolOrchestra Makefile

# Service Registration
SERVICE_NAME := tool-orchestrator
SERVICE_URL := http://tool-orchestrator.service.consul:9999
SERVICE_DESC := LLM tool orchestration framework using ReAct-style reasoning
SERVICE_SOURCE := https://github.com/geraldthewes/tool_orchestrator

.PHONY: help install setup test lint format clean interactive query check-endpoint server server-dev build deploy restart status unregister smoke-test

help:
	@echo "ToolOrchestra Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  make install       - Install Python dependencies"
	@echo "  make setup         - Full setup (install + copy env)"
	@echo ""
	@echo "Development:"
	@echo "  make test          - Run tests"
	@echo "  make lint          - Run linter (ruff)"
	@echo "  make format        - Format code (black)"
	@echo "  make clean         - Remove cache files"
	@echo ""
	@echo "Running:"
	@echo "  make interactive   - Start interactive CLI"
	@echo "  make query Q=\"...\" - Run a single query"
	@echo "  make check-endpoint- Test the orchestrator endpoint"
	@echo ""
	@echo "API Server:"
	@echo "  make server        - Start API server"
	@echo "  make server-dev    - Start API server with auto-reload"
	@echo ""
	@echo "Build and Deploy:"
	@echo "  make build         - Push code and build Docker image"
	@echo "  make deploy        - Deploy to Nomad cluster and restart to pull new image"
	@echo "  make restart       - Restart running allocations to pull new image"
	@echo "  make status        - Check deployment status"
	@echo "  make smoke-test    - Run post-deployment smoke tests"
	@echo "  make unregister    - Remove service from cluster registry"
	@echo ""

# Setup
install:
	pip install -r requirements.txt

setup: install
	@if [ ! -f .env ]; then cp .env.template .env; echo "Created .env from template"; fi

# Development
test:
	pytest tests/ -v

lint:
	ruff check src/ tests/

format:
	black src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true

# Running
interactive:
	python -m src.interactive

query:
	@if [ -z "$(Q)" ]; then echo "Usage: make query Q=\"your question here\""; exit 1; fi
	python -m src.interactive -q "$(Q)"

check-endpoint:
	@echo "Testing orchestrator endpoint..."
	@. ./.env 2>/dev/null || true; \
	ORCHESTRATOR_URL=$${ORCHESTRATOR_BASE_URL:-http://localhost:8001/v1}; \
	ORCHESTRATOR_MODEL=$${ORCHESTRATOR_MODEL:-nvidia/Nemotron-Orchestrator-8B}; \
	echo "Using endpoint: $$ORCHESTRATOR_URL"; \
	curl -s -X POST $$ORCHESTRATOR_URL/chat/completions \
		-H "Content-Type: application/json" \
		-d "{\"model\": \"$$ORCHESTRATOR_MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"Say hello\"}], \"max_tokens\": 50}" \
		| python -m json.tool || echo "Endpoint not responding"

# API Server
server:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000

server-dev:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Build and Deploy
build:
	git push origin
	jobforge submit-job --image-tags "latest" --watch --history deploy/build.yaml

deploy:
	nomad job run deploy/tool-orchestrator.nomad
	@echo "Restarting allocations to pull new image..."
	nomad job restart -on-error=fail tool-orchestrator
	@echo "Registering service with cluster registry..."
	register-service --name $(SERVICE_NAME) \
		--url "$(SERVICE_URL)" \
		--description "$(SERVICE_DESC)" \
		--type ondemand \
		--source "$(SERVICE_SOURCE)" \
		--endpoint api="/v1/chat/completions" \
		--endpoint health="/health"

unregister:
	register-service --unregister --name $(SERVICE_NAME)

restart:
	nomad job restart -on-error=fail tool-orchestrator

status:
	nomad job status tool-orchestrator

smoke-test:
	@echo "Running post-deployment smoke tests against $(SERVICE_URL)..."
	python scripts/post_deploy_tests/test_smoke.py $(SERVICE_URL)
