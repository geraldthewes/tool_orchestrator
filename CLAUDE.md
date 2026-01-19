# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ToolOrchestra is a Python framework for LLM tool orchestration using ReAct-style reasoning (Reason -> Action -> Observation -> Repeat). It enables an orchestrator LLM to coordinate tools (web search, Python execution, math calculations) and delegate tasks to specialized LLMs.

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run a single test file
pytest tests/test_tools.py -v

# Run a specific test
pytest tests/test_orchestration.py::TestOrchestrator::test_initialization -v

# Lint and format
ruff check src/ tests/
black src/ tests/

# Run interactive CLI
python -m src.interactive
python -m src.interactive -v          # verbose mode
python -m src.interactive -q "query"  # single query

# Run API server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000  # dev mode
```

## Build and Deploy

Uses JobForge for CI/CD. Configuration is in `deploy/build.yaml`.

**IMPORTANT: NEVER run `docker build` directly. Always use JobForge via `make build` or `jobforge submit-job`.**

For full deployment guidance, see the [Cluster Deployment Guide](https://cluster-docs-https.service.consul:8444/services/DEPLOYMENT-GUIDE.md).

**Preferred: Use Makefile targets**

```bash
make build    # Push code and build Docker image
make deploy   # Deploy to Nomad cluster and restart to pull new image
make restart  # Restart running allocations to pull new image
make status   # Check deployment status
```

**Manual commands (if needed)**

```bash
# Build and push Docker image (requires changes pushed to main branch)
git push origin
jobforge submit-job --image-tags "latest" --watch --history deploy/build.yaml

# Deploy to Nomad cluster
nomad job run deploy/tool-orchestrator.nomad

# Restart allocations to pull new image (required when job spec unchanged)
nomad job restart -on-error=fail tool-orchestrator

# Check deployment status
nomad job status tool-orchestrator
```

## Architecture

### Core Components

- **`src/orchestrator.py`**: Main ReAct loop implementation. `ToolOrchestrator` class manages the reasoning cycle: calls the orchestrator LLM, parses responses for Thought/Action/Action Input, executes tools, and feeds observations back until a Final Answer is reached.

- **`src/llm_call.py`**: LLM client wrapper for OpenAI-compatible endpoints used by the orchestrator.

- **`src/config.py`**: Environment-based configuration using dataclasses. All settings come from environment variables with defaults.

- **`src/config_loader.py`**: YAML configuration loader for delegate LLMs. Supports `${VAR:-default}` syntax for environment variable interpolation.

### Delegate System

Delegate LLMs are configured in `config/delegates.yaml`. Each delegate:
- Has a role (e.g., `reasoner`, `coder`, `fast`)
- Becomes a tool named `ask_{role}` (e.g., `ask_reasoner`)
- Specifies connection details, capabilities, and defaults

The `DelegateConfig` model in `src/models/delegate.py` defines the structure. Connection types are `openai_compatible` or `ollama`.

### Tools (`src/tools/`)

- **`search.py`**: Web search via SearXNG
- **`python_executor.py`**: Sandboxed Python execution with allowlisted imports (math, json, re, collections, etc.)
- **`math_solver.py`**: Mathematical expression evaluation using safe eval
- **`llm_delegate.py`**: Routes prompts to configured delegate LLMs

### API Server (`src/api/`)

FastAPI server providing OpenAI-compatible REST endpoints:
- `src/api/main.py`: Application factory and lifespan management
- `src/api/routes/chat.py`: Chat completions endpoint
- `src/api/routes/health.py`: Health check endpoint

## Configuration

Copy `.env.template` to `.env` and configure:

**Required:**
- `ORCHESTRATOR_BASE_URL`: Main orchestrator LLM endpoint
- `ORCHESTRATOR_MODEL`: Orchestrator model name

**Delegate LLMs:** Configured via `config/delegates.yaml` with env var overrides:
- `REASONING_LLM_BASE_URL`, `REASONING_LLM_MODEL`
- `CODING_LLM_BASE_URL`, `CODING_LLM_MODEL`
- `FAST_LLM_URL`, `FAST_LLM_MODEL`

**Tools:**
- `SEARXNG_ENDPOINT`: SearXNG web search endpoint

## Testing

Tests use mocked LLM responses to test orchestration logic without external services. Tool tests (math solver, Python executor) run locally without mocks.
