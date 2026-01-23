# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tool Orchestrator is a Python framework for LLM tool orchestration using ReAct-style reasoning (Reason -> Action -> Observation -> Repeat). It enables an orchestrator LLM to coordinate tools (web search, Python execution, math calculations) and delegate tasks to specialized LLMs.

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

- **`src/config.py`**: Configuration management. Loads unified config from `config/config.yaml`.

- **`src/config_loader.py`**: YAML configuration loader. Supports `${VAR:-default}` syntax for environment variable interpolation.

### Delegate System

Delegate LLMs are configured in the `delegates` section of `config/config.yaml`. Each delegate:
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

All configuration is in a single YAML file: `config/config.yaml`

```bash
# Copy template and configure
cp config/config.yaml.template config/config.yaml
```

The config file supports environment variable interpolation with `${VAR:-default}` syntax.

**Key sections:**
- `orchestrator`: Main orchestrator LLM endpoint and model
- `server`: API server settings (host, port, workers)
- `tools`: SearXNG endpoint and Python executor settings
- `fast_path`: Fast-path routing for simple queries
- `logging`: Log level configuration
- `langfuse`: Observability settings
- `dspy`: DSPy optimization paths
- `delegates`: Delegate LLM definitions (reasoner, coder, fast)

**Note:** `config/config.yaml` is gitignored as it may contain secrets.

## Testing

Tests use mocked LLM responses to test orchestration logic without external services. Tool tests (math solver, Python executor) run locally without mocks.
