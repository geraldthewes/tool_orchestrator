# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tool Orchestrator is a Python framework for LLM tool orchestration using ReAct-style reasoning (Reason -> Action -> Observation -> Repeat). It enables an orchestrator LLM to coordinate tools (web search, Python execution, math calculations) and delegate tasks to specialized LLMs.

The orchestration loop uses NVIDIA's Nemotron-native architecture: stateless prompt reconstruction with structured observation buffers and OpenAI function-calling format.

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
make push-config  # Push config/config.yaml to Consul KV (run after config changes)
make build        # Push code and build Docker image
make deploy       # Deploy to Nomad cluster and restart to pull new image
make restart      # Restart running allocations to pull new image
make status       # Check deployment status
```

**Configuration updates:** Run `make push-config` after editing `config/config.yaml`. The container fetches config from Consul at startup, so restart is needed for config changes to take effect.

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

- **`src/orchestrator.py`**: Main orchestrator entry point. `ToolOrchestrator` class wraps the `OrchestrationLoop` and provides the public API (`run()`, `get_trace()`).

- **`src/orchestration/`**: Nemotron-native orchestration package:
  - `loop.py`: Core orchestration loop using OpenAI function-calling. Stateless prompt reconstruction with structured observation buffers.
  - `buffers.py`: Observation buffers with hierarchical token budgeting (attempts, code, delegates, documents).
  - `tool_defs.py`: Converts ToolRegistry and delegates to OpenAI function-calling format.

- **`src/config.py`**: Configuration management. Loads unified config from `config/config.yaml`.

- **`src/config_loader.py`**: YAML configuration loader. Loads config from `config/config.yaml` (fetched from Consul in production).

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

### Query Router (`src/prompts/modules/router.py`)

DSPy-based query routing for fast-path decisions. Uses a fast LLM to determine whether a query needs full orchestration or can be answered directly.

## Configuration

All configuration is managed through `config/config.yaml` and stored in Consul KV.

### Configuration Workflow

```bash
# 1. Create local config from template (first time only)
cp config/config.yaml.template config/config.yaml

# 2. Edit config/config.yaml with actual values (including secrets)

# 3. Push config to Consul KV (required before deploy)
make push-config

# 4. Deploy (container fetches config from Consul at startup)
make deploy
```

**IMPORTANT:**
- `config/config.yaml` is gitignored - it contains secrets (API keys, Langfuse credentials)
- All configuration values go directly in config.yaml - no environment variable interpolation in production
- Always run `make push-config` after editing config.yaml to update the deployed configuration
- The Docker container fetches config from Consul at startup via `CONSUL_HTTP_ADDR`

### Key Sections

- `orchestrator`: Main orchestrator LLM endpoint, model, and observation budgets
- `server`: API server settings (host, port, workers)
- `tools`: SearXNG endpoint and Python executor settings
- `fast_path`: Fast-path routing for simple queries
- `logging`: Log level configuration
- `langfuse`: Observability settings (including public_key, secret_key, host)
- `delegates`: Delegate LLM definitions (reasoner, coder, fast)

### Local Development

For local development, copy the template and edit with your endpoints:

```bash
cp config/config.yaml.template config/config.yaml
# Edit with your local/dev endpoints
python -m src.interactive  # Uses local config
```

## Testing

Tests use mocked OpenAI client responses to test orchestration logic without external services. Tool tests (math solver, Python executor) run locally without mocks.

## Observability with Langfuse

Langfuse integration provides tracing for LLM calls, tool executions, and orchestration steps.

### Analyzing Traces

Use `scripts/analyze_langfuse_traces.py` to diagnose performance and debug issues:

```bash
# Basic analysis of recent traces
python scripts/analyze_langfuse_traces.py --limit 50

# Verbose output with failure details
python scripts/analyze_langfuse_traces.py --limit 100 --verbose

# Analyze all traces
python scripts/analyze_langfuse_traces.py --all-traces --limit 50

# Filter by timestamp
python scripts/analyze_langfuse_traces.py --since 2024-01-15T00:00:00

# Export detailed report to JSON
python scripts/analyze_langfuse_traces.py --output report.json
```

**What the script analyzes:**
- Error rates for LLM generations and tool calls
- Tool usage distribution (which tools are called most)
- Latency statistics (avg, p50, p95, max) for generations and tools
- Token usage (prompt and completion tokens)
- Detailed failure extraction with input/output context

### Key Tracing Components

- **`src/tracing/client.py`**: Langfuse client wrapper with graceful degradation
- **`src/tracing/context.py`**: `TracingContext`, `SpanContext`, `GenerationContext` for structured traces

## Scripts

Utility scripts in `scripts/`:

| Script | Purpose |
|--------|---------|
| `analyze_langfuse_traces.py` | Analyze Langfuse traces for debugging and diagnostics |
| `generate_examples.py` | Generate training examples |
| `expand_examples.py` | Expand existing examples with variations |
| `validate_examples.py` | Validate example format and quality |
