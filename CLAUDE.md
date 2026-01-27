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

## DSPy Checkpoint Deployment

After running DSPy optimization (`python scripts/optimize_prompts.py`), deploy the best checkpoint:

```bash
# Copy best checkpoint to deploy directory
mkdir -p deploy/checkpoints/orchestrator
cp data/checkpoints/orchestrator/checkpoint_*.json deploy/checkpoints/orchestrator/  # copy best
cp data/checkpoints/orchestrator/manifest.json deploy/checkpoints/orchestrator/
```

The service loads the best checkpoint at startup when `dspy.optimized_prompts_path` is configured:

```yaml
# In config/config.yaml
dspy:
  optimized_prompts_path: "deploy/checkpoints"
```

**Directory structure:**
- `data/checkpoints/` - Training artifacts (gitignored)
- `deploy/checkpoints/` - Deployment artifacts (committed)

**Key files:**
- `src/prompts/modules/orchestrator.py` - `_load_optimized_checkpoint()` loads at init
- `src/prompts/optimization/checkpoint.py` - `CheckpointManager` handles checkpoint I/O

## Testing

Tests use mocked LLM responses to test orchestration logic without external services. Tool tests (math solver, Python executor) run locally without mocks.

## Observability with Langfuse

Langfuse integration provides tracing for LLM calls, tool executions, and orchestration steps.

### Analyzing Traces

Use `scripts/analyze_langfuse_traces.py` to diagnose optimization performance and debug issues:

```bash
# Basic analysis of recent DSPy optimization traces
python scripts/analyze_langfuse_traces.py --limit 50

# Verbose output with failure details
python scripts/analyze_langfuse_traces.py --limit 100 --verbose

# Analyze all traces (not just dspy_optimization)
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

**Note:** Scores (e.g., `orchestration_quality_with_tools`) are computed locally by DSPy metric functions and NOT stored in Langfuse. The script analyzes execution quality proxies instead.

### Key Tracing Components

- **`src/tracing/client.py`**: Langfuse client wrapper with graceful degradation
- **`src/tracing/context.py`**: `TracingContext`, `SpanContext`, `GenerationContext` for structured traces
- **`src/prompts/adapters/lm_factory.py`**: `TracedLM` wrapper adds tracing to DSPy LM calls

## Scripts

Utility scripts in `scripts/`:

| Script | Purpose |
|--------|---------|
| `optimize_prompts.py` | Run DSPy optimization on orchestrator prompts |
| `analyze_langfuse_traces.py` | Analyze Langfuse traces for debugging and diagnostics |
| `generate_examples.py` | Generate training examples for DSPy optimization |
| `expand_examples.py` | Expand existing examples with variations |
| `validate_examples.py` | Validate example format and quality |
