# ToolOrchestra

An OpenAI-compatible API service for LLM tool orchestration using ReAct-style reasoning.

## Overview

ToolOrchestra provides an OpenAI-compatible REST API that orchestrates LLM tool calls and reasoning. It works as a drop-in replacement for OpenAI endpoints with any compatible client library, including OpenWebUI, LiteLLM, and the OpenAI Python SDK.

**Key Features:**
- **OpenAI-compatible API** - Point any OpenAI client at `/v1` and it just works
- **ReAct orchestration** - Reason -> Action -> Observation -> Repeat until final answer
- **Tool integration** - Web search (SearXNG), Python execution, math calculations
- **Delegate LLMs** - Route specialized tasks to reasoning, coding, or fast-response models
- **Streaming support** - Server-sent events for real-time responses
- **Observability** - Optional Langfuse tracing for request lifecycle monitoring

The service also includes an interactive CLI for development and testing.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Client Request                              │
│          (curl, OpenAI SDK, OpenWebUI, any HTTP client)             │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     ToolOrchestra API Server                        │
│                   (OpenAI-compatible /v1 endpoint)                  │
├─────────────────────────────────────────────────────────────────────┤
│                        Query Router                                 │
│           (fast-path vs orchestration routing)                      │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Orchestrator LLM                                 │
│              (Any OpenAI-compatible endpoint)                       │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
            ▼                   ▼                   ▼
┌───────────────────┐ ┌─────────────────┐ ┌─────────────────────────┐
│      Tools        │ │  Delegate LLMs  │ │       Search            │
├───────────────────┤ ├─────────────────┤ ├─────────────────────────┤
│ • python_execute  │ │ • Reasoning LLM │ │ • SearXNG               │
│ • calculate       │ │ • Coding LLM    │ │   (web search)          │
│                   │ │ • Fast LLM      │ │                         │
└───────────────────┘ └─────────────────┘ └─────────────────────────┘
```

## Prerequisites

- Python 3.10+
- Access to OpenAI-compatible LLM endpoint(s)
- Optional: SearXNG instance for web search
- Optional: Ollama for local LLM inference

## Quick Start

### 1. Install Dependencies

```bash
git clone <repo-url>
cd tool_orchestrator

# Install dependencies
pip install -r requirements.txt

# Or use make
make install
```

### 2. Configure Environment

```bash
# Copy template and edit with your LLM endpoints
cp .env.template .env

# Edit .env with your configuration
```

### 3. Start the API Server

```bash
# Production mode
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Development mode with auto-reload
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# With verbose logging
LOG_LEVEL=DEBUG uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Test the API

```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/v1/models

# Send a chat completion request
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tool-orchestrator",
    "messages": [{"role": "user", "content": "What is the square root of 144?"}]
  }'
```

## API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Create a chat completion (main endpoint) |
| `/v1/models` | GET | List available models |
| `/v1/models/{model_id}` | GET | Get model information |
| `/health` | GET | Health check |

Interactive API documentation is available at `/docs` (Swagger UI) and `/redoc` (ReDoc).

### Chat Completions

**Request Format:**

```json
{
  "model": "tool-orchestrator",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2 + 2?"}
  ],
  "temperature": 0.7,
  "max_tokens": 2048,
  "stream": false,
  "include_trace": false
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | `tool-orchestrator` | Model ID (always `tool-orchestrator`) |
| `messages` | array | required | Conversation messages with `role` and `content` |
| `temperature` | float | `0.7` | Sampling temperature (0.0 - 2.0) |
| `max_tokens` | int | `2048` | Maximum tokens in response (1 - 4096) |
| `stream` | bool | `false` | Enable streaming responses |
| `include_trace` | bool | `false` | Include orchestration trace in response |

**Response Format:**

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "tool-orchestrator",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "2 + 2 = 4"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 8,
    "total_tokens": 18
  }
}
```

### Streaming

Enable streaming by setting `stream: true`. Responses are returned as Server-Sent Events (SSE):

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tool-orchestrator",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

SSE format:
```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"},"index":0}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"delta":{},"index":0,"finish_reason":"stop"}]}

data: [DONE]
```

### Trace Inclusion

Request the full orchestration trace with `include_trace: true`:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tool-orchestrator",
    "messages": [{"role": "user", "content": "Calculate 15!"}],
    "include_trace": true
  }'
```

Response includes a `trace` array:
```json
{
  "choices": [...],
  "trace": [
    {
      "step": 1,
      "reasoning": "I need to calculate 15 factorial...",
      "action": "calculate",
      "action_input": "factorial(15)",
      "observation": "1307674368000",
      "is_final": false
    },
    {
      "step": 2,
      "reasoning": "I now have the answer.",
      "is_final": true
    }
  ]
}
```

### Error Handling

Errors follow the OpenAI error format:

```json
{
  "error": {
    "message": "No user message found in the request.",
    "type": "invalid_request_error",
    "code": null
  }
}
```

| HTTP Status | Description |
|-------------|-------------|
| 400 | Invalid request (validation error) |
| 404 | Model not found |
| 500 | Internal server error |

## Configuration

Configuration is managed through two files:

1. **`.env`** - Environment variables for endpoints, API keys, and runtime settings
2. **`config/delegates.yaml`** - Delegate LLM definitions (roles, capabilities, connection types)

### Quick Setup

```bash
# Copy template and edit with your endpoints
cp .env.template .env

# Edit .env with your LLM endpoints and settings
```

### Configuration Files

| File | Purpose |
|------|---------|
| `.env.template` | Template with all available environment variables and defaults |
| `config/delegates.yaml` | Delegate LLM configuration (supports `openai_compatible` and `ollama` connection types) |

### Delegate Connection Types

Delegates in `config/delegates.yaml` support two connection types:

- **`openai_compatible`** - For OpenAI-compatible APIs (vLLM, SGLang, Ollama `/v1`, etc.)
- **`ollama`** - For native Ollama API (`/api/chat`)

Example delegate configuration:
```yaml
delegates:
  fast:
    connection:
      type: "openai_compatible"  # or "ollama" for native Ollama API
      base_url: "${FAST_LLM_URL:-http://localhost:11434/v1}"
      model: "${FAST_LLM_MODEL:-llama3}"
```

## Integration Examples

### curl

```bash
# Simple request
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "tool-orchestrator", "messages": [{"role": "user", "content": "Hello!"}]}'

# With system message
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tool-orchestrator",
    "messages": [
      {"role": "system", "content": "You are a helpful math tutor."},
      {"role": "user", "content": "Explain the Pythagorean theorem"}
    ]
  }'

# Stream response
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "tool-orchestrator", "messages": [{"role": "user", "content": "Hello!"}], "stream": true}'
```

### Python (requests)

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "tool-orchestrator",
        "messages": [{"role": "user", "content": "What is 2 + 2?"}],
    },
)

result = response.json()
print(result["choices"][0]["message"]["content"])
```

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # ToolOrchestra doesn't require API keys
)

response = client.chat.completions.create(
    model="tool-orchestrator",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Search the web for the latest Python release"},
    ],
)

print(response.choices[0].message.content)
```

### OpenAI Python SDK (Streaming)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
)

stream = client.chat.completions.create(
    model="tool-orchestrator",
    messages=[{"role": "user", "content": "Write a haiku about programming"}],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
```

### Node.js / TypeScript

```typescript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:8000/v1",
  apiKey: "not-needed",
});

async function main() {
  const response = await client.chat.completions.create({
    model: "tool-orchestrator",
    messages: [{ role: "user", content: "What is the capital of France?" }],
  });

  console.log(response.choices[0].message.content);
}

main();
```

## Tools Reference

### Built-in Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `web_search` | Search via SearXNG | `query`, `categories`, `num_results` |
| `python_execute` | Safe Python sandbox | `code` |
| `calculate` | Math expressions | `expression` |

### Delegate LLMs

| Tool | Purpose | Use Case |
|------|---------|----------|
| `ask_gpt_oss` | Reasoning LLM | Complex reasoning, analysis |
| `ask_coder` | Coding LLM | Code generation, debugging |
| `ask_nemotron_nano` | Fast LLM | Quick responses |

## Development

### Interactive CLI

The project includes an interactive CLI for development and testing:

```bash
# Run interactive mode
python -m src.interactive

# Or run with verbose logging
python -m src.interactive -v

# Or run a single query
python -m src.interactive -q "What is the square root of 144?"
```

**CLI Commands:**

| Command | Description |
|---------|-------------|
| `/help` | Show help message |
| `/trace` | Show reasoning trace for last query |
| `/tools` | List available tools |
| `/verbose` | Toggle verbose mode |
| `/clear` | Clear conversation history |
| `/quit` | Exit the CLI |

### Programmatic Usage

```python
from src import ToolOrchestrator, run_query

# Simple usage
answer = run_query("What is the capital of France?")
print(answer)

# With full control
orchestrator = ToolOrchestrator(verbose=True)
result = orchestrator.run("Calculate 15! and explain why it's used")

# Get the reasoning trace
trace = orchestrator.get_trace()
for step in trace:
    print(f"Step {step['step']}: {step['reasoning']}")
```

### Make Commands

```bash
make install       # Install Python dependencies
make setup         # Full setup (install + copy env)
make test          # Run tests
make lint          # Run linter (ruff)
make format        # Format code (black)
make clean         # Remove cache files
make interactive   # Start interactive CLI
make query Q="..." # Run a single query
make check-endpoint # Test the orchestrator endpoint
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Test individual tools
pytest tests/test_tools.py -v

# Test orchestration
pytest tests/test_orchestration.py -v
```

### Prompt Optimization

ToolOrchestra uses [DSPy](https://dspy.ai/) for declarative prompt programming. The project includes 150+ training examples and supports automatic prompt optimization using GEPA (Genetic-Pareto optimization).

**Why GEPA?**

| Optimizer | Strengths | When to Use |
|-----------|-----------|-------------|
| **GEPA** | Outperforms MIPROv2 by 10%+, 35x fewer rollouts than GRPO, trajectory reflection | Best for ReAct-style programs, moderate datasets (50-200 examples) |
| MIPROv2 | Good for larger datasets, Bayesian optimization | Fallback if GEPA unavailable |
| BootstrapFewShot | Simple, fast, good starting point | Small datasets (~10 examples) |

**Running Optimization:**

```bash
# Dry run to verify examples load correctly
python scripts/optimize_prompts.py --dry-run

# Run GEPA optimization (default)
python scripts/optimize_prompts.py

# With specific options
python scripts/optimize_prompts.py --strategy gepa --gepa-auto medium --output-dir data/optimized_prompts

# Only optimize orchestrator module
python scripts/optimize_prompts.py --module orchestrator

# Use bootstrap for faster/simpler optimization
python scripts/optimize_prompts.py --strategy bootstrap
```

**CLI Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--strategy` | `gepa` | Optimization strategy (`gepa`, `mipro`, `bootstrap`) |
| `--output-dir` | `data/optimized_prompts` | Directory to save optimized modules |
| `--module` | `all` | Module to optimize (`all`, `orchestrator`, `router`) |
| `--gepa-auto` | `light` | GEPA preset (`light`, `medium`, `heavy`) |
| `--dev-ratio` | `0.8` | Ratio of examples for validation set |
| `--dry-run` | - | Show what would be done without running |
| `-v` | - | Enable verbose logging |

**Programmatic Usage:**

```python
from src.prompts.optimization import PromptOptimizer, load_all_training_examples
from src.prompts.modules import ToolOrchestratorModule

# Load training examples
examples = load_all_training_examples()

# Create optimizer with GEPA strategy
optimizer = PromptOptimizer(strategy="gepa", gepa_auto="light")

# Optimize orchestrator module
module = ToolOrchestratorModule()
optimized = optimizer.optimize_orchestrator(module, trainset=examples)

# Save optimized module
PromptOptimizer.save(optimized, "data/optimized_prompts/orchestrator.json")
```

## Project Structure

```
tool_orchestrator/
├── README.md                    # This file
├── Makefile                     # Quick commands
├── requirements.txt             # Python dependencies
├── .env.template                # Environment template
├── config/
│   └── delegates.yaml           # Delegate LLM definitions
├── data/
│   ├── examples/                # DSPy training examples (150+)
│   └── optimized_prompts/       # Optimized module outputs
├── deploy/
│   ├── build.yaml               # JobForge build config
│   └── tool-orchestrator.nomad  # Nomad job specification
├── scripts/
│   ├── langfuse-setup.sh        # Langfuse/Vault setup script
│   └── optimize_prompts.py      # DSPy prompt optimization CLI
├── src/
│   ├── __init__.py
│   ├── config.py                # Configuration management
│   ├── llm_call.py              # LLM client
│   ├── orchestrator.py          # ReAct loop
│   ├── query_router.py          # Fast-path routing
│   ├── interactive.py           # CLI interface
│   ├── api/                     # FastAPI server
│   │   ├── main.py
│   │   ├── schemas.py           # Pydantic models
│   │   └── routes/
│   │       ├── chat.py          # /v1/chat/completions
│   │       └── health.py        # /health
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── registry.py          # Tool registration
│   │   ├── search.py            # SearXNG integration
│   │   ├── python_executor.py   # Safe Python execution
│   │   ├── math_solver.py       # Math expressions
│   │   └── llm_delegate.py      # LLM delegation
│   ├── prompts/                 # DSPy prompt programming
│   │   ├── modules/             # DSPy modules (router, orchestrator)
│   │   ├── optimization/        # GEPA/MIPROv2/Bootstrap optimizers
│   │   └── adapters/            # LM adapters for DSPy
│   └── tracing/                 # Langfuse observability
│       ├── __init__.py
│       ├── client.py            # Langfuse client wrapper
│       └── context.py           # Request-scoped tracing
└── tests/
    ├── test_tools.py
    ├── test_orchestration.py
    └── test_tracing.py
```

## Troubleshooting

### API server not starting

1. Check if the port is already in use:
   ```bash
   lsof -i :8000
   ```

2. Verify dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

### Orchestrator not responding

1. Check if the endpoint is reachable:
   ```bash
   make check-endpoint
   ```

2. Verify the model is loaded:
   ```bash
   curl $ORCHESTRATOR_BASE_URL/models
   ```

### Delegate LLM errors

Check that each delegate LLM endpoint is accessible:

```bash
# Reasoning LLM
curl $REASONING_LLM_BASE_URL/models

# Coding LLM
curl $CODING_LLM_BASE_URL/models

# Fast LLM (Ollama)
curl $FAST_LLM_URL -d '{"model": "'"$FAST_LLM_MODEL"'", "messages": [{"role": "user", "content": "hi"}]}'
```

### SearXNG not returning results

Check SearXNG service:
```bash
curl "$SEARXNG_ENDPOINT?q=test&format=json"
```

## References

- [ReAct: Synergizing Reasoning and Acting](https://react-lm.github.io/)
- [Nemotron-Orchestrator-8B on Hugging Face](https://huggingface.co/nvidia/Nemotron-Orchestrator-8B)
- [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat)
