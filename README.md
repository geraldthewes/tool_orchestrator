# Tool Orchestrator

**Open-source LLM tool orchestration framework with OpenAI-compatible API**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OpenAI Compatible](https://img.shields.io/badge/OpenAI-Compatible-green.svg)](https://platform.openai.com/docs/api-reference)

A Python framework for LLM orchestration using ReAct-style reasoning (Reason -> Action -> Observation). Works as a drop-in replacement for OpenAI endpoints with built-in web search, Python execution, and multi-LLM delegation. Uses NVIDIA's Nemotron-native architecture with stateless prompt reconstruction and OpenAI function-calling.

## What is Tool Orchestrator?

Tool Orchestrator is an open-source LLM orchestration framework that provides an OpenAI-compatible REST API for AI agent workflows. It works as a self-hosted OpenAI alternative, enabling ReAct-style tool orchestration with any compatible client library, including OpenWebUI, LiteLLM, and the OpenAI Python SDK.

Unlike frameworks like LangChain or LlamaIndex that require custom integration code, Tool Orchestrator exposes a standard `/v1/chat/completions` endpoint—point any OpenAI client at it and it just works.

**Key Features:**
- **OpenAI-compatible API** - Point any OpenAI client at `/v1` and it just works
- **ReAct orchestration** - Reason -> Action -> Observation -> Repeat until final answer
- **Tool integration** - Web search (SearXNG), Python execution, math calculations
- **Delegate LLMs** - Route specialized tasks to reasoning, coding, or fast-response models
- **Streaming support** - Server-sent events for real-time responses
- **Observability** - Optional Langfuse tracing for request lifecycle monitoring

The service also includes an interactive CLI for development and testing.

## Why Tool Orchestrator?

### Comparison with Other LLM Frameworks

| Feature | Tool Orchestrator | LangChain | LlamaIndex | AutoGen |
|---------|-------------------|-----------|------------|---------|
| OpenAI-compatible API | **Yes** | No | No | No |
| Drop-in replacement | **Yes** | No | No | No |
| ReAct reasoning | **Yes** | Yes | Limited | Yes |
| Multi-LLM delegation | **Yes** | Yes | Yes | Yes |
| Built-in web search | **Yes** | Requires setup | Requires setup | No |

**Ideal for:**
- Teams wanting OpenAI-compatible endpoints without vendor lock-in
- Projects needing ReAct-style tool orchestration with minimal setup
- Developers who want to leverage multiple specialized LLMs

## Architecture: How LLM Tool Orchestration Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Client Request                              │
│          (curl, OpenAI SDK, OpenWebUI, any HTTP client)             │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Tool Orchestrator API Server                     │
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

## Quick Start: Set Up Your LLM Orchestrator

### 1. Install Dependencies

```bash
git clone <repo-url>
cd tool_orchestrator

# Install dependencies
pip install -r requirements.txt

# Or use make
make install
```

### 2. Configure

```bash
# Copy template and edit with your LLM endpoints
cp config/config.yaml.template config/config.yaml

# Edit config/config.yaml with your configuration
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
  "max_tokens": 8192,
  "stream": false,
  "include_trace": false
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | `tool-orchestrator` | Model ID (always `tool-orchestrator`) |
| `messages` | array | required | Conversation messages with `role` and `content` |
| `temperature` | float | `0.7` | Sampling temperature (0.0 - 2.0) |
| `max_tokens` | int | `8192` | Maximum tokens in response (1 - 8192) |
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

All configuration is managed through a single YAML file: `config/config.yaml`

### Quick Setup

```bash
# Copy template and edit with your endpoints
cp config/config.yaml.template config/config.yaml

# Edit config/config.yaml with your LLM endpoints and settings
```

### Configuration Sections

| Section | Purpose |
|---------|---------|
| `orchestrator` | Main orchestrator LLM endpoint and model |
| `server` | API server settings (host, port, workers) |
| `tools` | Tool endpoints (SearXNG, Python executor) |
| `fast_path` | Fast-path routing for simple queries |
| `logging` | Log level configuration |
| `langfuse` | Observability settings |
| `delegates` | Delegate LLM definitions |

**Note:** `config/config.yaml` is gitignored as it may contain secrets (API keys).

### Orchestrator Generation Parameters

The orchestrator supports additional generation parameters to prevent repetitive output (a common issue where the LLM gets stuck repeating the same JSON fragments):

```yaml
orchestrator:
  base_url: "http://localhost:8001/v1"
  model: "nvidia/Nemotron-Orchestrator-8B"
  temperature: 0.7
  max_steps: 10
  # Generation parameters to prevent repetitive output
  stop:
    - "<|im_end|>"
    - "<|endoftext|>"
  frequency_penalty: 0.3  # Discourage repeating tokens (0.0-2.0)
  presence_penalty: 0.1   # Discourage reusing tokens (0.0-2.0)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stop` | `[]` | Stop sequences to end generation. For Qwen3-derived models (like Nemotron-Orchestrator-8B), use `["<\|im_end\|>", "<\|endoftext\|>"]` |
| `frequency_penalty` | `0.0` | Penalty for repeating tokens. Higher values (0.3-0.5) discourage repetition. |
| `presence_penalty` | `0.0` | Penalty for using tokens that have appeared. Higher values encourage variety. |

These parameters are passed to the LLM inference endpoint (vLLM/SGLang/etc.) via the OpenAI-compatible API.

### Environment Variable Interpolation

The config file supports `${VAR:-default}` syntax for environment variables:

```yaml
orchestrator:
  base_url: "${ORCHESTRATOR_BASE_URL:-http://localhost:8001/v1}"
  model: "${ORCHESTRATOR_MODEL:-nvidia/Nemotron-Orchestrator-8B}"
```

### Delegate Connection Types

Delegates support two connection types:

- **`openai_compatible`** - For OpenAI-compatible APIs (vLLM, SGLang, Ollama `/v1`, etc.)
- **`ollama`** - For native Ollama API (`/api/chat`)

Example delegate configuration:
```yaml
delegates:
  fast:
    display_name: "Fast Responder"
    connection:
      type: "openai_compatible"  # or "ollama" for native Ollama API
      base_url: "${FAST_LLM_URL:-http://localhost:11434/v1}"
      model: "${FAST_LLM_MODEL:-llama3}"
    capabilities:
      context_length: 8192
      max_output_tokens: 2048
      specializations:
        - "quick_answers"
    defaults:
      temperature: 0.7
      timeout: 120
    description: "Fast responses for simple queries"
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
    api_key="not-needed",  # Tool Orchestrator doesn't require API keys
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

## Tools Reference: LLM Tool Calling

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
make security      # Run security scan (bandit)
make clean         # Remove cache files
make interactive   # Start interactive CLI
make query Q="..." # Run a single query
make check-endpoint # Test the orchestrator endpoint
make push-config   # Push config.yaml to Consul KV (required before first build)
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

## Project Structure

```
tool_orchestrator/
├── README.md                    # This file
├── Makefile                     # Quick commands
├── requirements.txt             # Python dependencies
├── config/
│   ├── config.yaml              # Main configuration (gitignored)
│   └── config.yaml.template     # Configuration template
├── data/
│   └── examples/                # Training examples
├── deploy/
│   ├── build.yaml               # JobForge build config
│   └── tool-orchestrator.nomad  # Nomad job specification
├── scripts/
│   ├── langfuse-setup.sh        # Langfuse/Vault setup script
│   └── analyze_langfuse_traces.py  # Trace analysis
├── src/
│   ├── __init__.py
│   ├── config.py                # Configuration management
│   ├── orchestrator.py          # Orchestrator entry point
│   ├── query_router/            # Fast-path routing
│   ├── interactive.py           # CLI interface
│   ├── orchestration/           # Nemotron-native orchestration
│   │   ├── loop.py              # Core loop (function-calling)
│   │   ├── buffers.py           # Observation buffers + token budgets
│   │   └── tool_defs.py         # OpenAI tool definitions
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
│   ├── prompts/                 # DSPy query routing
│   │   ├── modules/             # DSPy modules (router)
│   │   └── adapters/            # LM adapters for DSPy
│   └── tracing/                 # Langfuse observability
│       ├── __init__.py
│       ├── client.py            # Langfuse client wrapper
│       └── context.py           # Request-scoped tracing
└── tests/
    ├── test_tools.py
    ├── test_orchestration.py
    ├── test_orchestration/       # New orchestration tests
    │   ├── test_buffers.py
    │   ├── test_tool_defs.py
    │   └── test_loop.py
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

### Repetitive LLM output

If you see repeated JSON fragments in the orchestrator output (e.g., `"next_tool_name": "finish"` repeated many times), this indicates the LLM is stuck in a generation loop. Solutions:

1. **Add stop sequences** - Configure stop tokens for your model:
   ```yaml
   orchestrator:
     stop:
       - "<|im_end|>"
       - "<|endoftext|>"
   ```

2. **Increase frequency penalty** - Discourage token repetition:
   ```yaml
   orchestrator:
     frequency_penalty: 0.3
   ```

3. **Check logs** - The adapter logs warnings when repetition is detected:
   ```
   WARNING - Repetitive JSON pattern detected (N occurrences). Consider increasing frequency_penalty or adding stop sequences.
   ```

## Contributing

Contributions welcome! Please submit issues, feature requests, or pull requests.

If you find Tool Orchestrator useful, please consider giving it a star on GitHub.

[![GitHub stars](https://img.shields.io/github/stars/geraldthewes/tool_orchestrator?style=social)](https://github.com/geraldthewes/tool_orchestrator)

## References

- [Tool Orchestrator on GitHub](https://github.com/geraldthewes/tool_orchestrator)
- [ReAct: Synergizing Reasoning and Acting](https://react-lm.github.io/)
- [Nemotron-Orchestrator-8B on Hugging Face](https://huggingface.co/nvidia/Nemotron-Orchestrator-8B)
- [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat)
- [NVIDIA ToolOrchestra Reference](https://github.com/NVIDIA/ToolOrchestra)
