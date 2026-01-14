# ToolOrchestra

A Python framework for testing LLM tool orchestration using ReAct-style reasoning.

## Overview

ToolOrchestra provides a framework for testing LLM orchestration capabilities:
- Orchestrate tool calls (web search, Python execution, math calculations)
- Delegate complex tasks to specialized LLMs (reasoning, coding, fast response)
- Implement ReAct-style reasoning (Reason -> Action -> Observation -> Repeat)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Query                                  │
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

### 3. Run the Interactive CLI

```bash
# Run interactive mode
python -m src.interactive

# Or run with verbose logging
python -m src.interactive -v

# Or run a single query
python -m src.interactive -q "What is the square root of 144?"
```

## Configuration

All endpoints are configurable via environment variables. See `.env.template` for details.

### Required

| Variable | Default | Description |
|----------|---------|-------------|
| `ORCHESTRATOR_BASE_URL` | `http://localhost:8001/v1` | Main orchestrator LLM endpoint |
| `ORCHESTRATOR_MODEL` | `nvidia/Nemotron-Orchestrator-8B` | Orchestrator model name |

### Optional (Delegate LLMs)

| Variable | Default | Description |
|----------|---------|-------------|
| `REASONING_LLM_BASE_URL` | `http://localhost:30000/v1` | Complex reasoning LLM |
| `REASONING_LLM_MODEL` | `openai/gpt-oss-120b` | Reasoning model name |
| `CODING_LLM_BASE_URL` | `http://localhost:8000/v1` | Code generation LLM |
| `CODING_LLM_MODEL` | `qwen3-coder` | Coding model name |
| `FAST_LLM_URL` | `http://localhost:11434/api/chat` | Fast reasoning LLM (Ollama) |
| `FAST_LLM_MODEL` | `nemotron-3-nano` | Fast model name |

### Tools

| Variable | Default | Description |
|----------|---------|-------------|
| `SEARXNG_ENDPOINT` | `http://localhost:8080/search` | SearXNG search endpoint |

### Example Configurations

#### Using Ollama Locally

```bash
ORCHESTRATOR_BASE_URL=http://localhost:11434/v1
ORCHESTRATOR_MODEL=llama3:8b
```

#### Using OpenAI

```bash
ORCHESTRATOR_BASE_URL=https://api.openai.com/v1
ORCHESTRATOR_MODEL=gpt-4
```

#### Using vLLM

```bash
ORCHESTRATOR_BASE_URL=http://localhost:8000/v1
ORCHESTRATOR_MODEL=nvidia/Nemotron-Orchestrator-8B
```

## Usage Examples

### Interactive Mode

```
>>> What is 2 + 2?
Processing query...
═══════════════════════════════════════════════════════════════════════
ANSWER
═══════════════════════════════════════════════════════════════════════
2 + 2 = 4
═══════════════════════════════════════════════════════════════════════
(Completed in 2 steps)
Use /trace to see the full reasoning trace.
```

### Complex Queries

```
>>> Search the web for the latest Python release and tell me its new features

>>> Write a Python function to check if a number is prime, then test it

>>> What are the implications of quantum computing on cryptography?
```

### CLI Commands

| Command | Description |
|---------|-------------|
| `/help` | Show help message |
| `/trace` | Show reasoning trace for last query |
| `/tools` | List available tools |
| `/verbose` | Toggle verbose mode |
| `/clear` | Clear conversation history |
| `/quit` | Exit the CLI |

## Programmatic Usage

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

## Project Structure

```
tool_orchestrator/
├── README.md                    # This file
├── Makefile                     # Quick commands
├── requirements.txt             # Python dependencies
├── .env.template                # Environment template
├── config/
│   └── tools.json               # Tool definitions
├── src/
│   ├── __init__.py
│   ├── config.py                # Configuration management
│   ├── llm_call.py              # LLM client
│   ├── orchestrator.py          # ReAct loop
│   ├── interactive.py           # CLI interface
│   └── tools/
│       ├── __init__.py
│       ├── search.py            # SearXNG integration
│       ├── python_executor.py   # Safe Python execution
│       ├── math_solver.py       # Math expressions
│       └── llm_delegate.py      # LLM delegation
└── tests/
    ├── test_tools.py
    └── test_orchestration.py
```

## Development

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

## Troubleshooting

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
