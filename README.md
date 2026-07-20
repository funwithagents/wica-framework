# WICA

WICA is an agentic framework for agents that take multimodal inputs and return multimodal outputs. Its state is centered on a single **World** object — a store of the current context that can be serialized to text (an LLM-facing prompt).

## Development

```
uv sync --dev
uv run pytest
uv run ruff check .
```
