# AgentTree

Recursive tree-based browser automation framework with adaptive redundancy and majority voting.

## What this is

A local-first Python framework that takes a natural language task, recursively decomposes it into a dynamic tree of browser automation subtasks, executes leaf nodes in parallel across Chrome tabs using a forked Nanobrowser Chrome extension, applies adaptive redundancy with majority voting for quality, and aggregates results upward through the tree into a final structured output.

## Quick Start

```bash
# Install dependencies
uv sync

# Set up environment
cp .env.example .env
# Edit .env with your OpenRouter API key

# Run with mock browser (no Chrome extension needed)
uv run agenttree --mock "Compare the top 5 Python web frameworks for building APIs"

# Run with real browser (requires Chrome extension)
uv run agenttree "Find 50 apartments in SF near Caltrain under $3k"
```

## Architecture

```
User prompt
    |
    v
Custom Asyncio Tree Engine
    Root Node (LLM: branch or execute?)
        Child A (branch) -> Grandchild A1, A2, A3 (execute -> browser)
        Child B (execute -> browser)
        Child C (branch) -> more children...
    Tab Pool (Sem 15) | LLM Client | SQLite Store
    |
    WebSocket (localhost:9223)
    |
Nanobrowser Fork (Chrome Extension, Manifest V3)
```

## Key Features

- **Dynamic tree depth** -- LLM decides at each node whether to decompose or execute
- **Parallel execution** -- up to 15 Chrome tabs via asyncio.TaskGroup
- **Adaptive redundancy** -- run once, evaluate confidence, escalate if needed
- **Majority voting** -- DSPy majority + LLM pick-best for quality
- **Crash recovery** -- SQLite checkpointing at every state transition
- **Real Chrome sessions** -- uses your cookies, logins, bookmarks
- **CAPTCHA detection** -- alerts you when CAPTCHAs are detected

## Configuration

Edit `config.yaml` or use environment variables:

```yaml
tree:
  max_depth: 5
  max_breadth: 10

llm:
  model: "llama-3.3-70b-versatile"

costs:
  budget_usd: 10.00

resources:
  max_parallel_tabs: 15
```

## Chrome Extension Setup

1. `cd extension && pnpm install && pnpm build`
2. Open `chrome://extensions/` -> Developer mode -> Load unpacked -> select `extension/dist/`
3. The extension will auto-connect to AgentTree's WebSocket server on port 9223

## Resume a Run

```bash
uv run agenttree --resume <run_id>
```
