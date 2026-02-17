# MCP Version — Agentic Dataset Curation System

> **This codebase was converted from the OpenAI Agents SDK to the Model Context Protocol (MCP) by Claude (Anthropic AI), not by a human.**

## What Changed

The original system used the **OpenAI Agents SDK** (`agents` package) with:
- `Agent` class + `Runner.run_sync()` for the agentic loop
- `@function_tool` decorator to register Python functions as tools
- `ToolOutputImage` for inline image returns
- `set_default_openai_client()` to route through OpenRouter

This version uses the **Model Context Protocol (MCP)** instead:
- **`server.py`** — An MCP server (via `FastMCP`) that exposes all 9 curation tools over stdio transport (`init` + 8 curation tools)
- **`client.py`** — An MCP client that connects to the server, discovers tools, and runs an agentic LLM loop with Gemini 3 Flash via OpenRouter
- Tool definitions use MCP's JSON Schema format instead of `@function_tool`
- Images are returned as base64-encoded MCP content blocks instead of `ToolOutputImage`

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  client.py                      │
│  ┌───────────┐    ┌──────────────────────────┐  │
│  │ OpenRouter │◄──►│  Agentic Tool-Call Loop  │  │
│  │ (Gemini)  │    │  (max 100 turns)         │  │
│  └───────────┘    └──────────┬───────────────┘  │
│                              │ MCP (stdio)       │
└──────────────────────────────┼──────────────────┘
                               │
┌──────────────────────────────┼──────────────────┐
│                  server.py   │                   │
│              ┌───────────────▼────────────────┐  │
│              │         FastMCP Server         │  │
│              │  9 tools: init, search, sample,│  │
│              │  commit, undo_commit, status,  │  │
│              │  sample_from_committed,         │  │
│              │  aesthetics_rate, log_actions   │  │
│              └───────────────┬────────────────┘  │
│                              │                   │
│  ┌──────────────┐  ┌────────▼───────┐  ┌─────┐  │
│  │ Qwen3-VL-Emb │  │ HPSv3 Reward  │  │Weave│  │
│  │ (embeddings) │  │ (aesthetics)  │  │Trace│  │
│  └──────────────┘  └───────────────┘  └─────┘  │
└─────────────────────────────────────────────────┘
```

## Files

| File | Description |
|------|-------------|
| `server.py` | MCP server exposing 9 tools (`init` + 8 dataset curation tools) |
| `client.py` | Agent client — connects to server, runs LLM tool-call loop |
| `image_utils.py` | Image grid/stacking utilities (no SDK dependency) |
| `dataset_loader.py` | Loads embeddings from HuggingFace, initialises Qwen3-VL |
| `test_image_tool.py` | End-to-end test: calls `test_image` tool, sends image to LLM for description |
| `system_prompt.md` | Agent instructions (identical to original) |

## Usage

### Install dependencies

```bash
pip install mcp weave
```

### Run the agent

```bash
# From the mcp_version/ directory:
export WEAVE_PROJECT=your-project-name  # optional, default: aas2-mcp-client/server
python client.py
```

This will:
1. Spawn `server.py` as a subprocess (MCP stdio transport)
2. Connect and discover the 9 tools
3. Start the agentic loop with Gemini 3 Flash (model will call `init` first)
4. Trace LLM calls + MCP tool activity in Weave

> Note: all tools except `init` will return an error until initialization is done.

### Run just the server (for use with other MCP clients)

```bash
python server.py
```

The server communicates over stdio and can be connected to by any MCP-compatible client (e.g., Claude Desktop, custom clients, etc.).

### Test image transport (vision verification)

```bash
python test_image_tool.py
```

This connects to the MCP server, calls the `test_image` tool (which returns `123.jpg`), sends the image to kimi-k2.5 via OpenRouter, and prints the model's description. Verifies end-to-end that image content blocks survive the MCP stdio transport and are usable by vision models.

## Environment Variables

- `OPENROUTER_API_KEY` — Required for LLM calls (Gemini) and image captioning (Kimi-K2.5)
- `WEAVE_PROJECT` — Optional Weave project name (default: `aas2-mcp-client` in client, `aas2-mcp-server` in server)

## Key Differences from Original

| Aspect | Original (Agents SDK) | MCP Version |
|--------|----------------------|-------------|
| Tool registration | `@function_tool` decorator | `@mcp.tool()` decorator |
| Image returns | `ToolOutputImage` | Base64 MCP content blocks |
| Agent loop | `Runner.run_sync()` | Manual async tool-call loop |
| Transport | In-process function calls | MCP stdio (server subprocess) |
| Tool discovery | Hardcoded in `Agent()` constructor | Dynamic via `session.list_tools()` |
| Extensibility | Add tools to agent code | Any MCP client can connect to server |
