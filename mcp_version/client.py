"""
MCP Client / Agent loop for the dataset curation system.

Connects to the MCP tool server (server.py) via stdio transport,
discovers available tools, and runs an agentic LLM loop using
Gemini 3 Flash (via OpenRouter) that autonomously calls MCP tools.

Replaces the OpenAI Agents SDK Runner.run_sync() approach with a
manual tool-calling loop over the MCP protocol.
"""

import os
import sys
import json
import asyncio
import base64

import dotenv
dotenv.load_dotenv()

import wandb
from openai import AsyncOpenAI

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "system_prompt.md")
SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "server.py")
MAX_TURNS = 100
MODEL = "google/gemini-3-flash-preview"
INITIAL_PROMPT = "Psychedelic art"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mcp_tools_to_openai_tools(mcp_tools) -> list[dict]:
    """Convert MCP tool definitions to OpenAI function-calling format."""
    openai_tools = []
    for tool in mcp_tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema,
            },
        })
    return openai_tools


def parse_mcp_result(result) -> list[dict]:
    """Convert an MCP tool result into OpenAI-compatible message content parts."""
    parts = []
    for content_block in result.content:
        if content_block.type == "text":
            parts.append({"type": "text", "text": content_block.text})
        elif content_block.type == "image":
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{content_block.mimeType};base64,{content_block.data}"},
            })
    if not parts:
        parts.append({"type": "text", "text": "(empty result)"})
    return parts


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

async def run_agent():
    # Load system prompt
    with open(SYSTEM_PROMPT_PATH, "r") as f:
        system_prompt = f.read()

    # LLM client (OpenRouter)
    llm = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    # Connect to MCP server
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[SERVER_SCRIPT],
        env={**os.environ},
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Discover tools
            tools_result = await session.list_tools()
            openai_tools = mcp_tools_to_openai_tools(tools_result.tools)
            tool_names = [t.name for t in tools_result.tools]
            print(f"[INIT] Connected to MCP server. Available tools: {tool_names}")

            # Conversation history
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": INITIAL_PROMPT},
            ]

            # Agentic loop
            for turn in range(MAX_TURNS):
                print(f"\n{'='*60}")
                print(f"[TURN {turn + 1}/{MAX_TURNS}]")
                print(f"{'='*60}")

                # Call LLM
                response = await llm.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=openai_tools if openai_tools else None,
                )

                choice = response.choices[0]
                assistant_message = choice.message

                # Append assistant response to history
                messages.append(assistant_message.model_dump())

                # If no tool calls, agent is done (or just responding)
                if not assistant_message.tool_calls:
                    print(f"[ASSISTANT] {assistant_message.content}")
                    if choice.finish_reason == "stop":
                        print("\n[DONE] Agent finished.")
                        break
                    continue

                # Process tool calls
                for tool_call in assistant_message.tool_calls:
                    fn_name = tool_call.function.name
                    fn_args_str = tool_call.function.arguments
                    try:
                        fn_args = json.loads(fn_args_str)
                    except json.JSONDecodeError:
                        fn_args = {}

                    print(f"[TOOL CALL] {fn_name}({json.dumps(fn_args, ensure_ascii=False)[:200]})")

                    # Call MCP server
                    try:
                        mcp_result = await session.call_tool(fn_name, fn_args)
                        content_parts = parse_mcp_result(mcp_result)
                    except Exception as e:
                        print(f"[TOOL ERROR] {fn_name}: {e}")
                        content_parts = [{"type": "text", "text": f"Error: {e}"}]

                    # Log text parts
                    for part in content_parts:
                        if part.get("type") == "text":
                            text_preview = part["text"][:300]
                            print(f"[TOOL RESULT] {text_preview}")

                    # Append tool result to conversation
                    # For models that don't support image_url in tool results,
                    # we include text parts only in the tool message and add
                    # image parts as a follow-up user message.
                    text_content = " ".join(
                        p["text"] for p in content_parts if p.get("type") == "text"
                    )
                    image_parts = [p for p in content_parts if p.get("type") == "image_url"]

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": text_content or "(done)",
                    })

                    # If there are images, inject them as a user message so the
                    # model can see them (Gemini supports vision in user turns).
                    if image_parts:
                        messages.append({
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"[Image result from {fn_name}]"},
                                *image_parts,
                            ],
                        })

            else:
                print(f"\n[DONE] Reached max turns ({MAX_TURNS}).")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    wandb.init(project="aas2", name="Psychedelic art (MCP)")
    try:
        asyncio.run(run_agent())
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
