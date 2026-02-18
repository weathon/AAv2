"""
MCP client for the image-generation server.

Uses mcp.client.streamable_http + OpenAI SDK (OpenRouter or direct).
Images returned by tools are passed to the LLM as vision messages
and also saved to ./client_output/ for viewing.

Usage:
    uv run image_generation/client.py
    uv run image_generation/client.py "generate 3 anti-aesthetic city images"

Env vars (all optional):
    MCP_SERVER_URL       default http://127.0.0.1:8000/mcp
    LLM_BASE_URL         default https://openrouter.ai/api/v1
    OPENROUTER_API_KEY   for OpenRouter auth (preferred)
    OPENAI_API_KEY       fallback for OpenAI API
    LLM_MODEL            default qwen/qwen3.5-plus-02-15
    LLM_MAX_RETRIES      default 20
    MAX_TURNS            default 100
"""

import asyncio
import base64
import json
import os
import sys
from pathlib import Path

import dotenv
dotenv.load_dotenv()

from openai import AsyncOpenAI
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MCP_URL         = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000/mcp")
MODEL           = os.getenv("LLM_MODEL", "qwen/qwen3.5-plus-02-15")
LLM_BASE_URL    = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_API_KEY     = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "20"))
MAX_TURNS       = int(os.getenv("MAX_TURNS", "100"))
OUT_DIR         = Path(__file__).parent / "client_output"
SYSTEM_PROMPT   = (Path(__file__).parent / "system_prompt.md").read_text()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mcp_tools_to_openai(mcp_tools) -> list[dict]:
    return [{"type": "function", "function": {
        "name": t.name,
        "description": t.description or "",
        "parameters": t.inputSchema,
    }} for t in mcp_tools]


def parse_mcp_result(result) -> tuple[str, list[dict]]:
    """Split MCP result into (text_summary, openai_image_parts)."""
    texts, images = [], []
    for item in result.content:
        if item.type == "text":
            texts.append(item.text)
        elif item.type == "image":
            images.append({
                "type": "image_url",
                "image_url": {"url": f"data:{item.mimeType};base64,{item.data}"},
            })
    return "\n".join(texts) or "(done)", images


def _parse_tool_cost(text: str) -> float | None:
    """Extract 'Session total: $X.XXXX' from MCP tool result text."""
    marker = "Session total: $"
    idx = text.find(marker)
    if idx == -1:
        return None
    start = idx + len(marker)
    end = start
    while end < len(text) and (text[end].isdigit() or text[end] == "."):
        end += 1
    return float(text[start:end]) if end > start else None


def _extract_llm_cost(response) -> float:
    """Extract cost from an OpenRouter/OpenAI chat completion response."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0.0
    # OpenRouter returns cost in usage or usage.model_extra
    api_cost = getattr(usage, "cost", None)
    if api_cost is None and hasattr(usage, "model_extra"):
        api_cost = usage.model_extra.get("cost")
    return float(api_cost) if api_cost is not None else 0.0


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            text = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
            if text:
                parts.append(str(text))
        return " ".join(parts).strip()
    return str(content or "").strip()


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

async def run_agent(initial_prompt: str):
    llm = AsyncOpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    total_cost: float = 0.0          # client-side running total

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": initial_prompt},
    ]

    async with streamable_http_client(MCP_URL) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            tools_result = await session.list_tools()
            openai_tools = mcp_tools_to_openai(tools_result.tools)
            # Client-side finish tool — LLM calls this to end the session
            openai_tools.append({"type": "function", "function": {
                "name": "finish",
                "description": "Call this tool ONLY when all tasks are complete and the final batch has been committed. This ends the session.",
                "parameters": {"type": "object", "properties": {"summary": {"type": "string", "description": "Final summary of what was accomplished."}}, "required": ["summary"]},
            }})
            print(f"[INIT] Tools: {[t.name for t in tools_result.tools] + ['finish']}\n")

            for turn in range(MAX_TURNS):
                print(f"\n{'='*55} TURN {turn + 1} {'='*55}")

                # --- LLM call with retry + timeout ---
                response = None
                for attempt in range(1, LLM_MAX_RETRIES + 1):
                    try:
                        response = await asyncio.wait_for(
                            llm.chat.completions.create(
                                model=MODEL,
                                messages=messages,
                                tools=openai_tools,
                            ),
                            timeout=120,
                        )
                        msg = response.choices[0].message
                        if not msg.tool_calls and not _extract_text(msg.content):
                            raise RuntimeError("Empty LLM response")
                        break
                    except Exception as exc:
                        print(f"[WARN] LLM attempt {attempt}/{LLM_MAX_RETRIES}: {exc}")
                        if attempt == LLM_MAX_RETRIES:
                            raise RuntimeError(
                                f"LLM call failed after {LLM_MAX_RETRIES} attempts"
                            ) from exc
                        await asyncio.sleep(2 ** (attempt - 1))

                choice = response.choices[0]
                msg = choice.message
                messages.append(msg.model_dump())

                # --- Track agent LLM cost ---
                llm_cost = _extract_llm_cost(response)
                if llm_cost > 0:
                    total_cost += llm_cost
                    print(f"[COST] Agent LLM: ${llm_cost:.6f} | Running total: ${total_cost:.4f}")
                    try:
                        await session.call_tool("add_agent_cost", {"amount": llm_cost})
                    except Exception:
                        pass  # non-critical

                text = _extract_text(msg.content)
                if text:
                    print(f"[ASSISTANT] {text}")

                if not msg.tool_calls:
                    # Nudge the LLM to keep working — only stop if it
                    # explicitly calls the client-side `finish` tool.
                    messages.append({
                        "role": "user",
                        "content": "继续。调用工具继续工作。如果你已经完成了所有任务，请调用 finish 工具。",
                    })
                    continue

                # --- Execute tool calls ---
                finished = False
                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    try:
                        fn_args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        fn_args = {}

                    print(f"[TOOL] {fn_name}({json.dumps(fn_args, ensure_ascii=False)[:200]})")

                    # Client-side finish tool
                    if fn_name == "finish":
                        summary = fn_args.get("summary", "")
                        print(f"\n[DONE] Agent finished. Summary: {summary}")
                        print(f"[DONE] Total session cost: ${total_cost:.4f}")
                        finished = True
                        break

                    try:
                        mcp_result = await session.call_tool(fn_name, fn_args)
                        text_out, img_parts = parse_mcp_result(mcp_result)
                    except Exception as exc:
                        print(f"[TOOL ERROR] {exc}")
                        text_out, img_parts = f"Error: {exc}", []

                    print(f"[TOOL RESULT] {text_out[:300]}")

                    # Parse cost from tool result (format: "... Session total: $X.XXXX")
                    tool_cost = _parse_tool_cost(text_out)
                    if tool_cost is not None:
                        total_cost = tool_cost  # server total is authoritative
                        print(f"[COST] Server session total: ${total_cost:.4f}")

                    # Text result → tool role message
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": text_out,
                    })

                    # Images → separate user message (broader model compat)
                    if img_parts:
                        messages.append({
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"[Images from {fn_name}]"},
                                *img_parts,
                            ],
                        })
                        # Save images to disk for viewing
                        OUT_DIR.mkdir(exist_ok=True)
                        for i, part in enumerate(img_parts):
                            url = part["image_url"]["url"]
                            if "," in url:
                                header, b64 = url.split(",", 1)
                                ext = header.split("/")[1].split(";")[0]
                                out = OUT_DIR / f"{fn_name}_t{turn+1}_{tc.id[:6]}_{i}.{ext}"
                                out.write_bytes(base64.b64decode(b64))
                                print(f"[SAVED] {out}")
                if finished:
                    break
            else:
                print(f"\n[DONE] Reached max turns ({MAX_TURNS}). Total session cost: ${total_cost:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Prompt: ")
    asyncio.run(run_agent(prompt))
