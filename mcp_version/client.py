"""
MCP Client / Agent loop for the dataset curation system.

Connects to the MCP tool server (server.py) via SSE transport,
discovers available tools, and runs an agentic LLM loop using
Gemini 3 Flash (via OpenRouter) that autonomously calls MCP tools.

Replaces the OpenAI Agents SDK Runner.run_sync() approach with a
manual tool-calling loop over the MCP protocol.
"""

import os
import json
import asyncio
import base64
import datetime
import io

import dotenv
dotenv.load_dotenv()

from PIL import Image

from openai import AsyncOpenAI
import weave

from mcp import ClientSession
from mcp.client.sse import sse_client

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "system_prompt.md")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8765/sse")
MAX_TURNS = 100
MODEL = "google/gemini-3-flash-preview"
INITIAL_PROMPT = "Psychedelic art"
WEAVE_PROJECT = os.getenv("WEAVE_PROJECT", "aas2-mcp-client")
_WEAVE_ENABLED = False
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "4"))
SAMPLE_LOG_DIR = os.path.join(os.path.dirname(__file__), "sample_logs")

# Track original images for re-compression
_original_images = {}  # message_index -> [(content_index, original_data_url), ...]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@weave.op()
def _weave_event(event_type: str, payload: dict) -> dict:
    return {"event_type": event_type, **payload}


def _init_weave() -> None:
    global _WEAVE_ENABLED
    try:
        weave.init(WEAVE_PROJECT)
        _WEAVE_ENABLED = True
        print(f"[INIT] weave enabled for project '{WEAVE_PROJECT}'.")
    except Exception as e:
        raise RuntimeError(f"weave.init failed for project '{WEAVE_PROJECT}': {e}") from e


def _trace(event_type: str, **payload) -> None:
    if not _WEAVE_ENABLED:
        return
    blocked_keys = {
        "args",
        "raw_args",
        "initial_prompt",
        "content_preview",
        "tool_names",
        "messages",
    }
    sanitized = {}
    for k, v in payload.items():
        if k in blocked_keys:
            continue
        if isinstance(v, (list, tuple, dict)):
            continue
        if isinstance(v, str) and len(v) > 120:
            sanitized[k] = v[:120] + "...(truncated)"
        else:
            sanitized[k] = v
    try:
        _weave_event(event_type=event_type, payload=sanitized)
    except Exception as e:
        print(f"[WARN] weave trace failed for {event_type}: {e}")


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


def _extract_assistant_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and item.get("text"):
                    chunks.append(str(item["text"]))
            else:
                text_value = getattr(item, "text", None)
                if text_value:
                    chunks.append(str(text_value))
        return " ".join(chunks).strip()
    return str(content).strip()


def _save_data_url_image(data_url: str, output_path: str) -> None:
    if "," not in data_url:
        raise ValueError("Invalid data URL for image logging.")
    _, b64_data = data_url.split(",", 1)
    image_bytes = base64.b64decode(b64_data)
    with open(output_path, "wb") as f:
        f.write(image_bytes)


def _compress_image_for_llm(data_url: str, max_width: int = 1536, quality: int = 80) -> str:
    """Compress image for LLM message (in-memory only, for payload optimization).

    Resizes to max_width and converts to WebP (or JPEG fallback) with quality setting.
    WebP typically achieves 25-35% better compression than JPEG.
    Returns compressed base64 data URL (~5-15x smaller than original PNG).

    Args:
        data_url: Base64 data URL of image
        max_width: Maximum width/height for thumbnail (default 1536px)
        quality: Quality setting for compression (default 80, valid 0-100)
    """
    if "," not in data_url:
        return data_url  # Return original if invalid
    try:
        _, b64_data = data_url.split(",", 1)
        image_bytes = base64.b64decode(b64_data)
        img = Image.open(io.BytesIO(image_bytes))

        # Resize if larger than max_width
        if img.width > max_width or img.height > max_width:
            img.thumbnail((max_width, max_width), Image.Resampling.LANCZOS)

        # Try WebP first, fall back to JPEG if needed
        output = io.BytesIO()
        try:
            # WebP provides ~25-35% better compression than JPEG
            img.save(output, format="WEBP", quality=quality, optimize=True)
            compressed_b64 = base64.b64encode(output.getvalue()).decode()
            return f"data:image/webp;base64,{compressed_b64}"
        except Exception as webp_error:
            print(f"[WARN] WebP compression failed: {webp_error}, falling back to JPEG")
            output = io.BytesIO()
            img.save(output, format="JPEG", quality=quality, optimize=True)
            compressed_b64 = base64.b64encode(output.getvalue()).decode()
            return f"data:image/jpeg;base64,{compressed_b64}"
    except Exception as e:
        print(f"[WARN] Image compression failed: {e}, using original")
        return data_url


def _recompress_old_images(messages: list, recent_count: int = 10) -> None:
    """Re-compress images older than recent_count messages to 1024px.

    Scans message history and downgrades images beyond the recent window
    from 1536px to 1024px to save context window space. Uses stored originals
    to avoid quality degradation from double-compression.
    """
    total_messages = len(messages)
    cutoff_index = max(0, total_messages - recent_count)

    for msg_idx, image_list in list(_original_images.items()):
        if msg_idx >= cutoff_index:
            continue  # This message is still in recent window

        # Message is old, re-compress images to 1024px
        msg = messages[msg_idx]
        if not isinstance(msg.get("content"), list):
            continue

        for content_idx, original_url in image_list:
            if content_idx >= len(msg["content"]):
                continue

            part = msg["content"][content_idx]
            if part.get("type") == "image_url":
                # Re-compress from original to 1024px (lower quality)
                downgraded = _compress_image_for_llm(
                    original_url,
                    max_width=1024,
                    quality=75  # Slightly lower JPEG quality for older images
                )
                part["image_url"]["url"] = downgraded

        # Remove from tracking - already downgraded
        del _original_images[msg_idx]


def _write_sample_markdown(md_path: str, png_filename: str, llm_description: str, turn: int) -> None:
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    content = (
        "# Sample Debug Log\n\n"
        f"- turn: {turn}\n"
        f"- timestamp: {timestamp}\n\n"
        f"![sample](./{png_filename})\n\n"
        "## LLM Description\n\n"
        f"{llm_description.strip()}\n"
    )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(content)


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

async def run_agent():
    _trace("agent_start", model=MODEL, max_turns=MAX_TURNS, initial_prompt=INITIAL_PROMPT)
    os.makedirs(SAMPLE_LOG_DIR, exist_ok=True)
    # Load system prompt
    with open(SYSTEM_PROMPT_PATH, "r") as f:
        system_prompt = f.read()

    # LLM client (OpenRouter)
    llm = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    # Connect to MCP server (SSE transport — start server separately via run.sh)
    async with sse_client(MCP_SERVER_URL) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Discover tools
            tools_result = await session.list_tools()
            openai_tools = mcp_tools_to_openai_tools(tools_result.tools)
            tool_names = [t.name for t in tools_result.tools]
            print(f"[INIT] Connected to MCP server. Available tools: {tool_names}")
            _trace("mcp_tools_discovered", tool_names=tool_names, tool_count=len(tool_names))

            # Conversation history
            init_first_instruction = (
                "Before using any tool other than `init`, call `init` once. "
                "If a tool says resources are not initialized, call `init` and retry."
            )
            sample_log_instruction = (
                "After every `sample` call that returns images, your immediate next tool call "
                "must be `log_actions`, containing a concise factual description of visible image content."
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": init_first_instruction},
                {"role": "system", "content": sample_log_instruction},
                {"role": "user", "content": INITIAL_PROMPT},
            ]
            pending_sample_log = False
            pending_sample_artifact = None

            # Agentic loop
            for turn in range(MAX_TURNS):
                print(f"\n{'='*60}")
                print(f"[TURN {turn + 1}/{MAX_TURNS}]")
                print(f"{'='*60}")

                # Call LLM with retries (network/provider failures are transient)
                response = None
                # Re-compress old images to save context window
                _recompress_old_images(messages, recent_count=10)
                for attempt in range(1, LLM_MAX_RETRIES + 1):
                    try:
                        response = await llm.chat.completions.create(
                            model=MODEL,
                            messages=messages,
                            tools=openai_tools if openai_tools else None,
                        )
                        first_choice = response.choices[0]
                        first_message = first_choice.message
                        first_text = _extract_assistant_text(first_message.content)
                        if not first_message.tool_calls and not first_text:
                            raise RuntimeError("Empty assistant response content from LLM")
                        break
                    except Exception as e:
                        print(f"[WARN] LLM call failed {attempt}/{LLM_MAX_RETRIES}: {e}")
                        _trace("llm_retry", turn=turn + 1, attempt=attempt, max_retries=LLM_MAX_RETRIES, error=str(e))
                        if attempt == LLM_MAX_RETRIES:
                            raise RuntimeError(
                                f"LLM call failed after {LLM_MAX_RETRIES} attempts: {e}"
                            ) from e
                        await asyncio.sleep(2 ** (attempt - 1))

                _trace("llm_response", turn=turn + 1, finish_reason=response.choices[0].finish_reason)

                choice = response.choices[0]
                assistant_message = choice.message

                # Append assistant response to history
                messages.append(assistant_message.model_dump())

                # If no tool calls, agent is done (or just responding)
                if not assistant_message.tool_calls:
                    assistant_text = _extract_assistant_text(assistant_message.content)
                    print(f"[ASSISTANT] {assistant_text}")
                    _trace(
                        "assistant_message",
                        turn=turn + 1,
                        has_tool_calls=False,
                        content_preview=assistant_text[:300],
                    )
                    if pending_sample_log:
                        print("[WARN] sample log enforcement: assistant must call log_actions next.")
                        _trace("sample_log_enforcement", turn=turn + 1, status="missing_tool_call")
                        messages.append({
                            "role": "system",
                            "content": (
                                "Policy reminder: call `log_actions` now with a concise factual "
                                "description of the latest sampled images before any other action."
                            ),
                        })
                        continue
                    if choice.finish_reason == "stop":
                        print("\n[DONE] Agent finished.")
                        _trace("agent_done", reason="stop", turn=turn + 1)
                        break
                    continue

                # Process tool calls
                if pending_sample_log:
                    first_tool_name = assistant_message.tool_calls[0].function.name
                    if first_tool_name != "log_actions":
                        print(
                            "[WARN] sample log enforcement: first tool must be log_actions "
                            f"(got {first_tool_name})."
                        )
                        _trace(
                            "sample_log_enforcement",
                            turn=turn + 1,
                            status="wrong_first_tool",
                            first_tool=first_tool_name,
                        )
                        messages.append({
                            "role": "system",
                            "content": (
                                "Policy reminder: your next tool call must be `log_actions` to "
                                "describe the latest sampled images."
                            ),
                        })
                        continue

                for tool_idx, tool_call in enumerate(assistant_message.tool_calls):
                    fn_name = tool_call.function.name
                    fn_args_str = tool_call.function.arguments
                    try:
                        fn_args = json.loads(fn_args_str)
                    except json.JSONDecodeError:
                        print(f"[WARN] Invalid JSON args for tool {fn_name}: {fn_args_str}")
                        _trace("tool_args_parse_error", turn=turn + 1, tool=fn_name, raw_args=fn_args_str)
                        fn_args = {}

                    print(f"[TOOL CALL] {fn_name}({json.dumps(fn_args, ensure_ascii=False)[:200]})")
                    _trace("tool_call", turn=turn + 1, tool=fn_name, args=fn_args)

                    # Call MCP server
                    try:
                        mcp_result = await session.call_tool(fn_name, fn_args)
                        content_parts = parse_mcp_result(mcp_result)
                        _trace("tool_result", turn=turn + 1, tool=fn_name, status="ok", part_count=len(content_parts))
                    except Exception as e:
                        print(f"[TOOL ERROR] {fn_name}: {e}")
                        _trace("tool_result", turn=turn + 1, tool=fn_name, status="error", error=str(e))
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
                        # Store originals for potential re-compression
                        message_index = len(messages)
                        _original_images[message_index] = [
                            (i + 1, part["image_url"]["url"])  # +1 because text part is index 0
                            for i, part in enumerate(image_parts)
                        ]

                        # Compress images for LLM (payload + context window optimization)
                        compressed_image_parts = [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": _compress_image_for_llm(part["image_url"]["url"])
                                },
                            }
                            for part in image_parts
                        ]
                        messages.append({
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"[Image result from {fn_name}]"},
                                *compressed_image_parts,
                            ],
                        })
                        if fn_name == "sample":
                            pending_sample_log = True
                            sample_id = f"turn{turn + 1:03d}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
                            png_filename = f"{sample_id}.png"
                            md_filename = f"{sample_id}.md"
                            png_path = os.path.join(SAMPLE_LOG_DIR, png_filename)
                            md_path = os.path.join(SAMPLE_LOG_DIR, md_filename)
                            try:
                                _save_data_url_image(image_parts[0]["image_url"]["url"], png_path)
                                pending_sample_artifact = {
                                    "turn": turn + 1,
                                    "png_filename": png_filename,
                                    "md_path": md_path,
                                }
                                print(f"[LOG] Saved sample image to {png_path}")
                            except Exception as e:
                                print(f"[WARN] Failed to save sample image artifact: {e}")
                                pending_sample_artifact = None
                            _trace("sample_log_enforcement", turn=turn + 1, status="required")
                            messages.append({
                                "role": "user",
                                "content": (
                                    "Now call `log_actions` with 1-2 concise factual sentences "
                                    "describing visible content in the sampled images."
                                ),
                            })
                            for skipped_call in assistant_message.tool_calls[tool_idx + 1 :]:
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": skipped_call.id,
                                    "content": (
                                        "Skipped by policy: after `sample` image output, "
                                        "the next tool must be `log_actions`."
                                    ),
                                })
                            break

                    if fn_name == "log_actions" and pending_sample_log:
                        if pending_sample_artifact is not None:
                            llm_description = str(fn_args.get("msg", "")).strip()
                            if llm_description:
                                try:
                                    _write_sample_markdown(
                                        md_path=pending_sample_artifact["md_path"],
                                        png_filename=pending_sample_artifact["png_filename"],
                                        llm_description=llm_description,
                                        turn=pending_sample_artifact["turn"],
                                    )
                                    print(f"[LOG] Saved sample markdown to {pending_sample_artifact['md_path']}")
                                except Exception as e:
                                    print(f"[WARN] Failed to write sample markdown artifact: {e}")
                            else:
                                print("[WARN] log_actions msg is empty; sample markdown was not written.")
                        pending_sample_artifact = None
                        pending_sample_log = False
                        _trace("sample_log_enforcement", turn=turn + 1, status="satisfied")

            else:
                print(f"\n[DONE] Reached max turns ({MAX_TURNS}).")
                _trace("agent_done", reason="max_turns", max_turns=MAX_TURNS)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    _init_weave()
    asyncio.run(run_agent())


if __name__ == "__main__":
    main()
