"""
Test: call the test_image MCP tool, send the returned image to an LLM,
and print its description.  Verifies end-to-end image transport via MCP.

Requires the MCP server to be running (start via run.sh or manually).
"""

import os
import asyncio

import dotenv
dotenv.load_dotenv()

from openai import AsyncOpenAI
from mcp import ClientSession
from mcp.client.sse import sse_client

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8765/sse")
MODEL = "moonshotai/kimi-k2.5"
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "4"))


def _extract_text(content) -> str:
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


async def test_image_tool():
    llm = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    async with sse_client(MCP_SERVER_URL) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            print("[TEST] Connected to MCP server.")

            print("[TEST] Calling init tool...")
            init_result = await session.call_tool("init", {})
            if init_result.content:
                print(f"[TEST] Init: {init_result.content[0].text}")

            # Call test_image tool
            print("[TEST] Calling test_image tool...")
            result = await session.call_tool("test_image", {})

            # Parse result
            image_parts = []
            text_parts = []
            for block in result.content:
                if block.type == "text":
                    text_parts.append(block.text)
                    print(f"[TEST] Text: {block.text}")
                elif block.type == "image":
                    print(f"[TEST] Got image: {len(block.data)} base64 chars")
                    image_parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{block.mimeType};base64,{block.data}"
                        },
                    })

            if not image_parts:
                print("[FAIL] No image returned from test_image tool!")
                return

            # Send image to LLM and ask it to describe
            print(f"\n[TEST] Sending image to {MODEL} for description...")
            response = None
            for attempt in range(1, LLM_MAX_RETRIES + 1):
                try:
                    response = await llm.chat.completions.create(
                        model=MODEL,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "Describe this image in detail. What do you see?"},
                                    *image_parts,
                                ],
                            }
                        ],
                    )
                    description = _extract_text(response.choices[0].message.content)
                    if not description:
                        raise RuntimeError("Empty description content from LLM")
                    break
                except Exception as e:
                    print(f"[WARN] LLM call failed {attempt}/{LLM_MAX_RETRIES}: {e}")
                    if attempt == LLM_MAX_RETRIES:
                        raise RuntimeError(
                            f"LLM call failed after {LLM_MAX_RETRIES} attempts: {e}"
                        ) from e
                    await asyncio.sleep(2 ** (attempt - 1))

            print(f"\n{'='*60}")
            print(f"MODEL DESCRIPTION:")
            print(f"{'='*60}")
            print(description)
            print(f"{'='*60}")
            print("\n[PASS] Image transport and model vision verified!")


if __name__ == "__main__":
    asyncio.run(test_image_tool())
