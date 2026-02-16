"""
Test: call the test_image MCP tool, send the returned image to an LLM,
and print its description.  Verifies end-to-end image transport via MCP.
"""

import os
import sys
import json
import asyncio

import dotenv
dotenv.load_dotenv()

from openai import AsyncOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "server.py")
MODEL = "moonshotai/kimi-k2.5"
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "4"))


async def test_image_tool():
    llm = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[SERVER_SCRIPT],
        env={**os.environ},
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
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
                    break
                except Exception as e:
                    print(f"[WARN] LLM call failed {attempt}/{LLM_MAX_RETRIES}: {e}")
                    if attempt == LLM_MAX_RETRIES:
                        raise RuntimeError(
                            f"LLM call failed after {LLM_MAX_RETRIES} attempts: {e}"
                        ) from e
                    await asyncio.sleep(2 ** (attempt - 1))

            description = response.choices[0].message.content
            print(f"\n{'='*60}")
            print(f"MODEL DESCRIPTION:")
            print(f"{'='*60}")
            print(description)
            print(f"{'='*60}")
            print("\n[PASS] Image transport and model vision verified!")


if __name__ == "__main__":
    asyncio.run(test_image_tool())
