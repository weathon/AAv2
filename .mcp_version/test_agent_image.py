"""Minimal test: save image to tmp file and let the agent Read it."""

import os
import shutil
import tempfile
import anyio

from claude_agent_sdk import (
    tool, create_sdk_mcp_server, ClaudeSDKClient, ClaudeAgentOptions,
    AssistantMessage, TextBlock,
)


TMP_DIR = tempfile.mkdtemp(prefix="agent_img_")


@tool("get_test_image", "Returns the file path of a test image.", {})
async def get_test_image(args):
    src = os.path.join(os.path.dirname(__file__), "123.jpg")
    dst = os.path.join(TMP_DIR, "test_image.jpg")
    shutil.copy2(src, dst)
    return {"content": [
        {"type": "text", "text": f"Test image saved at: {dst}\nUse the Read tool to view it, then describe what you see."},
    ]}


server = create_sdk_mcp_server("test-image", tools=[get_test_image])


async def main():
    options = ClaudeAgentOptions(
        mcp_servers={"test-image": server},
        allowed_tools=["Read"],
        permission_mode="bypassPermissions",
        max_turns=5,
    )
    async with ClaudeSDKClient(options=options) as client:
        await client.query("Call the get_test_image tool and describe what you see in the image.")
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(block.text)


if __name__ == "__main__":
    anyio.run(main)
