"""
Test script to verify the MCP server starts correctly and tools work.
Connects as an MCP client via stdio, lists tools, and calls a few.
"""

import os
import sys
import json
import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "server.py")


async def test_server():
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[SERVER_SCRIPT],
        env={**os.environ},
    )

    print("[TEST] Connecting to MCP server...")
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            print("[TEST] Server initialized!")

            # 1. List tools
            print("\n[TEST] Listing tools...")
            tools_result = await session.list_tools()
            tool_names = [t.name for t in tools_result.tools]
            print(f"  Found {len(tool_names)} tools: {tool_names}")

            expected = {"search", "sample", "commit", "undo_commit", "status",
                        "sample_from_committed", "aesthetics_rate", "log_actions"}
            missing = expected - set(tool_names)
            extra = set(tool_names) - expected
            if missing:
                print(f"  MISSING tools: {missing}")
            if extra:
                print(f"  UNEXPECTED tools: {extra}")
            if not missing and not extra:
                print("  All 8 expected tools present!")

            # Print tool schemas
            for tool in tools_result.tools:
                print(f"\n  Tool: {tool.name}")
                print(f"    Description: {tool.description[:100] if tool.description else 'None'}...")
                print(f"    Schema keys: {list(tool.inputSchema.get('properties', {}).keys())}")

            # 2. Test log_actions (lightweight, no GPU)
            print("\n[TEST] Calling log_actions...")
            result = await session.call_tool("log_actions", {"msg": "MCP server test run"})
            print(f"  Result: {result.content[0].text if result.content else 'empty'}")

            # 3. Test status (lightweight)
            print("\n[TEST] Calling status...")
            result = await session.call_tool("status", {})
            text = result.content[0].text if result.content else "empty"
            print(f"  Result: {text[:200]}")

            # 4. Test search (uses embeddings + GPU)
            print("\n[TEST] Calling search...")
            result = await session.call_tool("search", {
                "query": "psychedelic mandala",
                "dataset": "artwork",
                "negative_prompts": ["watermark", "text overlay"],
                "negative_threshold": 0.3,
                "t": 5,
            })
            for block in result.content:
                if block.type == "text":
                    print(f"  Text: {block.text}")
                elif block.type == "image":
                    print(f"  Image: base64 data ({len(block.data)} chars)")

            # 5. Test sample (uses embeddings)
            print("\n[TEST] Calling sample...")
            result = await session.call_tool("sample", {
                "query": "psychedelic mandala",
                "dataset": "artwork",
                "min_threshold": 0.3,
                "max_threshold": 0.6,
                "count": 3,
            })
            for block in result.content:
                if block.type == "text":
                    print(f"  Text: {block.text}")
                elif block.type == "image":
                    print(f"  Image: base64 data ({len(block.data)} chars)")

            # 6. Test commit
            print("\n[TEST] Calling commit...")
            result = await session.call_tool("commit", {
                "query": "psychedelic mandala",
                "dataset": "artwork",
                "threshold": 0.5,
                "negative_prompts": ["watermark"],
                "negative_threshold": 0.3,
                "message": "test commit from MCP test script",
            })
            commit_text = result.content[0].text if result.content else ""
            print(f"  Result: {commit_text}")

            # Extract commit ID for undo
            commit_id = None
            if "Committed with ID:" in commit_text:
                commit_id = commit_text.split("Committed with ID: ")[1].split(",")[0]

            # 7. Test status after commit
            print("\n[TEST] Calling status after commit...")
            result = await session.call_tool("status", {})
            print(f"  Result: {result.content[0].text[:300] if result.content else 'empty'}")

            # 8. Test sample_from_committed
            if commit_id:
                print(f"\n[TEST] Calling sample_from_committed({commit_id})...")
                result = await session.call_tool("sample_from_committed", {
                    "commit_id": commit_id,
                    "n": 3,
                })
                for block in result.content:
                    if block.type == "text":
                        print(f"  Text: {block.text}")
                    elif block.type == "image":
                        print(f"  Image: base64 data ({len(block.data)} chars)")

            # 9. Test undo_commit
            if commit_id:
                print(f"\n[TEST] Calling undo_commit({commit_id})...")
                result = await session.call_tool("undo_commit", {"commit_id": commit_id})
                print(f"  Result: {result.content[0].text if result.content else 'empty'}")

            print("\n[TEST] All tests completed!")


if __name__ == "__main__":
    asyncio.run(test_server())
