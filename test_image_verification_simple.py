#!/usr/bin/env python3
"""
Simple test to verify the LLM receives image outputs from tools.
Creates a simple test image and asks the LLM to describe it.
"""

import os
import dotenv
dotenv.load_dotenv()

from PIL import Image, ImageDraw
import io
import base64

from agents import Agent, Runner, ModelSettings, function_tool, ToolOutputImage
from openai import AsyncOpenAI
from agents import set_default_openai_client, set_tracing_disabled, set_default_openai_api

# Setup OpenRouter client
custom_client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))
set_default_openai_client(custom_client)
set_tracing_disabled(True)

# FIX 1: Use Chat Completions API
set_default_openai_api("chat_completions")

# FIX 2: Monkey-patch to preserve images
from agents.models.chatcmpl_converter import Converter
_original_items_to_messages = Converter.items_to_messages

@classmethod
def _patched_items_to_messages(cls, items, model=None, preserve_thinking_blocks=False, preserve_tool_output_all_content=False):
    print(f"\n[DEBUG] items_to_messages called with preserve_tool_output_all_content={preserve_tool_output_all_content}")

    messages = _original_items_to_messages.__func__(
        cls, items, model=model,
        preserve_thinking_blocks=preserve_thinking_blocks,
        preserve_tool_output_all_content=True  # Force to True
    )

    # DEBUG: Log what content types are being sent
    for i, msg in enumerate(messages):
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            if isinstance(content, list):
                types = [c.get("type") for c in content]
                print(f"\n✓ [VERIFICATION] Tool message #{i} sent with content types: {types}")
                for item in content:
                    if item.get("type") == "image_url":
                        print(f"  - Image detected in tool output!")
            else:
                print(f"\n✗ [VERIFICATION] Tool message #{i} is a string ({len(content)} chars) - no images!")
    return messages

Converter.items_to_messages = _patched_items_to_messages
print("[DEBUG] Monkey-patch installed for Converter.items_to_messages")

# Create a tool that returns a test image
@function_tool
def get_test_image():
    """Return a simple test image with colored shapes."""
    # Create a simple test image with colored shapes
    img = Image.new('RGB', (400, 300), color='white')
    draw = ImageDraw.Draw(img)

    # Draw shapes
    draw.rectangle([50, 50, 150, 150], fill='red', outline='darkred', width=3)
    draw.ellipse([200, 50, 300, 150], fill='blue', outline='darkblue', width=3)
    draw.polygon([(200, 200), (250, 100), (300, 200)], fill='green', outline='darkgreen')
    draw.text((130, 250), "Test Image", fill='black')

    # Encode to base64
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    base64_image = base64.b64encode(buffer.getvalue()).decode()

    print(f"\n[TOOL OUTPUT] Returning test image with colored shapes")
    print(f"              Image size: {img.size}, Format: JPEG")

    return ToolOutputImage(
        image=base64_image,
        format="jpeg"
    )

# Create agent with the tool
agent = Agent(
    name="ImageDescriber",
    tools=[get_test_image],
    instructions="You are an image analyst. When asked, use the get_test_image tool to fetch an image, then describe what you see in detail. Focus on colors, shapes, composition, and layout.",
    model_settings=ModelSettings(parallel_tool_calls=False),
    model="google/gemini-3-flash-preview"
)

print("=" * 70)
print("IMAGE VERIFICATION TEST")
print("=" * 70)
print("\n[SETUP] Agent configured to:")
print("  - Use OpenRouter with Gemini 3 Flash")
print("  - Preserve images in tool outputs (preserve_tool_output_all_content=True)")
print("  - Tool: get_test_image() returns ToolOutputImage with test image")
print("\n[TEST] Running agent with prompt to describe image\n")

result = Runner.run_sync(
    agent,
    "Get the test image and describe what you see in detail. What shapes and colors are present?",
    max_turns=5
)

# Extract the actual response text
response_text = str(result.output) if hasattr(result, 'output') else str(result)

print("\n" + "=" * 70)
print("LLM RESPONSE:")
print("=" * 70)
print(response_text)
print("\n" + "=" * 70)
print("VERIFICATION:")
print("=" * 70)

# Check if LLM actually described the visual content
visual_keywords = ["red", "blue", "green", "rectangle", "circle", "triangle", "ellipse", "shape", "color"]
response_lower = response_text.lower()
found_keywords = [kw for kw in visual_keywords if kw in response_lower]

if found_keywords:
    print(f"✓ SUCCESS: The LLM received and analyzed the image!")
    print(f"  Found visual vocabulary: {', '.join(found_keywords)}")
    print(f"\n  This confirms the image was successfully passed to the model.")
else:
    print(f"✗ ISSUE: The LLM did not describe visual content.")
    print(f"  (No color/shape keywords found in response)")
    print(f"\n  The image may not have been included in the message sent to the LLM.")
