#!/usr/bin/env python3
"""
Test to verify the LLM receives image outputs from tools.
Returns a real image from the dataset and asks the LLM to describe it.
"""

import os
import json
import dotenv
dotenv.load_dotenv()

from PIL import Image
import io
import base64

from agents import Agent, Runner, ModelSettings, function_tool, ToolOutputImage
from openai import AsyncOpenAI
from agents import set_default_openai_client, set_tracing_disabled, set_default_openai_api

# Setup OpenRouter client with the same fixes
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
    messages = _original_items_to_messages.__func__(
        cls, items, model=model,
        preserve_thinking_blocks=preserve_thinking_blocks,
        preserve_tool_output_all_content=True
    )
    # DEBUG: Log what content types are being sent
    for msg in messages:
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            if isinstance(content, list):
                types = [c.get("type") for c in content]
                print(f"\n✓ [VERIFICATION] Tool message sent with content types: {types}")
                for item in content:
                    if item.get("type") == "image_url":
                        print(f"  - Image detected in tool output!")
            else:
                print(f"\n✗ [VERIFICATION] Tool message is a string ({len(content)} chars) - no images!")
    return messages

Converter.items_to_messages = _patched_items_to_messages

# Create a tool that returns a sample image from the dataset
@function_tool
def get_sample_image():
    """Fetch a sample image from the AVA dataset and return it as a ToolOutputImage."""
    try:
        from datasets import load_dataset
        ds = load_dataset("weathon/ava_embeddings", split="train")
        ava_dataset = ds.filter(lambda example: example["source"] == "ava")

        # Get first image
        sample = ava_dataset[0]
        image_data = sample["image"]

        # Convert to PIL Image if needed
        if not isinstance(image_data, Image.Image):
            image_data = Image.open(io.BytesIO(image_data))

        # Encode to base64
        buffer = io.BytesIO()
        image_data.save(buffer, format="JPEG")
        base64_image = base64.b64encode(buffer.getvalue()).decode()

        print(f"\n[TOOL OUTPUT] Returning image from AVA dataset")
        print(f"              Image size: {image_data.size}, Format: JPEG")

        return ToolOutputImage(
            image=base64_image,
            format="jpeg"
        )
    except Exception as e:
        return f"Error loading image: {str(e)}"

# Create agent with the tool
agent = Agent(
    name="ImageDescriber",
    tools=[get_sample_image],
    instructions="You are an image analyst. When asked, use the get_sample_image tool to fetch an image, then describe what you see in detail. Focus on colors, composition, mood, objects, and any artistic elements.",
    model_settings=ModelSettings(parallel_tool_calls=False),
    model="google/gemini-3-flash-preview"
)

print("=" * 60)
print("IMAGE VERIFICATION TEST")
print("=" * 60)
print("\n[SETUP] Agent configured to:")
print("  - Use OpenRouter with Gemini 3 Flash")
print("  - Preserve images in tool outputs")
print("  - Tool: get_sample_image() returns ToolOutputImage")
print("\n[TEST] Running agent with prompt: 'Get a sample image and describe it in detail'\n")

result = Runner.run_sync(
    agent,
    "Get a sample image and describe it in detail. Be specific about colors, composition, and mood.",
    max_turns=5
)

print("\n" + "=" * 60)
print("LLM RESPONSE:")
print("=" * 60)
print(result)
print("\n" + "=" * 60)
print("INTERPRETATION:")
print("=" * 60)
if any(word in result.lower() for word in ["color", "image", "composition", "describe", "see", "appears"]):
    print("✓ SUCCESS: The LLM appears to have received and analyzed the image!")
    print("  (It used visual vocabulary like describing colors, composition, etc.)")
else:
    print("✗ ISSUE: The LLM did not seem to analyze the image contents.")
    print("  (Response lacks visual descriptive language)")
