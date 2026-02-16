# %%
import os
import json
import dotenv
dotenv.load_dotenv()

import wandb

from agents import Agent, Runner, ModelSettings

from search_tools import search, sample, aesthetics_rate
from commit_tools import commit, undo_commit, status, sample_from_committed, dataset_commits, log_actions

# Load system prompt
with open("system_prompt.md", "r") as f:
    system_prompt = f.read()
from openai import AsyncOpenAI
from agents import set_default_openai_client, set_tracing_disabled, set_default_openai_api
custom_client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))
set_default_openai_client(custom_client)
set_tracing_disabled(True)

# FIX 1: Use Chat Completions API instead of Responses API
# OpenRouter doesn't fully support the Responses API format for tool outputs with images
set_default_openai_api("chat_completions")

# FIX 2: Monkey-patch items_to_messages to preserve image content in tool outputs
# By default, the chat completions converter strips images from tool outputs (extract_text_content).
# Gemini via OpenRouter supports images in tool results, so we need preserve_tool_output_all_content=True.
from agents.models.chatcmpl_converter import Converter
_original_items_to_messages = Converter.items_to_messages

@classmethod
def _patched_items_to_messages(cls, items, model=None, preserve_thinking_blocks=False, preserve_tool_output_all_content=False):
    # Always preserve image content in tool outputs for OpenRouter/Gemini
    return _original_items_to_messages.__func__(
        cls, items, model=model,
        preserve_thinking_blocks=preserve_thinking_blocks,
        preserve_tool_output_all_content=True
    )

Converter.items_to_messages = _patched_items_to_messages

# FIX 3: Remove Reasoning parameter (OpenAI-specific, not supported by Gemini via OpenRouter)
# Initialize agent without reasoning settings
agent = Agent(name="Assistant",
              tools=[search, commit, sample, aesthetics_rate, undo_commit, status, sample_from_committed, log_actions],
              instructions=system_prompt,
              model_settings=ModelSettings(
                parallel_tool_calls=False,
              ),
              model="google/gemini-3-flash-preview")

# Initialize dataset.json if it doesn't exist, or load existing commits
if os.path.exists("dataset.json"):
    with open("dataset.json", "r") as f:
        try:
            dataset_commits.update(json.load(f))
        except json.JSONDecodeError:
            dataset_commits.clear()
else:
    with open("dataset.json", "w") as f:
        json.dump({}, f)

wandb.init(project="aas2", name="Psychedelic art")

result = Runner.run_sync(agent, "Psychedelic art", max_turns=100)

wandb.finish()
