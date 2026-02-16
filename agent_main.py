# %%
import os
import json
import dotenv
dotenv.load_dotenv()

import wandb

from agents import Agent, Runner, ModelSettings
from openai.types.shared import Reasoning

from search_tools import search, sample, aesthetics_rate
from commit_tools import commit, undo_commit, status, sample_from_committed, dataset_commits, log_actions

# Load system prompt
with open("system_prompt.md", "r") as f:
    system_prompt = f.read()
from openai import AsyncOpenAI
from agents import set_default_openai_client, set_tracing_disabled
from agents import Agent, Runner
custom_client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))
set_default_openai_client(custom_client)
set_tracing_disabled(True)
# Initialize agent
agent = Agent(name="Assistant",
              tools=[search, commit, sample, aesthetics_rate, undo_commit, status, sample_from_committed, log_actions],
              instructions=system_prompt,  
            #   model_settings=ModelSettings(
            #     reasoning=Reasoning(effort="medium"),
            #     parallel_tool_calls=False, 
            #   ),
            #   model="gpt-5") 
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
