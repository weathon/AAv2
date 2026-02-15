# %%
import os
import json
import dotenv
dotenv.load_dotenv()

from agents import Agent, Runner, ModelSettings
from openai.types.shared import Reasoning

from search_tools import search, sample, aesthetics_rate
from commit_tools import commit, undo_commit, status, sample_from_committed, dataset_commits

# Load system prompt
with open("system_prompt.md", "r") as f:
    system_prompt = f.read()

# Initialize agent
agent = Agent(name="Assistant",
              tools=[search, commit, sample, aesthetics_rate, undo_commit, status, sample_from_committed],
              instructions=system_prompt,
              model_settings=ModelSettings(
                reasoning=Reasoning(effort="medium"),
              ),
              model="gpt-5.2")
            #   model="google/gemini-3-flash-preview")

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

result = Runner.run_sync(agent, "Psychedelic art", max_turns=100)
