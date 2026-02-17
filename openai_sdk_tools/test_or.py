from openai import AsyncOpenAI
from agents import set_default_openai_client, set_tracing_disabled
from agents import Agent, Runner
import dotenv, os
dotenv.load_dotenv()
custom_client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))
set_default_openai_client(custom_client)
set_tracing_disabled(True)

from agents import set_default_openai_api 

set_default_openai_api("chat_completions")


agent = Agent(name="Assistant", instructions="You are a helpful assistant", model="moonshotai/kimi-k2.5")

result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print(result.final_output) 