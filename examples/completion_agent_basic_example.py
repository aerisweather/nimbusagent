import os

from nimbusagent.agent.completion import CompletionAgent

agent = CompletionAgent(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    system_message="You are a helpful assistant."
)

response = agent.ask("How many states are in the United States?")
print(response)
