import os
from typing import List

from nimbusagent.agent.completion import CompletionAgent

history: List[dict] = [
    {
        "role": "user",
        "content": "My favorite color is blue."
    }
]

agent = CompletionAgent(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    system_message="You are a helpful assistant.",
    message_history=history,
)

response = agent.ask("What is my favorite color?")
print(response)
