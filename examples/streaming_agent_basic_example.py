import os
import sys

from nimbusagent.agent.streaming import StreamingAgent

agent = StreamingAgent(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    system_message="You are a helpful assistant."
)

response = agent.ask("Can you list the last 4 US presidents?")
for chunk in response:
    sys.stdout.write(chunk)

sys.stdout.write("\n\n")
sys.stdout.flush()
