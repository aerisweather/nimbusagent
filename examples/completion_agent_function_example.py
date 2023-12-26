import json
import os
from typing import Literal, Dict

from nimbusagent.agent.completion import CompletionAgent


# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_weather(location: str, unit: Literal["celsius", "fahrenheit"] = "fahrenheit") -> Dict:
    """
    Get the current weather in a given location

    :param location: The city and state, e.g. San Francisco, CA
    :param unit: The unit to return the temperature in, either celsius or fahrenheit
    :return: The current weather in the given location
    """
    if "tokyo" in location.lower():
        content = json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
    elif "san francisco" in location.lower():
        content = json.dumps({"location": "San Francisco", "temperature": "30", "unit": unit})
    elif "paris" in location.lower():
        content = json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
    else:
        content = json.dumps({"location": location, "temperature": "unknown"})

    return {"content": content}


agent = CompletionAgent(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model_name="gpt-4-1106-preview",
    system_message="You are a helpful assistant.",
    functions=[get_current_weather],
    use_tool_calls=True  # If False, will disable tool calls and force the deprecated function calls
)

response = agent.ask("What's the weather like in San Francisco, Tokyo, and Paris?")
print(response)
