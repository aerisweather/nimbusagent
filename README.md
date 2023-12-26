## Overview

NimbusAgent is a Python module that provides an interface for interacting with OpenAI's GPT models, including GPT-4 and
GPT-3.5 Turbo. The module is designed to facilitate easy integration of advanced AI capabilities into applications,
offering features like automatic moderation, memory management, and support for both streaming and non-streaming
responses.

## Features

- Integration with OpenAI GPT models (GPT-4, GPT-3.5 Turbo)
- Automated content moderation
- Customizable AI responses with adjustable parameters
- Support for both streaming and non-streaming response handling
- Internal memory management for maintaining conversation context
- Extensible function handling for custom AI functionalities

## Installation

To install NimbusAgent, run the following command:

```bash
pip install nimbusagent
```

## Usage

### CompletionAgent

First, import the necessary classes from the nimbusagent package, create an instance of the CompletionAgent class,
passing in your OpenAI API key and the name of the model you want to use:

```python
from nimbusagent.agent.completion import CompletionAgent

agent = CompletionAgent(
    openai_api_key="YOUR_OPENAI_API_KEY",
    model_name="gpt-4-0613",
    max_tokens=2000
)

response = agent.ask("What's the weather like today?")
print(response)
```

### StreamingAgent

If you want to use a streaming agent, create an instance of the StreamingAgent class instead. When using a streaming
agent, you can use the ask() method to send a prompt to the agent, and then iterate over the response to get the chunks
of text as they are generated:

```python
from nimbusagent.agent.completion import StreamingAgent

agent = StreamingAgent(
    openai_api_key="YOUR_OPENAI_API_KEY",
    model_name="gpt-4-0613",
    max_tokens=2000
)
response = agent.ask("What's the weather like today?")
for chunk in response:
    print(chunk)
```

### Configuration Parameters

When initializing an instance of `BaseAgent`, `CompletionAgent`, or `StreamingAgent`, several configuration parameters
can be passed to customize the agent's behavior. Below is a detailed description of these parameters:

#### `openai_api_key`

- **Description**: The API key for accessing OpenAI services.
- **Type**: `str`
- **Default**: `None` (The system will look for an environment variable `OPENAI_API_KEY` if not provided)

#### `model_name`

- **Description**: The name of the primary OpenAI GPT model to use.
- **Type**: `str`
- **Default**: `'gpt-4-0613'`

#### `secondary_model_name`

- **Description**: The name of the secondary OpenAI GPT model to use. The secondary model may be requested by a custom
  function to provide additional context for the AI, such as to rephrase a common response to it is custom. For these
  simpler responses you may choose to use a cheaper model, such as `gpt-3.5-turbo`.
- **Type**: `str`
- **Default**: `'gpt-3.5-turbo'`

#### `temperature`

- **Description**: Controls the randomness of the AI's responses.
- **Type**: `float`
- **Default**: `0.1`

#### `max_tokens`

- **Description**: The maximum number of tokens to generate in each response.
- **Type**: `int`
- **Default**: `500`

#### `functions`

- **Description**: A list of custom functions that the agent can use.
- **Type**: `Optional[list]`
- **Default**: `None`

#### `functions_embeddings`

- **Description**: Embeddings for the functions to help the AI understand them better.
- **Type**: `Optional[List[dict]]`
- **Default**: `None`

#### `functions_always_use`

- **Description**: Functions that should always be used by the agent.
- **Type**: `Optional[List[str]]`
- **Default**: `None`

#### `functions_pattern_groups`

- **Description**: Pattern groups for matching functions to user queries.
- **Type**: `Optional[List[dict]]`
- **Default**: `None`

#### `functions_k_closest`

- **Description**: The number of closest functions to consider when handling a query.
- **Type**: `int`
- **Default**: `3`

#### `system_message`

- **Description**: A system message that sets the context for the agent.
- **Type**: `str`
- **Default**: `"You are a helpful assistant."`

#### `message_history`

- **Description**: Pre-existing chat history for the agent to consider.
- **Type**: `Optional[List[dict]]`
- **Default**: `None`

#### `calling_function_start_callback` and `calling_function_stop_callback`

- **Description**: Callback functions triggered when a custom function starts or stops.
- **Type**: `Optional[callable]`
- **Default**: `None`

#### `perform_moderation`

- **Description**: Whether the agent should moderate the responses for inappropriate content.
- **Type**: `bool`
- **Default**: `True`

#### `moderation_fail_message`

- **Description**: The message returned when moderation fails.
- **Type**: `str`
- **Default**: `"I'm sorry, I can't help you with that as it is not appropriate."`

#### `memory_max_entries` and `memory_max_tokens`

- **Description**: Limits for the agent's memory in terms of entries and tokens.
- **Types**: `int`
- **Defaults**: `20` for `memory_max_entries`, `2000` for `memory_max_tokens`

#### `internal_thoughts_max_entries`

- **Description**: The maximum number of entries for the agent's internal thoughts.
- **Type**: `int`
- **Default**: `8`

#### `loops_max`

- **Description**: The maximum number of loops the agent will process for a single query.
- **Type**: `int`
- **Default**: `8`

#### `send_events`

- **Description**: Whether the agent should send events (useful for streaming responses).
- **Type**: `bool`
- **Default**: `False`

### Example of Initialization

Here's an example of how you might initialize a `CompletionAgent` with some of these parameters:

```python
agent = CompletionAgent(
    openai_api_key="your_api_key_here",
    model_name="gpt-4-0613",
    temperature=0.7,
    max_tokens=500,
    perform_moderation=True,
    system_message="You are a helpful assistant."
)
```

You can customize these parameters based on the specific requirements of your application or the behaviors you expect
from the agent.

### Advanced Usage and Examples

- For more advanced use cases such as handling multi-turn conversations or integrating custom AI functionalities, refer
  to the 'examples' directory in our repository.

## Getting Support

- For support, please open an issue in our GitHub repository.

## License

- NimbusAgent is released under the MIT License See the LICENSE file for more details.

## Todo

- [ ] Add support for Azure OpenAI API
- [ ] Add support for OpenAI Assistant API
- [x] Add support for new OpenAI Tool Calls vs now deprecated Function calls
- [x] Add Function call examples

## Stay Updated

- Follow our GitHub repository to stay updated with new releases and changes.

