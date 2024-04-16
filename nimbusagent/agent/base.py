import os
from typing import Optional, List, Union, Literal, Dict

import openai
from openai import OpenAI

from nimbusagent.functions.handler import FunctionHandler
from nimbusagent.memory.base import AgentMemory
from nimbusagent.utils.helper import is_query_safe, FUNCTIONS_EMBEDDING_MODEL

SYS_MSG = """You are a helpful assistant."""

MODERATION_FAIL_MSG = """I'm sorry, I can't help you with that as it is not appropriate."""

HAVING_TROUBLE_MSG = """I'm sorry, I'm having trouble understanding you."""
DEFAULT_MODEL_NAME = 'gpt-4-0613'
DEFAULT_TEMP = 0.1
DEFAULT_SECONDARY_MODEL_NAME = 'gpt-3.5-turbo'


class BaseAgent:
    def __init__(
            self,
            openai_api_key: str = None,
            model_name: str = DEFAULT_MODEL_NAME,
            secondary_model_name: str = DEFAULT_SECONDARY_MODEL_NAME,
            temperature: float = DEFAULT_TEMP,
            max_tokens: int = 500,

            functions: Optional[list] = None,
            functions_embeddings: Optional[List[dict]] = None,
            functions_embeddings_model: str = FUNCTIONS_EMBEDDING_MODEL,
            functions_always_use: Optional[List[str]] = None,
            functions_pattern_groups: Optional[List[dict]] = None,
            functions_k_closest: int = 3,
            function_min_similarity: float = 0.5,

            function_max_tokens: int = 2000,
            use_tool_calls: bool = True,

            system_message: str = SYS_MSG,
            message_history: Optional[List[Dict[str, str]]] = None,

            calling_function_start_callback: Optional[callable] = None,
            calling_function_stop_callback: Optional[callable] = None,

            perform_moderation: bool = True,
            moderation_fail_message: str = MODERATION_FAIL_MSG,

            memory_max_entries: int = 20,
            memory_max_tokens: int = 2000,

            internal_thoughts_max_entries: int = 8,
            loops_max: int = 10,

            send_events: bool = False,
            max_event_size: int = 2000,

            on_complete: callable = None,
    ):
        """
        Base Agent Class for Nimbus Agent

        Args:
            openai_api_key: the OpenAI API key to use
            model_name: The name of the model to use (default: 'gpt-4-0613')
            secondary_model_name: The name of the secondary model to use (default: 'gpt-3.5-turbo')
            temperature: The temperature for the response sampling (default: 0.1)
            functions: The list of functions to use (default: None)
            functions_embeddings: The list of function embeddings to use (default: None)
            functions_embeddings_model: The model to use for function embeddings (default: 'text-embedding-ada-002')
            functions_pattern_groups: The list of function pattern groups to use (default: None)
            functions_k_closest: The number of closest embedding functions to use (default: 3)
            function_min_similarity: The minimum similarity to use for embedding functions (default: 0.5)
            functions_always_use: The list of functions to always use (default: None)
            function_max_tokens: The maximum number of tokens to allow for function call. (default: 2500) 0 = unlimited
            use_tool_calls: True if parallel functions should be allowed (default: True). Functions are being
                            deprecated though tool_calls are still a bit beta, so for now this can be set to
                            False to continue using function calls.
            system_message: The message to send to the user when the agent starts
                            (default: "You are a helpful assistant.")
            message_history: The message history to use (default: None)
            calling_function_start_callback: The callback to call when a function is called (default: None)
            calling_function_stop_callback: The callback to call when a function is stopped (default: None)
            perform_moderation: True if moderation should be performed (default: True)
            moderation_fail_message: The message to send to the user when a message is not appropriate
                                    (default: "I'm sorry, I can't help you with that as it is not appropriate.")
            memory_max_entries: The maximum number of entries to store in the memory (default: 20)
            memory_max_tokens: The maximum number of tokens to store in the memory (default: 2000)
            internal_thoughts_max_entries: The maximum number of entries to store in the internal thoughts (default: 3)
            loops_max: The maximum number of loops to allow (default: 5)
            send_events: True if events should be sent (default: False)
            max_event_size: The maximum size of an event (default: 2000)
            on_complete: The callback to call when the agent completes a response.
                The response is passed to the callable. This can be useful with streaming. (default: None)
        """

        self.client = OpenAI(api_key=openai_api_key if openai_api_key is not None else os.getenv("OPENAI_API_KEY"))

        # self.internal_thoughts: A list that captures the agent's intermediate
        # processing and thoughts during a single 'ask' session. It gets cleared
        # at the beginning of every new 'ask' to ensure no residual information
        # affects the new processing.
        self.internal_thoughts = []
        self.internal_thoughts_max_entries = internal_thoughts_max_entries
        self.model_name = model_name
        self.secondary_model_name = secondary_model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_message = None
        self.set_system_message(system_message)
        self.last_response = None
        self.perform_moderation = perform_moderation
        self.moderation_fail_message = moderation_fail_message
        self.loops_max = loops_max
        self.send_events = send_events
        self.max_event_size = max_event_size
        self.calling_function_start_callback = calling_function_start_callback
        self.calling_function_stop_callback = calling_function_stop_callback
        self.on_complete = on_complete

        self.chat_history = AgentMemory(max_messages=memory_max_entries, max_tokens=memory_max_tokens)
        if message_history is not None:
            if self._history_needs_moderation(message_history):
                raise ValueError('The message history contains inappropriate content.')
            self.chat_history.set_chat_history(message_history)

        self.function_handler = self._init_function_handler(
            functions=functions,
            functions_embeddings=functions_embeddings,
            functions_embeddings_model=functions_embeddings_model,
            functions_k_closest=functions_k_closest,
            functions_always_use=functions_always_use,
            functions_pattern_groups=functions_pattern_groups,
            function_max_tokens=function_max_tokens,
            function_min_similarity=function_min_similarity)
        self.use_tool_calls = use_tool_calls

    def set_system_message(self, message: str) -> None:
        """Sets the system message.
        :param message: The system message to set
        """
        self.system_message = {"role": "system", "content": message}

    def _init_function_handler(self,
                               functions: Optional[List],
                               functions_embeddings: Optional[List],
                               functions_embeddings_model: str = FUNCTIONS_EMBEDDING_MODEL,
                               functions_k_closest: int = 3,
                               function_min_similarity: float = 0.5,
                               functions_always_use: Optional[List[str]] = None,
                               functions_pattern_groups: Optional[List[dict]] = None,
                               function_max_tokens: int = 0) -> FunctionHandler:
        """Initializes the function handler.
        Returns a FunctionHandler instance.

        :param functions: The list of functions to use
        :param functions_embeddings: The list of function embeddings to use
        :param functions_k_closest: The number of closest functions to use
        :param functions_always_use: The list of functions to always use
        :param functions_pattern_groups: The list of function pattern groups to use
        :return: A FunctionHandler instance
         """

        return FunctionHandler(
            functions=functions,
            embeddings=functions_embeddings,
            embeddings_model=functions_embeddings_model,
            k_nearest=functions_k_closest,
            min_similarity=function_min_similarity,
            always_use=functions_always_use,
            pattern_groups=functions_pattern_groups,
            calling_function_start_callback=self.calling_function_start_callback,
            calling_function_stop_callback=self.calling_function_stop_callback,
            max_tokens=function_max_tokens,
            chat_history=self.chat_history
        )

    # noinspection PyUnresolvedReferences
    def _create_chat_completion(
            self, messages: list, use_functions: bool = True,
            function_call: Union[str, Literal['auto', 'none']] = 'auto',
            stream=False,
            use_secondary_model: bool = False, force_no_functions: bool = False
    ) -> openai.types.chat.ChatCompletion:

        """Creates a chat completion, streaming or not.
        :param messages: The messages to use
        :param use_functions: True if functions should be used
        :param function_call: The function call to use, 'auto' or 'none' or a function name
        :param stream: True if streaming should be used
        :param use_secondary_model: True if the secondary model should be used
        :param force_no_functions: True if functions should be forced to not be used
        :return: An openai chat completion
        """
        model_name = self.secondary_model_name if use_secondary_model else self.model_name

        if use_functions and self.function_handler.functions and not force_no_functions:
            if self.use_tool_calls:
                # noinspection PyTypeChecker
                res = self.client.chat.completions.create(
                    model=model_name,
                    temperature=self.temperature,
                    messages=messages,
                    tools=self.function_handler.functions_to_tools(),
                    tool_choice=function_call,
                    stream=stream)
            else:
                # noinspection PyTypeChecker
                res = self.client.chat.completions.create(
                    model=model_name,
                    temperature=self.temperature,
                    messages=messages,
                    functions=self.function_handler.functions,
                    function_call=function_call,
                    stream=stream)
        else:
            res = self.client.chat.completions.create(
                model=model_name,
                temperature=self.temperature,
                messages=messages,
                stream=stream)
        return res

    def _history_needs_moderation(self, history: List[Dict[str, str]]) -> bool:
        """Handles history moderation.
        Returns True if the history contains inappropriate content, False otherwise.
        :param history: The history to check
        :return: True if the history contains inappropriate content, False otherwise
        """
        if not self.perform_moderation or not history:
            return False

        content_list = [d['content'] for d in history if 'content' in d]

        return self._needs_moderation(" ".join(content_list))

    def _needs_moderation(self, query: str) -> bool:
        """Checks if a query requires moderation.
        Returns True if the query requires moderation, False otherwise.
        :param query: The query to check
        :return: True if the query requires moderation, False otherwise
        """
        return self.perform_moderation and not is_query_safe(query)

    def _clear_internal_thoughts(self) -> None:
        """Clears the internal thoughts of the agent."""
        self.internal_thoughts = []

    def _append_to_chat_history(self, role: str, content: str) -> None:
        """Appends a new message to the chat history.
        :param role: The role of the message
        :param content: The content of the message
        """
        self.chat_history.append({'role': role, 'content': content})

    # noinspection PyUnresolvedReferences
    def get_last_response(self) -> Optional[Union[openai.types.chat.ChatCompletion, str]]:
        """Returns the last response.
        :return: The last response
        """
        return self.last_response

    def get_chat_history(self) -> List[Dict[str, str]]:
        """Returns the chat history.
        :return: The chat history
        """
        return self.chat_history.get_chat_history()

    def get_functions(self) -> Optional[list]:
        """
        Returns the functions.
        :return: The functions
        """
        return self.function_handler.functions

    def clear_chat_history(self) -> None:
        """Clears the chat history."""
        self.chat_history.clear_chat_history()

    def handle_on_complete(self) -> None:
        """Handles the on_complete callback."""
        if self.on_complete and self.last_response:
            self.on_complete(self.last_response)

    def _clear_last_response(self) -> None:
        """Clears the last response."""
        self.last_response = ""
