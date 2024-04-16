import json
import logging
import time
from typing import Generator, List

from nimbusagent.agent.base import BaseAgent, HAVING_TROUBLE_MSG

EVENT_TYPE_FUNCTION = "function"
EVENT_TYPE_DATA = "data"


class StreamingAgent(BaseAgent):
    """Agent that streams responses to the user and can hanldle openai function calls.
    This agent is meant to be used in a streaming context, where the user can see the response as it is generated.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def ask(self, query: str, max_retries: int = 1) -> Generator[str, None, None]:
        """
        Ask the agent a question and return a generator that yields the response.
        :param query:  The query to ask the agent.
        :param max_retries:  The maximum number of times to retry the query if the AI fails to respond.
        :return:  A generator that yields the response.
        """
        if self._needs_moderation(query):
            self.last_response = self.moderation_fail_message
            yield self.moderation_fail_message

        else:
            self._clear_internal_thoughts()
            self._clear_last_response()
            self.function_handler.get_functions_from_query_and_history(query, self.get_chat_history())
            self._append_to_chat_history('user', query)

            ai_response = self._generate_streaming_response(max_retries=max_retries)
            content_accumulated = []
            for content in ai_response:
                content_accumulated.append(content)
                yield content

            self.last_response = "".join(content_accumulated)
            self._append_to_chat_history('assistant', self.last_response)

        self.handle_on_complete()

    def _generate_streaming_response(self, max_retries: int = 1) -> Generator[str, None, None]:
        """
        Generate a response from the AI and return a generator that yields the response.
        :param max_retries:  The maximum number of times to retry the query if the AI fails to respond.
        :return:  A generator that yields the response.
        """

        def generate() -> Generator[str, None, None]:
            """
            Generate a response from the AI and return a generator that yields the response.
            :return:  A generator that yields the response.
            """
            retries = max_retries

            def output_post_content(post_content: List[str]):
                if post_content:
                    post_content_str = f"{' '.join(post_content)}\n"
                    return post_content_str
                return ""

            def output_content(out_content: str):
                if out_content:
                    return out_content
                return ""

            def output_event(event_type: str, name: str, data: any):

                if not data:
                    return f"[[[{event_type}:{name}]]]"

                if not isinstance(data, str):
                    data = json.dumps(data)
                    if len(data) > self.max_event_size:
                        data = '{"error":"data too large"}'

                return f"[[[{event_type}:{name}:{data}]]]"

            loops = 0
            post_content_items = []
            use_secondary_model = False
            force_no_functions = False
            tool_calls = []
            while loops < self.loops_max:
                loops += 1
                has_content = False
                try:
                    if len(self.internal_thoughts) == 1:
                        if self.function_handler.always_use:
                            self.function_handler.remove_functions_mappings(self.function_handler.always_use)

                    stream = self._create_chat_completion(
                        messages=[self.system_message] + self.chat_history.get_chat_history() + self.internal_thoughts,
                        stream=True,
                        use_secondary_model=use_secondary_model,
                        force_no_functions=force_no_functions
                    )
                    func_call = {
                        "name": None,
                        "arguments": "",
                    }
                    use_secondary_model = False
                    force_no_functions = False

                    for message in stream:
                        if message is None or not message.choices or not message.choices[0]:
                            continue

                        delta = message.choices[0].delta
                        if not delta:
                            break

                        if delta.tool_calls:
                            tool_call = delta.tool_calls[0]
                            index = tool_call.index
                            if index == len(tool_calls):
                                tool_calls.append({
                                    "id": None,
                                    "type": "function",
                                    "function": {
                                        "name": "",
                                        "arguments": "",
                                    }
                                })

                            if tool_call.id:
                                tool_calls[index]['id'] = tool_call.id
                            if tool_call.function:
                                if tool_call.function.name:
                                    tool_calls[index]['function']['name'] = tool_call.function.name
                                if tool_call.function.arguments:
                                    tool_calls[index]['function']['arguments'] += tool_call.function.arguments

                        elif delta.function_call:
                            if delta.function_call.name:
                                func_call["name"] = delta.function_call.name
                            if delta.function_call.arguments:
                                func_call["arguments"] += delta.function_call.arguments

                        finish_reason = message.choices[0].finish_reason

                        if finish_reason == "tool_calls":
                            self.internal_thoughts.append({
                                "role": "assistant",
                                'content': None,
                                'tool_calls': tool_calls
                            })

                            # Handle tool calls
                            logging.info("Handling tool calls: %s", tool_calls)
                            content_send_directly_to_user = []

                            for tool_call in tool_calls:
                                func_name = tool_call['function']["name"]
                                if func_name is None:
                                    continue

                                func_args = tool_call['function']["arguments"]

                                if self.send_events:
                                    yield output_event(EVENT_TYPE_FUNCTION, func_name, func_args)

                                func_results = self.function_handler.handle_function_call(func_name, func_args)
                                if func_results is not None:
                                    if func_results.stream_data and self.send_events:
                                        for key, value in func_results.stream_data.items():
                                            yield output_event(EVENT_TYPE_DATA, key, value)

                                    if func_results.send_directly_to_user and func_results.content:
                                        content_send_directly_to_user.append(func_results.content)
                                        continue

                                    if func_results.content:
                                        self.internal_thoughts.append({
                                            'tool_call_id': tool_call['id'],
                                            "role": "tool",
                                            'name': func_name,
                                            'content': func_results.content
                                        })

                                    if func_results.use_secondary_model:
                                        use_secondary_model = True
                                    if func_results.force_no_functions:
                                        force_no_functions = True

                            if content_send_directly_to_user:
                                yield output_content("\n".join(content_send_directly_to_user))
                                yield output_post_content(post_content_items)
                                return

                            tool_calls = []  # reset tool calls

                        elif finish_reason == "function_call":
                            if self.send_events:
                                yield output_event(EVENT_TYPE_FUNCTION, func_call['name'],
                                                   json.dumps(self.function_handler.get_args(func_call['arguments'])))

                            # Handle function call
                            logging.info("Handling function call: %s", func_call)
                            func_results = self.function_handler.handle_function_call(func_call["name"],
                                                                                      func_call["arguments"])
                            if func_results is not None:
                                if func_results.stream_data and self.send_events:
                                    for key, value in func_results.stream_data.items():
                                        yield output_event(EVENT_TYPE_DATA, key, value)

                                if func_results.send_directly_to_user and func_results.content:
                                    yield func_results.content
                                    yield output_post_content(post_content_items)
                                    return

                                # Add the function call to the internal thoughts so the AI knows it called it
                                self.internal_thoughts.append({
                                    "role": "assistant",
                                    'content': None,
                                    'function_call': {
                                        'name': func_call['name'],
                                        'arguments': func_call['arguments']
                                    }
                                })

                                self.internal_thoughts.append({
                                    "role": "function",
                                    'content': func_results.content,
                                    'name': func_call['name']
                                })

                                if func_results.post_content:
                                    post_content_items.append(func_results.post_content)
                                if func_results.use_secondary_model:
                                    use_secondary_model = True
                                if func_results.force_no_functions:
                                    force_no_functions = True

                        content = delta.content
                        if content is not None:
                            has_content = True
                            yield output_content(delta.content)

                        if message.choices[0].finish_reason == 'stop':
                            yield output_post_content(post_content_items)
                            return
                        if len(self.internal_thoughts) > self.internal_thoughts_max_entries:
                            if post_content_items:
                                yield output_post_content(post_content_items)
                            else:
                                num_thoughts = len(self.internal_thoughts)
                                logging.error(f"Too many internal thoughts: {num_thoughts}.")
                                yield "Too many internal thoughts."
                            return

                except Exception as e:
                    logging.error("Exception encountered: %s (%s)", str(e), type(e).__name__, exc_info=True)

                    if retries > 0 and not has_content:
                        retries -= 1
                        time.sleep(1)
                        continue
                    yield output_content("AI temporarily unavailable.")
                    break

            if loops >= self.loops_max:
                yield output_content(HAVING_TROUBLE_MSG)

        return generate()
