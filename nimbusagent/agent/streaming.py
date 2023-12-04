import json
import logging
import time
from typing import Generator, List

from nimbusagent.agent.base import BaseAgent, HAVING_TROUBLE_MSG


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
            yield self.moderation_fail_message
            return

        self._clear_internal_thoughts()
        self.function_handler.get_functions_from_query_and_history(query, self.get_chat_history())
        self._append_to_chat_history('user', query)

        ai_response = self._generate_streaming_response(max_retries=max_retries)
        content_accumulated = []
        for content in ai_response:
            content_accumulated.append(content)
            yield content

        self._append_to_chat_history('assistant', "".join(content_accumulated))

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
                    return f"{' '.join(post_content)}\n"
                return ""

            loops = 0
            post_content_items = []
            use_secondary_model = False
            force_no_functions = False
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
                        if delta.function_call:
                            if delta.function_call.name:
                                func_call["name"] = delta.function_call.name
                            if delta.function_call.arguments:
                                func_call["arguments"] += delta.function_call.arguments

                        if message.choices[0].finish_reason == "function_call":
                            if self.send_events:
                                json_data = json.dumps(self.function_handler.get_args(func_call['arguments']))
                                yield f"[[[function:{func_call['name']}:{json_data}]]]"

                            # Handle function call
                            logging.info("Handling function call: %s", func_call)
                            func_results = self.function_handler.handle_function_call(func_call["name"],
                                                                                      func_call["arguments"])
                            if func_results is not None:
                                if func_results.stream_data and self.send_events:
                                    for key, value in func_results.stream_data.items():
                                        json_value = json.dumps(value)
                                        yield f"[[[data:{key}:{json_value}]]]"

                                if func_results.send_directly_to_user and func_results.content:
                                    yield func_results.content
                                    yield output_post_content(post_content_items)
                                    return

                                if func_results.assistant_thought:
                                    self.internal_thoughts.append(func_results.assistant_thought)
                                if func_results.internal_thought:
                                    self.internal_thoughts.append(func_results.internal_thought)
                                if func_results.post_content:
                                    post_content_items.append(func_results.post_content)
                                if func_results.use_secondary_model:
                                    use_secondary_model = True
                                if func_results.force_no_functions:
                                    force_no_functions = True

                        content = delta.content
                        if content is not None:
                            has_content = True
                            yield delta.content

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
                    yield "AI temporarily unavailable."
                    break

            if loops >= self.loops_max:
                yield HAVING_TROUBLE_MSG

        return generate()
