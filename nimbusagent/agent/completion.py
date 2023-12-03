from typing import Optional, Union

import openai

from nimbusagent.agent.base import BaseAgent


class CompletionAgent(BaseAgent):
    """
    Agent that can handle openai function calls and can generate responsee, without streaming.
    This agent is meant to be used in a non-streaming context, where the user cannot see the response as it is generated.
    This means it will take longer to generate a response, as we must wait for openAI to generate and respond.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # noinspection PyUnresolvedReferences
    def ask(self, query: str) -> Optional[str]:
        """
        Ask the agent a question and return the response.
        :param query:  The query to ask the agent.
        :return:  The response.
        """
        if self._needs_moderation(query):
            return self.moderation_fail_message

        self._clear_internal_thoughts()
        self.function_handler.get_functions_from_query_and_history(query, self.get_chat_history())
        self._append_to_chat_history('user', query)
        res = self._generate_response()
        self.last_response = res

        if res is None:
            return None
        elif isinstance(res, str):
            self._append_to_chat_history('function', res)
            return res
        else:
            self._append_to_chat_history(res.choices[0].message.role, res.choices[0].message.content)
            return res.choices[0].message.content

    # noinspection PyUnresolvedReferences
    def _generate_response(self) -> Optional[Union[openai.types.chat.ChatCompletion, str]]:
        """
        Generate a response object based on the response from the AI
        :return:  The response object.
        """
        loop = 0
        while loop < self.loops_max:
            loop += 1

            if len(self.internal_thoughts) == 1:
                if self.function_handler.always_use:
                    self.function_handler.remove_functions_mappings(self.function_handler.always_use)

            res = self._create_chat_completion(
                [self.system_message] + self.chat_history.get_chat_history() + self.internal_thoughts
            )
            finish_reason = res.choices[0].finish_reason

            if finish_reason == 'stop' or len(self.internal_thoughts) > self.internal_thoughts_max_entries:
                return res
            elif finish_reason == 'function_call':
                func_name = res.choices[0].message.function_call.name
                args_str = res.choices[0].message.function_call.arguments
                func_results = self.function_handler.handle_function_call(func_name, args_str)

                if func_results:
                    if func_results.assistant_thought:
                        self.internal_thoughts.append(func_results.assistant_thought)

                    if 'internal_thought' in func_results:
                        self.internal_thoughts.append(func_results['internal_thought'])

                    if func_results.send_directly_to_user and func_results.content:
                        return func_results.content
            else:
                raise ValueError(f"Unexpected finish reason: {finish_reason}")

        return None
