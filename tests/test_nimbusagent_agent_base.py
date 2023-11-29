import os
import unittest
from unittest.mock import patch, MagicMock

import openai

from nimbusagent.agent.base import BaseAgent

os.environ['OPENAI_API_KEY'] = 'some key'


class TestBaseAgent(unittest.TestCase):

    @patch('openai.OpenAI')
    def test_initialization(self, mock_openai):
        agent = BaseAgent(openai_api_key="test_key")
        self.assertEqual(agent.model_name, 'gpt-4-0613')
        self.assertIsNotNone(agent.client)

    def test_set_system_message(self):
        agent = BaseAgent()
        agent.set_system_message("Test Message")
        expected_message = {"role": "system", "content": "Test Message"}
        self.assertEqual(agent.system_message, expected_message)

    @patch('nimbusagent.utils.helper.is_query_safe', return_value=False)
    def test_history_needs_moderation(self, mock_is_query_safe):
        agent = BaseAgent()
        history = [{'role': 'user', 'content': 'inappropriate content'}]
        self.assertTrue(agent._history_needs_moderation(history))

    @patch('openai.chat.completions.create')
    def test_create_chat_completion(self, mock_chat_create):
        mock_chat_create.return_value = MagicMock(spec=openai.types.chat.ChatCompletion)

        agent = BaseAgent(openai_api_key="test_key")

        # Test without using functions
        messages = [{'role': 'user', 'content': 'Hello'}]
        response = agent._create_chat_completion(messages, use_functions=False)
        mock_chat_create.assert_called_with(
            model=agent.model_name,
            temperature=agent.temperature,
            messages=messages,
            stream=False
        )
        self.assertIsInstance(response, openai.types.chat.ChatCompletion)

        # Test using functions
        agent.function_handler.functions = ["some_function"]
        response = agent._create_chat_completion(messages, use_functions=True)
        mock_chat_create.assert_called_with(
            model=agent.model_name,
            temperature=agent.temperature,
            messages=messages,
            functions=["some_function"],
            function_call='auto',
            stream=False
        )


if __name__ == '__main__':
    unittest.main()
