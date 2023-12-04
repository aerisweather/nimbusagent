import os
import unittest
from unittest.mock import patch, MagicMock

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

    def test_create_chat_completion(self):
        agent = BaseAgent(openai_api_key="test_key")

        # Mock the create method on the instance
        mock_chat_create = MagicMock()
        agent.client.chat.completions.create = mock_chat_create

        # Set the return value to a MagicMock
        mock_chat_create.return_value = MagicMock()

        # Test without using functions
        messages = [{'role': 'user', 'content': 'Hello'}]
        response = agent._create_chat_completion(messages, use_functions=False)

        # Validate that the mocked method was called with the expected arguments
        mock_chat_create.assert_called_with(
            model=agent.model_name,
            temperature=agent.temperature,
            messages=messages,
            stream=False
        )

        # Validate that the response is a MagicMock (mocked response)
        self.assertIsInstance(response, MagicMock)


if __name__ == '__main__':
    unittest.main()
