import unittest

from nimbusagent.memory.base import AgentMemory


class TestAgentMemory(unittest.TestCase):

    def test_initialization(self):
        memory = AgentMemory(max_tokens=10, max_messages=5)
        self.assertEqual(memory.max_tokens, 10)
        self.assertEqual(memory.max_messages, 5)
        self.assertEqual(memory.num_tokens, 0)
        self.assertEqual(memory.get_chat_length(), 0)

    def test_tokenize(self):
        memory = AgentMemory()
        self.assertEqual(memory.tokenize("hello"), 1)

    def test_add_entry(self):
        memory = AgentMemory()
        with self.assertRaises(ValueError):
            memory.add_entry({"content": "hello"})

        memory.add_entry({"role": "user", "content": "hello"})
        self.assertEqual(memory.get_chat_length(), 1)

    def test_trim_excess_entries(self):
        memory = AgentMemory(max_tokens=10, max_messages=2)
        memory.add_entry({"role": "user", "content": "hello"})
        memory.add_entry({"role": "user", "content": "world"})
        memory.add_entry({"role": "user", "content": "!"})
        self.assertEqual(memory.get_chat_length(), 2)
        self.assertEqual(memory.get_total_tokens(), 2)

    def test_get_chat_history(self):
        memory = AgentMemory()
        memory.add_entry({"role": "user", "content": "hello"})
        self.assertEqual(memory.get_chat_history(), [{"role": "user", "content": "hello"}])

    def test_clear_chat_history(self):
        memory = AgentMemory()
        memory.add_entry({"role": "user", "content": "hello"})
        memory.clear_chat_history()
        self.assertEqual(memory.get_chat_length(), 0)

    def test_set_chat_history(self):
        memory = AgentMemory()
        history = [{"role": "user", "content": "hello"}, {"role": "agent", "content": "world"}]
        memory.set_chat_history(history)
        self.assertEqual(memory.get_chat_history(), history)

    def test_get_chat_length(self):
        memory = AgentMemory()
        memory.add_entry({"role": "user", "content": "hello"})
        self.assertEqual(memory.get_chat_length(), 1)

    def test_get_total_tokens(self):
        memory = AgentMemory()
        memory.add_entry({"role": "user", "content": "hello"})
        self.assertEqual(memory.get_total_tokens(), 1)

    def test_resize(self):
        memory = AgentMemory(max_tokens=10, max_messages=2)
        memory.add_entry({"role": "user", "content": "hello"})
        memory.add_entry({"role": "user", "content": "world"})
        memory.resize(max_tokens=8, max_messages=1)
        self.assertEqual(memory.get_chat_length(), 1)
        self.assertEqual(memory.get_total_tokens(), 0)

    def test_initialization_with_initial_history(self):
        initial_history = [{"role": "user", "content": "hello"}, {"role": "agent", "content": "hi"}]
        memory = AgentMemory(initial_history=initial_history)
        self.assertEqual(memory.get_chat_length(), 2)
        self.assertEqual(memory.get_total_tokens(), 2)  # Assuming "hello" and "hi" are 1 token each

    def test_complex_string_tokenization(self):
        memory = AgentMemory()
        complex_string = "Hello, world! 😊 こんにちは"
        token_count = memory.tokenize(complex_string)
        self.assertGreater(token_count, 1)  # Token count should be more than 1 for a complex string

    def test_adding_multiple_entries(self):
        memory = AgentMemory(max_tokens=5, max_messages=3)
        entries = [
            {"role": "user", "content": "hello"},
            {"role": "user", "content": "world"},
            {"role": "agent", "content": "hi"},
        ]
        for entry in entries:
            memory.add_entry(entry)
        self.assertEqual(memory.get_chat_length(), 3)
        self.assertLessEqual(memory.get_total_tokens(), 5)

    def test_edge_case_for_token_limit(self):
        memory = AgentMemory(max_tokens=3)
        memory.add_entry({"role": "user", "content": "hello world"})  # Assuming this is 3 tokens
        memory.add_entry({"role": "user", "content": "another message"})  # This should not be added
        self.assertEqual(memory.get_chat_length(), 1)

    def test_adding_whitespace_entries(self):
        memory = AgentMemory()
        memory.add_entry({"role": "user", "content": "   "})
        self.assertEqual(memory.get_chat_length(), 0)

    def test_get_chat_history_as_text(self):
        memory = AgentMemory()
        memory.add_entry({"role": "user", "content": "hello"})
        text_history = memory.get_chat_history_as_text()
        self.assertEqual(text_history, "user: hello")


if __name__ == '__main__':
    unittest.main()
