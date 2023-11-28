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


if __name__ == '__main__':
    unittest.main()
