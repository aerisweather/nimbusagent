from nimbusagent.memory.base import AgentMemory
import pytest


class TestAgentMemory:
    @pytest.fixture(autouse=True)
    def setup_memory(self):
        self.memory = AgentMemory(
            max_tokens=2, max_messages=2, token_encoding="cl100k_base"
        )

    @pytest.mark.parametrize("max_tokens, max_messages", [(10, 5)])
    def test_initialization(self, max_tokens, max_messages):
        memory = AgentMemory(
            max_tokens=max_tokens,
            max_messages=max_messages,
            token_encoding="cl100k_base",
        )

        assert memory.max_tokens == max_tokens
        assert memory.max_messages == max_messages
        assert memory.num_tokens == 0
        assert memory.get_chat_length() == 0

    def test_tokenize(self):
        assert self.memory.tokenize("hello") == 1

    def test_add_entry(self):
        self.memory.add_entry({"role": "user", "content": "hello"})
        assert self.memory.get_chat_length() == 1

        with pytest.raises(ValueError):
            self.memory.add_entry({"content": "hello"})

    @pytest.mark.parametrize("max_tokens, max_messages", [(10, 2)])
    def test_trim_excess_entries(self, max_tokens, max_messages):
        memory = AgentMemory(
            max_tokens=max_tokens,
            max_messages=max_messages,
            token_encoding="cl100k_base",
        )
        memory.add_entry({"role": "user", "content": "hello"})
        memory.add_entry({"role": "user", "content": "world"})
        memory.add_entry({"role": "user", "content": "!"})
        assert memory.get_chat_length() == 2
        assert memory.get_total_tokens() == 2

    def test_get_chat_history(self):
        self.memory.add_entry({"role": "user", "content": "hello"})
        assert self.memory.get_chat_history() == [{"role": "user", "content": "hello"}]

    def test_clear_chat_history(self):
        self.memory.add_entry({"role": "user", "content": "hello"})
        self.memory.clear_chat_history()
        assert self.memory.get_chat_length() == 0

    def test_set_chat_history(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "agent", "content": "world"},
        ]
        self.memory.set_chat_history(history)
        assert self.memory.get_chat_history() == history

    def test_get_chat_length(self):
        self.memory.add_entry({"role": "user", "content": "hello"})
        assert self.memory.get_chat_length() == 1

    def test_get_total_tokens(self):
        self.memory.add_entry({"role": "user", "content": "hello"})
        assert self.memory.get_total_tokens() == 1

    @pytest.mark.parametrize("max_tokens, max_messages", [(10, 2)])
    def test_resize(self, max_tokens, max_messages):
        memory = AgentMemory(
            max_tokens=max_tokens,
            max_messages=max_messages,
            token_encoding="cl100k_base",
        )
        memory.add_entry({"role": "user", "content": "hello"})
        memory.add_entry({"role": "user", "content": "world"})
        memory.resize(max_tokens_resize=8, max_messages_resize=1)
        assert memory.get_chat_length() == 1
        assert memory.get_total_tokens() == 0

    def test_initialization_with_initial_history(self):
        initial_history = [
            {"role": "user", "content": "hello"},
            {"role": "agent", "content": "hi"},
        ]
        memory = AgentMemory(
            max_tokens=10,
            max_messages=10,
            initial_history=initial_history,
            token_encoding="cl100k_base",
        )
        # Assuming "hello" and "hi" are 1 token each
        assert memory.get_chat_length() == 2
        assert memory.get_total_tokens() == 2

    def test_complex_string_tokenization(self):
        complex_string = "Hello, world! 😊 こんにちは"
        token_count = self.memory.tokenize(complex_string)
        assert token_count > 1  # Token count should be more than 1 for a complex string

    def test_adding_multiple_entries(self):
        memory = AgentMemory(max_tokens=5, max_messages=3, token_encoding="cl100k_base")
        entries = [
            {"role": "user", "content": "hello"},
            {"role": "user", "content": "world"},
            {"role": "agent", "content": "hi"},
        ]
        for entry in entries:
            memory.add_entry(entry)
        assert memory.get_chat_length() == 3
        assert memory.get_total_tokens() <= 5

    def test_edge_case_for_token_limit(self):
        memory = AgentMemory(
            max_tokens=3, max_messages=10, token_encoding="cl100k_base"
        )

        # Assuming this is 3 tokens
        memory.add_entry({"role": "user", "content": "hello world"})
        # This should not be added
        memory.add_entry({"role": "user", "content": "another message"})
        assert memory.get_chat_length() == 1

    def test_adding_whitespace_entries(self):
        self.memory.add_entry({"role": "user", "content": "   "})
        assert self.memory.get_chat_length() == 0

    def test_get_chat_history_as_text(self):
        self.memory.add_entry({"role": "user", "content": "hello"})
        text_history = self.memory.get_chat_history_as_text()
        assert text_history == "user: hello"
