import tiktoken


class AgentMemory:
    """
    Class that stores the chat history and token counts for the agent.
    This is basic memory that utilizes a simple list to store the chat history, and limits the list to a maximum
    number of tokens and entries. Tiktoken is used to tokenize the content.

    :param max_tokens:  The maximum number of tokens to store in the chat history.
    :param max_messages:  The maximum number of messages to store in the chat history.
    :param initial_history:  The initial chat history to use.  If None, the chat history will be empty.
    """

    def __init__(
        self,
        max_tokens: int,
        max_messages: int,
        token_encoding: str,
        initial_history: list[dict[str, str]] | None = None,
    ):
        self.encoding = tiktoken.get_encoding(token_encoding)
        self.chat_history = []
        self.token_counts = (
            []
        )  # Stores token counts for corresponding chat_history entries
        self.num_tokens = 0
        self.max_tokens = max_tokens
        self.max_messages = max_messages

        if initial_history:
            self.set_chat_history(initial_history)

    def tokenize(self, content: str) -> int:
        """
        Tokenize the content and return the number of tokens.
        :param content:  The content to tokenize.
        :return:  The number of tokens.
        """
        return len(self.encoding.encode(content))

    def clear_chat_history(self):
        """
        Clear the chat history.
        """
        self.chat_history = []
        self.token_counts = []
        self.num_tokens = 0

    def add_entry(self, entry: dict[str, str]):
        """
        Add an entry to the chat history.
        :param entry:  The entry to add. The entry must have both 'role' and 'content' fields.
        """
        # logging.info(entry)
        if "role" not in entry or "content" not in entry:
            raise ValueError("Each entry must have both 'role' and 'content' fields.")

        content = entry["content"].strip()
        # skip empty messages
        if not content:
            return

        token_count = self.tokenize(entry["content"])
        self.token_counts.append(token_count)
        self.chat_history.append(entry)  # content remains untokenized
        self.num_tokens += token_count
        self._trim_excess_entries()

    def append(self, entry: dict[str, str]):
        """
        Add an entry to the chat history.
        :param entry:  The entry to add. The entry must have both 'role' and 'content' fields.
        """
        self.add_entry(entry)

    def _trim_excess_entries(self):
        """
        Trim the chat history to the maximum number of tokens and entries.
        """
        while (self.num_tokens > self.max_tokens) or (
            len(self.chat_history) > self.max_messages
        ):
            self.num_tokens -= self.token_counts.pop(0)
            self.chat_history.pop(0)

    def get_chat_history(self) -> list[dict[str, str]]:
        """
        Get the chat history.
        :return:  The chat history.
        """
        return self.chat_history.copy()

    def get_chat_history_as_text(self) -> str:
        """
        Get the chat history as text.
        :return:  The chat history as text, in "role: content" format.
        """
        return "\n".join(
            [f"{entry['role']}: {entry['content']}" for entry in self.chat_history]
        )

    def set_chat_history(self, new_history: list[dict[str, str]]):
        """
        Set the chat history.
        :param new_history: The new chat history. Each entry must have both 'role' and 'content' fields.
        """
        self.clear_chat_history()
        last_entry = None
        for entry in new_history:
            # Ensure only 'role' and 'content' fields are present
            filtered_entry = {
                "role": entry.get("role", "").strip(),
                "content": entry.get("content", "").strip(),
            }

            # Skip entries missing 'role' or 'content'
            if not filtered_entry["role"] or not filtered_entry["content"]:
                continue

            # Skip duplicate consecutive entries based on 'content'
            if last_entry is None or last_entry["content"] != filtered_entry["content"]:
                self.add_entry(filtered_entry)
                last_entry = filtered_entry

    def get_chat_length(self) -> int:
        """
        Get the number of entries in the chat history.
        :return:  The number of entries in the chat history.
        """
        return len(self.chat_history)

    def get_total_tokens(self) -> int:
        """
        Get the total number of tokens in the chat history.
        :return:  The total number of tokens in the chat history.
        """
        return self.num_tokens

    def resize(
        self,
        max_tokens_resize: int | None = None,
        max_messages_resize: int | None = None,
    ):
        """
        Resize the chat history. If the new maximum number of tokens or entries is smaller than the current number of
        tokens or entries, the chat history will be trimmed to the new maximum.
        :param max_tokens:  The new maximum number of tokens.  If None, no limit is enforced.
        :param max_messages:  The new maximum number of entries.  If None, no limit is enforced.
        """
        if max_tokens_resize is not None:
            self.max_tokens = max_tokens_resize
            self._trim_excess_entries()

        if max_messages_resize is not None:
            self.max_messages = max_messages_resize
            while len(self.chat_history) > self.max_messages:
                dropped_tokens = self.chat_history.pop(0)
                self.num_tokens -= len(dropped_tokens)
