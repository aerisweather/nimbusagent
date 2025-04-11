import os
from unittest.mock import patch

import openai.types
import requests
from nimbusagent.utils import helper

os.environ["OPENAI_API_KEY"] = "some key"


class TestHelperFunctions:
    @patch("openai.resources.Moderations.create")
    def test_is_query_safe_network_failure(self, mock_moderation_create):
        # Simulate a network failure using requests.exceptions.ConnectionError
        mock_moderation_create.side_effect = requests.exceptions.ConnectionError()

        # Call the is_query_safe function and expect it to handle the network failure gracefully
        result = helper.is_query_safe("test query")

        assert result == False  # if openai is unavailable then the query is unsafe

    @patch("openai.resources.Embeddings.create")
    def test_get_embedding(self, mock_embedding_create):
        # First part of the test
        embedding = openai.types.Embedding(
            embedding=[0.1, 0.2], index=0, object="embedding"
        )

        mock_embedding_create.return_value = openai.types.CreateEmbeddingResponse(
            id="emb-123",
            model="text-embedding-ada-002",
            object="list",
            data=[embedding],
            usage=openai.types.create_embedding_response.Usage(
                prompt_tokens=0, total_tokens=0
            ),
        )

        # {"data": [{"embedding": [0.1, 0.2]}]}
        assert helper.get_embedding("some text") == [0.1, 0.2]
        # Reset the mock
        mock_embedding_create.reset_mock()

        # Second part of the test
        mock_embedding_create.side_effect = Exception("Some error")
        assert helper.get_embedding("some text") is None

    def test_cosine_similarity(self):
        assert helper.cosine_similarity([1, 0], [0, 1]) == 0.0
        assert helper.cosine_similarity([1, 0], [1, 0]) == 1.0

    @patch("nimbusagent.utils.helper.get_embedding")
    def test_find_similar_embedding_list(self, mock_get_embedding):
        mock_get_embedding.return_value = [0.2, 0.1]
        function_embeddings = [
            {"name": "func1", "embedding": [0.2, 0.1]},
            {"name": "func2", "embedding": [0.1, 0.3]},
        ]
        result = helper.find_similar_embedding_list("some query", function_embeddings)

        assert result[0]["name"] == "func1"

    def test_combine_lists_unique(self):
        assert helper.combine_lists_unique([1, 2], [2, 3]) == [1, 2, 3]
        assert helper.combine_lists_unique([1, 2], [2, 3, 4]) == [1, 2, 3, 4]

    def test_combine_lists_unique_with_tuples(self):
        list1 = (1, 2)  # Using a tuple instead of a list
        set2 = {3, 4}  # Set can remain as it is

        # Call the combine_lists_unique function with a tuple and a set
        # Depending on your function's implementation, you might expect a specific result or an exception
        result = helper.combine_lists_unique(list1, set2)

        # Assert the expected behavior
        # If your function can handle tuples, check the resulting list
        # If not, you might use `self.assertRaises(TypeError, helper.combine_lists_unique, list1, set2)`
        assert result == [1, 2, 3, 4]

    @patch("nimbusagent.utils.helper.get_embedding")
    def test_find_similar_embedding_list_with_none_embeddings(self, mock_get_embedding):
        # noinspection PyTypeChecker
        result = helper.find_similar_embedding_list("some query", None)
        assert result is None

    @patch("nimbusagent.utils.helper.get_embedding")
    def test_find_similar_embedding_list_with_empty_embeddings(
        self, mock_get_embedding
    ):
        result = helper.find_similar_embedding_list("some query", [])
        assert result is None

    @patch("nimbusagent.utils.helper.get_embedding")
    def test_find_similar_embedding_list_with_empty_query(self, mock_get_embedding):
        function_embeddings = [{"name": "func1", "embedding": [0.2, 0.1]}]
        result = helper.find_similar_embedding_list("", function_embeddings)
        assert result is None

    @patch("nimbusagent.utils.helper.get_embedding")
    def test_find_similar_embedding_list_get_embedding_returns_none(
        self, mock_get_embedding
    ):
        mock_get_embedding.return_value = None
        function_embeddings = [{"name": "func1", "embedding": [0.2, 0.1]}]
        result = helper.find_similar_embedding_list("some query", function_embeddings)
        assert result is None
