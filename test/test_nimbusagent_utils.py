import unittest
from unittest.mock import patch

from nimbusagent.utils import helper


class TestHelperFunctions(unittest.TestCase):

    @patch("openai.Moderation.create")
    def test_is_query_safe(self, mock_moderation_create):
        mock_moderation_create.return_value = {'results': [{'flagged': False}]}
        self.assertTrue(helper.is_query_safe("Is it going to rain today?"))

        mock_moderation_create.return_value = {'results': [{'flagged': True}]}
        self.assertFalse(helper.is_query_safe("Some unsafe query"))

    @patch("openai.Embedding.create")
    def test_get_embedding(self, mock_embedding_create):
        mock_embedding_create.return_value = {"data": [{"embedding": [0.1, 0.2]}]}
        self.assertEqual(helper.get_embedding("some text"), [0.1, 0.2])

        mock_embedding_create.side_effect = Exception("Some error")
        self.assertIsNone(helper.get_embedding("some text"))

    def test_cosine_similarity(self):
        self.assertAlmostEqual(helper.cosine_similarity([1, 0], [0, 1]), 1)
        self.assertAlmostEqual(helper.cosine_similarity([1, 0], [1, 0]), 0)

    @patch("api.nimbusagent.utils.helper.get_embedding")
    def test_find_similar_embedding_list(self, mock_get_embedding):
        mock_get_embedding.return_value = [0.2, 0.1]
        function_embeddings = [
            {'name': 'func1', 'embedding': [0.2, 0.1]},
            {'name': 'func2', 'embedding': [0.1, 0.3]}
        ]
        result = helper.find_similar_embedding_list("some query", function_embeddings)
        self.assertEqual(result[0]['name'], 'func1')

    def test_combine_lists_unique(self):
        self.assertEqual(helper.combine_lists_unique([1, 2], [2, 3]), [1, 2, 3])
        self.assertEqual(helper.combine_lists_unique([1, 2], {3, 4}), [1, 2, 3, 4])


if __name__ == '__main__':
    unittest.main()
