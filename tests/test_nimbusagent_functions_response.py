import unittest

from nimbusagent.functions.responses import FuncResponse, DictFuncResponse


class TestFuncResponses(unittest.TestCase):

    def test_func_response_initialization(self):
        fr = FuncResponse(content="hello", summarize_only=True)
        self.assertEqual(fr.content, "hello")
        self.assertEqual(fr.summarize_only, True)

    def test_dict_func_response_initialization(self):
        dfr = DictFuncResponse({'content': 'hello'})
        self.assertEqual(dfr.data['content'], 'hello')
        self.assertEqual(dfr.content, 'hello')


if __name__ == '__main__':
    unittest.main()
