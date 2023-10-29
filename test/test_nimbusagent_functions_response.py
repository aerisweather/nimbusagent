import unittest

from nimbusagent.functions.responses import FuncResponse, InternalFuncResponse, DictFuncResponse


class TestFuncResponses(unittest.TestCase):

    def test_func_response_initialization(self):
        fr = FuncResponse(content="hello", summarize_only=True)
        self.assertEqual(fr.content, "hello")
        self.assertEqual(fr.summarize_only, True)

    def test_func_response_to_internal_response(self):
        fr = FuncResponse(content="hello")
        internal_response = fr.to_internal_response("MyFunc")
        self.assertEqual(internal_response.internal_thought['role'], 'function')
        self.assertEqual(internal_response.internal_thought['name'], 'MyFunc')
        self.assertEqual(internal_response.internal_thought['content'], 'hello')

    def test_internal_func_response_initialization(self):
        ifr = InternalFuncResponse(content="hello", internal_thought={'role': 'function'})
        self.assertEqual(ifr.content, "hello")
        self.assertEqual(ifr.internal_thought['role'], 'function')

    def test_dict_func_response_initialization(self):
        dfr = DictFuncResponse({'content': 'hello'})
        self.assertEqual(dfr.data['content'], 'hello')

    def test_dict_func_response_to_internal_response(self):
        dfr = DictFuncResponse({'content': 'hello'})
        internal_response = dfr.to_internal_response("MyFunc")
        self.assertEqual(internal_response.internal_thought['role'], 'function')
        self.assertEqual(internal_response.internal_thought['name'], 'MyFunc')
        self.assertEqual(internal_response.internal_thought['content'], 'hello')


if __name__ == '__main__':
    unittest.main()
