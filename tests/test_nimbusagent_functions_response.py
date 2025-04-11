from nimbusagent.functions.responses import FuncResponse, DictFuncResponse


class TestFuncResponses:

    def test_func_response_initialization(self):
        fr = FuncResponse(content="hello", summarize_only=True)
        assert fr.content == "hello"
        assert fr.summarize_only == True

    def test_dict_func_response_initialization(self):
        dfr = DictFuncResponse({"content": "hello"})
        assert dfr.data == {"content": "hello"}
        assert dfr.content == "hello"
