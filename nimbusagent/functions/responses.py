from typing import Optional, Dict

from pydantic import BaseModel


class FuncResponse(BaseModel):
    """
    A response from a function. This is the response that is returned from a function call. It contains the content of
    the response, whether to only summarize the content, whether to send the response directly to the user, the content
    to post to the chat history, the data to stream to the user, whether to use the secondary model, and whether to
    force no functions.

    :param content:  The content of the response.
    :param summarize_only:  Whether to only summarize the content.
    :param send_directly_to_user:  Whether to send the response directly to the user.
    :param post_content:  The content to post to the chat history.
    :param stream_data:  The data to stream to the user.
    :param use_secondary_model:  Whether to use the secondary model.
    :param force_no_functions:  Whether to force no functions.
    """
    name: str = None
    arguments: str = None
    content: str = None
    summarize_only: bool = False
    send_directly_to_user: bool = False
    post_content: Optional[str] = None
    stream_data: Optional[dict] = None
    use_secondary_model: bool = False
    force_no_functions: bool = False


class DictFuncResponse(FuncResponse):
    """
    A response from a function. This is the response that is returned from a function call. It contains the content of
    the response, whether to only summarize the content, whether to send the response directly to the user, the content
    to post to the chat history, the data to stream to the user, whether to use the secondary model, and whether to
    force no functions.
    """
    data: Dict = None

    def __init__(self, init_data: Dict):
        super().__init__()
        self.data = init_data
        self.content = init_data.get('content', '')
        self.summarize_only = init_data.get('summarize_only', False)
        self.send_directly_to_user = init_data.get('send_directly_to_user', False)
        self.post_content = init_data.get('post_content', None)
        self.stream_data = init_data.get('stream_data', None)
        self.use_secondary_model = init_data.get('use_secondary_model', False)
        self.force_no_functions = init_data.get('force_no_functions', False)
