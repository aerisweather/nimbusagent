from typing import Optional

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
    content: str = None
    summarize_only: bool = False
    send_directly_to_user: bool = False
    post_content: Optional[str] = None
    stream_data: Optional[dict] = None
    use_secondary_model: bool = False
    force_no_functions: bool = False

    def to_internal_response(self, func_name: str, args_str: str = None):
        """
        Convert this response to an internal response.
        :param func_name:  The name of the function.
        :param args_str:  The arguments of the function.
        :return:  The internal response.
        """
        if not self.content:
            return None

        asst_thought_content = f"#{func_name}"
        if args_str:
            args_str = args_str.replace("\n", " ")
            asst_thought_content += f"({args_str})"
        else:
            asst_thought_content += "()"
        internal_asst_thought = {'role': 'assistant', 'content': asst_thought_content}
        internal_msg = {'role': 'function', 'name': func_name, 'content': self.content}
        return InternalFuncResponse(content=self.content, send_directly_to_user=self.send_directly_to_user,
                                    internal_thought=internal_msg,
                                    assistant_thought=internal_asst_thought,
                                    post_content=self.post_content,
                                    stream_data=self.stream_data,
                                    use_secondary_model=self.use_secondary_model,
                                    force_no_functions=self.force_no_functions)


class InternalFuncResponse(FuncResponse):
    """
    An internal response from a function. This is the response that is returned from a function call. It contains the
    content of the response, whether to send the response directly to the user, the content to post to the chat history,
    the data to stream to the user, whether to use the secondary model, and whether to force no functions.
    :param content:  The content of the response.
    :param send_directly_to_user:  Whether to send the response directly to the user.
    :param post_content:  The content to post to the chat history.
    :param stream_data:  The data to stream to the user.
    :param use_secondary_model:  Whether to use the secondary model.
    :param force_no_functions:  Whether to force no functions.
    """
    internal_thought: dict = None
    assistant_thought: dict = None


class DictFuncResponse:
    """
    A response from a function. This is the response that is returned from a function call. It contains the content of
    the response, whether to only summarize the content, whether to send the response directly to the user, the content
    to post to the chat history, the data to stream to the user, whether to use the secondary model, and whether to
    force no functions.
    """
    def __init__(self, data: dict):
        self.data = data

    def to_internal_response(self, func_name: str):
        """
        Convert this response to an internal response.
        :param func_name:  The name of the function.
        :return:  The internal response.
        """
        content, send_directly_to_user, post_content, stream_data, use_secondary_model, force_no_functions = (
            self.data.get('content', ''),
            self.data.get('send_directly_to_user', False),
            self.data.get('post_content', None),
            self.data.get('data', None),
            self.data.get('use_secondary_model', False),
            self.data.get('force_no_functions', False)
        )
        if not content:
            return None
        res_msg = {'role': 'function', 'name': func_name, 'content': content}
        return InternalFuncResponse(content=content, send_directly_to_user=send_directly_to_user,
                                    internal_thought=res_msg,
                                    post_content=post_content,
                                    stream_data=stream_data,
                                    use_secondary_model=use_secondary_model,
                                    force_no_functions=force_no_functions)
