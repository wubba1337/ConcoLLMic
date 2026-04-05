from openai.types.chat import ChatCompletionMessageToolCall

from app.data_structures import MessageThread


def test_add_model_with_dict_tool_calls():
    msg_thread = MessageThread()
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "batch_tool", "arguments": '{"items": []}'},
        }
    ]

    msg_thread.add_model(message="", tools=tool_calls)
    messages = msg_thread.to_msg()

    assert len(messages) == 1
    assert messages[0]["role"] == "assistant"
    assert messages[0]["tool_calls"] == tool_calls


def test_add_model_with_openai_tool_call_objects():
    msg_thread = MessageThread()
    tool_obj = ChatCompletionMessageToolCall(
        id="call_2",
        type="function",
        function={"name": "batch_tool", "arguments": '{"items": [1]}'},
    )

    msg_thread.add_model(message="", tools=[tool_obj])
    messages = msg_thread.to_msg()

    assert len(messages) == 1
    assert messages[0]["role"] == "assistant"
    assert messages[0]["tool_calls"] == [
        {
            "id": "call_2",
            "type": "function",
            "function": {"name": "batch_tool", "arguments": '{"items": [1]}'},
        }
    ]
