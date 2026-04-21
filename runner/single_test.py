# from core.adapter.openai_adapter import OpenAIAdapter
from tools.simple_websocket import websocket
def single_test(test):
    adapter = OpenAIAdapter()

def single_test(input_text):
    adapter = OpenAIAdapter()
    response = adapter.get_response(input_text)
    print(f"输入: {input_text}")
    print(f"输出: {response}")
    return response

if __name__ == "__main__":
    single_test("我忘记密码了，怎么办？")
