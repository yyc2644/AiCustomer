# 加载百炼的 API Key 用于调用通义千问大模型
import os
from config.load_key import load_key
load_key()
from openai import OpenAI

print(f'''你配置的 API Key 是：{os.environ["DASHSCOPE_API_KEY"][:5]+"*"*5}''')
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def generate_sentences(word: str, num_sentences: int = 10):

    user_prompt = f"""请用印地语生成 {num_sentences} 个不同的完整句子，每个句子都必须包含单词 "{word}"。
    要求：
    1. 每个句子都必须是自然、流畅的印地语句子
    2. 每个句子都必须包含单词 "{word}"
    3. 句子之间要有不同的语境和表达方式
    4. 直接输出句子，每行一个句子，不要编号，不要其他说明文字
    
    生成的句子："""

    system_prompt = "你是一个印地语专家，擅长生成自然流畅的印地语句子。请直接输出句子，不要添加任何说明或编号。"


    print(f"\n正在为单词 '{word}' 生成 {num_sentences} 个句子...")


def get_qwen_stream_response(user_prompt,system_prompt):
    response = client.chat.completions.create(
        model="qwen-max",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        stream=True
    )
    for chunk in response:
        yield chunk.choices[0].delta.content



response = get_qwen_stream_response(user_prompt=f"""请用印地语生成 10 个不同的完整句子，每个句子都必须包含单词 "विड्रॉल नहीं हो पाया"。
    要求：
    1. 每个句子都必须是自然、流畅的印地语句子
    2. 每个句子都必须包含单词 "विड्रॉल नहीं हो पाया"
    3. 句子之间要有不同的语境和表达方式
    4. 直接输出句子，每行一个句子，不要编号，不要其他说明文字
    
    生成的句子：""",
                                    system_prompt="你是一个印地语专家，擅长生成自然流畅的印地语句子。请直接输出句子，不要添加任何说明或编号。")
for chunk in response:
    print(chunk, end="")