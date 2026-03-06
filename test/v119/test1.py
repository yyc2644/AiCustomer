# 加载百炼的 API Key 用于调用通义千问大模型
import os
from config.load_key import load_key
from tools.similarity import similarity

load_key()
print(f'''你配置的 API Key 是：{os.environ["DASHSCOPE_API_KEY"][:5] + "*" * 5}''')

from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"), base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")


def get_response(user_prompt, system_prompt,temperature,top_p=0.8):
    response = client.chat.completions.create(
        model="qwen-max",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        stream=True,
        temperature=temperature,
        top_p=top_p,
    )
    for chunk in response:
        yield chunk.choices[0].delta.content

user_prompt = "डिपॉज़िट नहीं हो पा रहा है"
system_prompt = "你是一个客服用户。 请基于用户输入的内容，生成 50 条【语义尽可能相似】的短句，： 要求： 1. 不引入新条件 2. 不引入时间、状态、对象变化 3. 口语化 4. 语言和用户输入的语言保持一致 5.每次生成的答案需要不一致"

requests = get_response(user_prompt, system_prompt,temperature=1.8,)
for chunk in requests:
    # print(chunk,end="")
    print("1",chunk,"相似度：", similarity(user_prompt, chunk))