'''
三种方式调用api
deepseek
ali百炼
本地
'''

import os
from openai import OpenAI
from config.load_key import load_api_keys, get_api_key

system_prompt = '''
你是精通葡萄牙语（欧洲变体）的客户服务专家，专门处理用户提现问题。你的任务是根据用户的指示生成最真实、最口语化的用户提问。

生成规则：
1. 使用欧洲葡萄牙语，术语准确自然
2. 问题必须口语化，反映真实用户情绪（担忧、困惑、不耐烦等）
3. 必须包含"criptomoeda"或"moeda digital"字眼
4. 问题要有层次：基础流程类、问题故障类、费用限额类、安全确认类
5. 覆盖多种场景：钱包、银行账户、信用卡、不同币种等
6. 输出严格使用JSON格式，并且将用户的问题 ，原封不动的放在json的第一行中
'''
user_prompt = '''
生成10个和 Horário de chegada do depósito 相关的问题
'''


def deep_call(system_prompt: str = "", user_prompt: str = "告诉我你是什么模型"):
    deepseek_key = get_api_key("DEEPSEEK")
    ds_api_base = "https://api.deepseek.com"
    client = OpenAI(
        api_key=deepseek_key,
        base_url=ds_api_base)
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False
    )
    print(response.choices[0].message.content)
    return (response.choices[0].message.content)


def dashscope_call(system_prompt: str = "", user_prompt: str = "告诉我你是什么模型"):
    dashscope_key = os.environ.get('DASHSCOPE_API_KEY')
    ali_api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    client = OpenAI(
        api_key=dashscope_key,
        base_url=ali_api_base)

    response = client.chat.completions.create(
        # model="deepseek-chat",
        model="qwen-plus",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        stream=False
    )

    print(response.choices[0].message.content)
    return (response.choices[0].message.content)


def local_call(local_key="lm-studio", base_url="http://192.168.3.191:1234/v1",
               system_prompt: str = "", user_prompt: str = "告诉我你是什么模型"):
    # 本地不需要key，只需要一个url,但是用的时候需要注意关掉vpn
    # win主机 base_url="http://192.168.3.191:1234/v1"
    # mac主机 base_url="http://192.168.4.104:1234/v1"
    model = "deepseek-r1-distill-qwen-1.5b"
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]
    # messages = [
    #     # {"role": "system", "content": "You are a helpful assistant"},
    #     {"role": "user", "content": "你是谁"}
    # ]

    client = OpenAI(base_url=base_url, api_key=local_key)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False
    )
    print(response.choices[0].message.content)
    return (response.choices[0].message.content)


if __name__ == '__main__':
    local_call()
