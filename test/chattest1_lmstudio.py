'''
第2次实验，直接通过本地运行deepseek-r1:1.5b，验证问题和回答
'''
from openai import OpenAI
import json
import time
import requests

client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")

def call_ollama(prompt: str) -> None:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": True,
        # 如需调参，可加 options={"temperature":0.7, "top_p":0.9}
    }

    start = time.perf_counter()
    response = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=120)
    response.raise_for_status()

    response = client.chat.completions.create(
        model="local-model",  # 这里不需要改名，LM Studio 会自动选择当前加载的模型
        messages=[
            {"role": "user", "content": "你好，你现在是哪个模型？"}
        ]
    )
    print("模型输出：", end="", flush=True)
    final_stats = {}
    for line in response.iter_lines():
        if not line:
            continue
        data = json.loads(line)
        if chunk := data.get("response"):
            print(chunk, end="", flush=True)
        if data.get("done"):
            final_stats = data  # Ollama 在最后一个块里附带 token & duration
            break

    print()  # 换行
    duration = time.perf_counter() - start
    prompt_tokens = final_stats.get("prompt_eval_count", 0)
    completion_tokens = final_stats.get("eval_count", 0)
    total_tokens = prompt_tokens + completion_tokens

    print(f"耗时：{duration:.2f}s "
          f"(prompt_eval={final_stats.get('prompt_eval_duration', 0) / 1e9:.2f}s, "
          f"eval={final_stats.get('eval_duration', 0) / 1e9:.2f}s)")
    print(f"token 消耗：prompt {prompt_tokens}, "
          f"completion {completion_tokens}, total {total_tokens}")

    print(response.choices[0].message.content)
# print(response.choices[0].message["content"])
if __name__ == "__main__":
    for i in range (100):
        question = "你是谁"
        call_ollama(question)

