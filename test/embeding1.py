# 导入依赖
from llama_index.embeddings.dashscope import DashScopeEmbedding,DashScopeTextEmbeddingModels
from llama_index.core import SimpleDirectoryReader,VectorStoreIndex
from llama_index.llms.openai_like import OpenAILike
import json
import time
import requests
# 这两行代码是用于消除 WARNING 警告信息，避免干扰阅读学习，生产环境中建议根据需要来设置日志级别
# import logging
# logging.basicConfig(level=logging.ERROR)


from config.load_key import load_key
import os

load_key()
# 生产环境中请勿将 API Key 输出到日志中，避免泄露
print(f'''你配置的 API Key 是：{os.environ["DASHSCOPE_API_KEY"][:5]+"*"*5}''')

print("正在解析文件...")
# LlamaIndex提供了SimpleDirectoryReader方法，可以直接将指定文件夹中的文件加载为document对象，对应着解析过程
documents = SimpleDirectoryReader('../BehaviorTree').load_data()

print("正在创建索引...")
# from_documents方法包含切片与建立索引步骤
index = VectorStoreIndex.from_documents(
    documents,
    # 指定embedding 模型
    embed_model=DashScopeEmbedding(
        # 你也可以使用阿里云提供的其它embedding模型：https://help.aliyun.com/zh/model-studio/getting-started/models#3383780daf8hw
        model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2
    ))
print("正在创建提问引擎...")

index.storage_context.persist("../knowledge_base/tree_test")
print("索引文件保存到了../knowledge_base/tree_test")

# index = rag.load_index(persist_path='./knowledge_base/test')
# query_engine = rag.create_query_engine(index=index)

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "deepseek-r1:1.5b"
# def call_ollama(prompt: str) -> None:
#     payload = {
#         "model": MODEL_NAME,
#         "prompt": prompt,
#         "stream": True,
#         # 如需调参，可加 options={"temperature":0.7, "top_p":0.9}
#     }
#
#     start = time.perf_counter()
#     response = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=120)
#     response.raise_for_status()
#
#     print("模型输出：", end="", flush=True)
#     final_stats = {}
#     for line in response.iter_lines():
#         if not line:
#             continue
#         data = json.loads(line)
#         if chunk := data.get("response"):
#             print(chunk, end="", flush=True)
#         if data.get("done"):
#             final_stats = data  # Ollama 在最后一个块里附带 token & duration
#             break
#
#     print()  # 换行
#     duration = time.perf_counter() - start
#     prompt_tokens = final_stats.get("prompt_eval_count", 0)
#     completion_tokens = final_stats.get("eval_count", 0)
#     total_tokens = prompt_tokens + completion_tokens
#
#     print(f"耗时：{duration:.2f}s "
#           f"(prompt_eval={final_stats.get('prompt_eval_duration', 0)/1e9:.2f}s, "
#           f"eval={final_stats.get('eval_duration', 0)/1e9:.2f}s)")
#     print(f"token 消耗：prompt {prompt_tokens}, "
#           f"completion {completion_tokens}, total {total_tokens}")

# if __name__ == "__main__":
#     question = "你是谁"
#     call_ollama(question)