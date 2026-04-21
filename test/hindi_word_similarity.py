'''
输入一个印地语单词，通过大模型生成10个回答（句子），
并计算单词和句子之间的语义相似度
使用本地的 text-embedding-trotr-paraphrase-multilingual-minilm-l12-v2 模型
'''

import time
import random
import os
from typing import List, Dict, Optional, Generator
from pathlib import Path
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 加载 API Key 配置
from config.load_key import load_key
load_key()

# 千问模型配置
QWEN_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_API_KEY = os.getenv("DASHSCOPE_API_KEY")
QWEN_MODEL_NAME = "qwen-max"

# 创建 OpenAI 客户端（兼容千问 API）
client = OpenAI(base_url=QWEN_API_BASE, api_key=QWEN_API_KEY)

# Embedding 模型名称或路径
# 可以是 HuggingFace 模型名称或本地路径
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
# 如果模型在本地，可以设置为本地路径，例如：
# EMBEDDING_MODEL_NAME = "./models/text-embedding-trotr-paraphrase-multilingual-minilm-l12-v2"


def load_embedding_model(model_path: str = None):
    """
    加载本地的 embedding 模型
    
    Args:
        model_path: 模型路径或名称，如果为 None 则使用默认配置
        
    Returns:
        SentenceTransformer: embedding 模型对象
    """
    model_name = model_path or EMBEDDING_MODEL_NAME
    print(f"正在加载 embedding 模型: {model_name}")
    
    try:
        # 尝试从本地路径加载
        if "/" in model_name or "\\" in model_name or Path(model_name).exists():
            print("从本地路径加载模型...")
            model = SentenceTransformer(model_name)
        else:
            # 从 HuggingFace 加载（首次运行会下载）
            print("从 HuggingFace 加载模型（首次运行可能需要下载，请耐心等待...）")
            model = SentenceTransformer(model_name)
        
        print("✓ Embedding 模型加载完成")
        return model
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("尝试使用默认模型名称...")
        # 如果失败，尝试使用完整的 HuggingFace 路径
        try:
            model = SentenceTransformer(f"sentence-transformers/{EMBEDDING_MODEL_NAME}")
            print("✓ Embedding 模型加载完成")
            return model
        except Exception as e2:
            print(f"加载默认模型也失败: {e2}")
            raise


def get_qwen_stream_response(user_prompt: str, system_prompt: str) -> Generator[str, None, None]:
    """
    获取千问模型的流式响应
    
    Args:
        user_prompt: 用户提示词
        system_prompt: 系统提示词
        
    Yields:
        str: 流式返回的内容片段
    """
    response = client.chat.completions.create(
        model=QWEN_MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        stream=True
    )
    
    for chunk in response:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def generate_sentences(word: str, num_sentences: int = 10) -> List[str]:
    """
    通过千问大模型生成包含指定单词的句子（使用流式响应）
    
    Args:
        word: 印地语单词
        num_sentences: 要生成的句子数量（默认10个）
        
    Returns:
        生成的句子列表
    """
    print(f"\n正在为单词 '{word}' 生成 {num_sentences} 个句子...")
    
    # 构建提示词
    user_prompt = f"""请用印地语生成 {num_sentences} 个不同的完整句子，每个句子都必须包含单词 "{word}"。
要求：
1. 每个句子都必须是自然、流畅的印地语句子
2. 每个句子都必须包含单词 "{word}"
3. 句子之间要有不同的语境和表达方式
4. 直接输出句子，每行一个句子，不要编号，不要其他说明文字

生成的句子："""
    
    system_prompt = "你是一个印地语专家，擅长生成自然流畅的印地语句子。请直接输出句子，不要添加任何说明或编号。"
    
    generated_sentences = []
    attempts = 0
    max_attempts = 3
    
    while len(generated_sentences) < num_sentences and attempts < max_attempts:
        attempts += 1
        print(f"\n--- 生成尝试 {attempts}/{max_attempts} ---")
        print("正在生成（流式输出）: ", end="", flush=True)
        
        try:
            start = time.perf_counter()
            content = ""
            
            # 使用流式响应
            for chunk in get_qwen_stream_response(user_prompt, system_prompt):
                if chunk:
                    content += chunk
                    print(chunk, end="", flush=True)
            
            print()  # 换行
            duration = time.perf_counter() - start
            
            content = content.strip()
            
            # 解析生成的句子（按行分割）
            sentences = [s.strip() for s in content.split('\n') if s.strip()]
            
            # 过滤掉不包含目标单词的句子
            valid_sentences = [s for s in sentences if word in s]
            
            # 去重
            seen = set()
            for s in valid_sentences:
                if s not in seen:
                    seen.add(s)
                    generated_sentences.append(s)
                    if len(generated_sentences) >= num_sentences:
                        break
            
            print(f"本次生成了 {len(valid_sentences)} 个有效句子")
            print(f"耗时: {duration:.2f}s")
            
        except Exception as e:
            print(f"\n生成过程中出现错误: {e}")
            continue
    
    if len(generated_sentences) < num_sentences:
        print(f"\n⚠️  警告: 只生成了 {len(generated_sentences)} 个句子，少于要求的 {num_sentences} 个")
    
    return generated_sentences[:num_sentences]


def calculate_similarity(word: str, sentences: List[str], embedding_model) -> List[Dict]:
    """
    计算单词和句子之间的语义相似度
    
    Args:
        word: 印地语单词
        sentences: 生成的句子列表
        embedding_model: SentenceTransformer 模型对象
        
    Returns:
        包含句子和相似度的字典列表
    """
    print(f"\n正在计算单词 '{word}' 与 {len(sentences)} 个句子的语义相似度...")
    
    # 获取单词的 embedding
    word_embedding = embedding_model.encode([word], convert_to_numpy=True)
    
    # 获取所有句子的 embeddings
    sentence_embeddings = embedding_model.encode(sentences, convert_to_numpy=True)
    
    # 计算余弦相似度
    similarities = cosine_similarity(word_embedding, sentence_embeddings)[0]
    
    # 构建结果列表
    results = []
    for i, (sentence, similarity) in enumerate(zip(sentences, similarities), 1):
        results.append({
            "序号": i,
            "单词": word,
            "句子": sentence,
            "相似度": float(similarity)
        })
    
    # 按相似度降序排序
    results.sort(key=lambda x: x["相似度"], reverse=True)
    
    # 重新编号
    for i, result in enumerate(results, 1):
        result["序号"] = i
    
    return results


def display_results(results: List[Dict]):
    """
    显示结果
    
    Args:
        results: 包含句子和相似度的结果列表
    """
    print("\n" + "="*80)
    print("生成结果和相似度排序")
    print("="*80)
    
    for result in results:
        print(f"\n序号 {result['序号']}:")
        print(f"  单词: {result['单词']}")
        print(f"  句子: {result['句子']}")
        print(f"  相似度: {result['相似度']:.4f}")
    
    print("\n" + "="*80)
    print("统计信息:")
    print(f"  总句子数: {len(results)}")
    if results:
        similarities = [r["相似度"] for r in results]
        print(f"  最高相似度: {max(similarities):.4f}")
        print(f"  最低相似度: {min(similarities):.4f}")
        print(f"  平均相似度: {sum(similarities)/len(similarities):.4f}")
    print("="*80)


def main():
    """
    主函数
    """
    print("="*80)
    print("印地语单词语义相似度分析工具")
    print("="*80)
    
    # 可选：允许用户指定模型路径
    print("\n提示: 如果模型在本地，请输入模型路径（直接回车使用默认模型）:")
    custom_model_path = input().strip()
    model_path = custom_model_path if custom_model_path else None
    
    # 加载 embedding 模型
    embedding_model = load_embedding_model(model_path)
    
    # 获取用户输入
    print("\n请输入一个印地语单词:")
    word = input().strip()
    
    if not word:
        print("错误: 单词不能为空")
        return
    
    # 生成句子
    sentences = generate_sentences(word, num_sentences=10)
    
    if not sentences:
        print("错误: 未能生成任何句子")
        return
    
    print(f"\n✓ 成功生成 {len(sentences)} 个句子")
    
    # 计算相似度
    results = calculate_similarity(word, sentences, embedding_model)
    
    # 显示结果
    display_results(results)
    
    # 可选：保存结果到文件
    save_option = input("\n是否保存结果到文件? (y/n): ").strip().lower()
    if save_option == 'y':
        import json
        from pathlib import Path
        
        output_file = f"hindi_similarity_{word}_{int(time.time())}.json"
        output_path = Path(output_file)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 结果已保存到: {output_path}")


if __name__ == "__main__":
    main()

