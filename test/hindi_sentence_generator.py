'''
基于 chattest1.py，生成印地语完整句子
1. 从 Excel 文件读取"分类子问题"和"子问题应答"
2. 保存为 JSONL 格式
3. 读取 JSONL，生成完整句子
4. 将生成的句子和回答保存为 JSON 或 XLS
'''

import json
import time
import requests
import pandas as pd
import random
from pathlib import Path
from typing import List, Dict, Optional
from openai import OpenAI

# LM Studio 配置
LM_STUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
LM_STUDIO_API_KEY = "lm-studio"
MODEL_NAME = "local-model"  # LM Studio 会自动选择当前加载的模型

# 创建 OpenAI 客户端
client = OpenAI(base_url=LM_STUDIO_BASE_URL, api_key=LM_STUDIO_API_KEY)



# 兼容不同命名方式的列名
COLUMN_CANDIDATES = {
    "question": ["分类子问题", "问题", "问句", "question", "短句"],
    "answer": ["子问题应答", "答案", "回复", "answer", "应答"],
    "category": ["问题分类", "分类", "category"],
    "question_id": ["分类子问题ID", "问题ID", "问句ID", "question_id"],
    "answer_id": ["子问题应答ID", "答案ID", "回复ID", "answer_id"],
}


def pick_column(row: dict, candidates: List[str]) -> str:
    """从候选列名中选择第一个非空值"""
    for key in candidates:
        if key in row:
            value = row[key]
            if pd.notna(value):
                text = str(value).strip()
                if text and text.lower() != "nan":
                    return text
    return ""


def extract_from_excel(excel_path: str, output_jsonl: str) -> List[Dict]:
    """
    从 Excel 文件提取"分类子问题"和"子问题应答"，保存为 JSONL
    
    Args:
        excel_path: Excel 文件路径
        output_jsonl: 输出 JSONL 文件路径
        
    Returns:
        提取的数据列表
    """
    excel_path = Path(excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel 文件不存在: {excel_path}")
    
    print(f"正在读取 Excel 文件: {excel_path}")
    df = pd.read_excel(excel_path, engine="openpyxl").fillna("")
    
    records = []
    for idx, row in enumerate(df.to_dict(orient="records"), start=1):
        question = pick_column(row, COLUMN_CANDIDATES["question"])
        answer = pick_column(row, COLUMN_CANDIDATES["answer"])
        category = pick_column(row, COLUMN_CANDIDATES["category"])
        question_id = pick_column(row, COLUMN_CANDIDATES["question_id"])
        answer_id = pick_column(row, COLUMN_CANDIDATES["answer_id"])
        
        # 只处理有问题的记录
        if not question:
            continue
            
        record = {
            "id": question_id or answer_id or f"record_{idx}",
            "问题分类": category,
            "预期输入": question,  # 分类子问题
            "预期输出": answer,    # 子问题应答
        }
        records.append(record)
    
    # 保存为 JSONL
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"已提取 {len(records)} 条记录，保存到: {output_path}")
    return records


def load_from_jsonl(jsonl_path: str, max_records: Optional[int] = None) -> List[Dict]:
    """
    从 JSONL 文件加载数据
    
    Args:
        jsonl_path: JSONL 文件路径
        max_records: 最大加载记录数，None 表示加载全部
        
    Returns:
        数据记录列表
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL 文件不存在: {jsonl_path}")
    
    records = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_records and i >= max_records:
                break
            try:
                record = json.loads(line.strip())
                records.append(record)
            except json.JSONDecodeError:
                continue
    
    print(f"从 {jsonl_path} 加载了 {len(records)} 条记录")
    return records


def call_ollama(
    prompt: str,
    hide_thinking: bool = True,
    temperature: float = 0.7,
    top_p: float = 0.9,
    timeout_seconds: float = 60.0,
) -> Dict:
    """
    调用 LM Studio API 生成回复（使用 OpenAI 兼容接口）
    
    Args:
        prompt: 输入的提示词
        hide_thinking: 是否隐藏思考过程（默认 True，只显示最终结果）
        temperature: 温度参数，控制随机性（0.0-2.0），越高越随机
        top_p: 核采样参数，控制多样性（0.0-1.0）
        timeout_seconds: 单次生成最大耗时，超时将停止当前生成
        
    Returns:
        包含生成结果和统计信息的字典
    """
    start = time.perf_counter()
    
    if not hide_thinking:
        print("模型输出：", end="", flush=True)
    else:
        print("正在生成...", end="", flush=True)
    
    generated_text = ""
    usage_info = None
    in_thinking = False  # 标记是否在思考过程中
    thinking_buffer = ""  # 用于存储思考过程内容
    
    # 使用流式调用，添加随机性参数
    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[ {"role": "system", "content": "请直接回答，不要展示思考或推理步骤，也不要输出 <think> 标签。"},
                    {"role": "user", "content": prompt}],
        stream=True,
        temperature=temperature,
        top_p=top_p,
        timeout=timeout_seconds,
        stop=["<think>", "</think>", "思考：", "推理："],
    )
    
    # 处理流式响应
    for chunk in stream:
        # 超时保护
        if time.perf_counter() - start > timeout_seconds:
            print("\n⚠️  本次生成超时，已停止。")
            break
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            generated_text += content
            
            if hide_thinking:
                # 检测思考过程标记（deepseek-r1 使用 <think> 和 </think>）
                content_lower = content.lower()
                
                # 检测开始思考标记
                if "<think>" in content_lower or "推理：" in content or "思考：" in content:
                    in_thinking = True
                    thinking_buffer += content
                    continue
                
                # 检测结束思考标记
                if "</think>" in content_lower:
                    in_thinking = False
                    thinking_buffer = ""
                    continue
                
                # 如果在思考过程中，不输出
                if in_thinking:
                    thinking_buffer += content
                    continue
                
                # 不在思考过程中，正常输出
                print(content, end="", flush=True)
            else:
                # 显示所有内容
                print(content, end="", flush=True)
        
        # 尝试从最后一个 chunk 获取 usage 信息
        if hasattr(chunk, 'usage') and chunk.usage:
            usage_info = chunk.usage
    
    print()  # 换行
    
    # 后处理：移除思考过程标记（如果还有残留）
    if hide_thinking:
        import re
        # 移除 <think>...</think> 标签及其内容（不区分大小写）
        generated_text = re.sub(r'<think>.*?</think>', '', generated_text, flags=re.DOTALL | re.IGNORECASE)
        # 移除其他可能的思考过程标记
        generated_text = re.sub(r'推理：.*?思考结束', '', generated_text, flags=re.DOTALL)
        generated_text = re.sub(r'思考：.*?思考结束', '', generated_text, flags=re.DOTALL)
        # 清理多余空白
        generated_text = re.sub(r'\s+', ' ', generated_text).strip()
    
    duration = time.perf_counter() - start
    
    # 获取 token 统计信息
    if usage_info:
        prompt_tokens = usage_info.prompt_tokens or 0
        completion_tokens = usage_info.completion_tokens or 0
    else:
        # 如果流式响应没有 usage，使用估算值（印地语大约每个字符 0.3 tokens）
        prompt_tokens = int(len(prompt) * 0.3)
        completion_tokens = int(len(generated_text) * 0.3)
    
    total_tokens = prompt_tokens + completion_tokens

    print(f"耗时：{duration:.2f}s")
    print(f"token 消耗：prompt {prompt_tokens}, "
          f"completion {completion_tokens}, total {total_tokens}")
    
    return {
        "generated_text": generated_text,
        "duration": duration,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "stats": {}
    }


def build_prompt(short_sentence: str) -> str:
    """
    构建用于生成完整句子的提示词
    
    Args:
        short_sentence: 印地语短句
        
    Returns:
        完整的提示词
    """
    prompt = f"""请将以下印地语短句扩展为一个完整、自然、流畅的句子。保持原意不变，但使句子更加完整和自然。

短句：{short_sentence}

完整句子："""
    return prompt


def generate_complete_sentences(
    records: List[Dict],
    output_json: Optional[str] = None,
    output_xls: Optional[str] = None,
    save_results: bool = True,
    num_generations: int = 10
) -> List[Dict]:
    """
    为多个短句生成完整句子，每个短句生成多次
    
    Args:
        records: 数据记录列表，每个记录包含"预期输入"和"预期输出"
        output_json: 输出 JSON 文件路径（可选）
        output_xls: 输出 XLS 文件路径（可选）
        save_results: 是否保存结果到文件
        num_generations: 每个问题生成的次数（默认10次）
        
    Returns:
        结果列表，每个元素包含原始数据和生成的完整句子
    """
    results = []
    
    for idx, record in enumerate(records, 1):
        short_sentence = record.get("预期输入", "").strip()
        expected_output = record.get("预期输出", "").strip()
        
        if not short_sentence:
            print(f"\n跳过第 {idx} 条记录：缺少预期输入")
            continue
        
        print(f"\n{'='*60}")
        print(f"处理第 {idx}/{len(records)} 条记录")
        print(f"原始短句（预期输入）：{short_sentence}")
        if expected_output:
            print(f"预期输出：{expected_output}")
        print(f"将为该短句生成 {num_generations} 次不同的完整句子")
        print(f"{'='*60}")
        
        prompt = build_prompt(short_sentence)
        generated_sentences = []  # 存储所有生成的句子
        
        # 为每个问题生成多次
        for gen_idx in range(1, num_generations + 1):
            print(f"\n--- 生成第 {gen_idx}/{num_generations} 次 ---")
            
            # 使用不同的随机参数增加多样性
            # temperature: 0.7-1.2 之间随机，增加随机性
            # top_p: 0.85-0.95 之间随机，增加多样性
            temperature = random.uniform(0.7, 1.2)
            top_p = random.uniform(0.85, 0.95)
            
            result = call_ollama(prompt, hide_thinking=True, temperature=temperature, top_p=top_p)
            complete_sentence = result["generated_text"].strip()
            
            # 检查是否与之前的生成结果重复
            if complete_sentence in generated_sentences:
                print(f"⚠️  第 {gen_idx} 次生成与之前重复，将重新生成...")
                # 如果重复，使用更高的随机性重新生成
                temperature = random.uniform(1.0, 1.5)
                top_p = random.uniform(0.9, 0.99)
                result = call_ollama(prompt, hide_thinking=True, temperature=temperature, top_p=top_p)
                complete_sentence = result["generated_text"].strip()
            
            generated_sentences.append(complete_sentence)
            
            result_data = {
                "id": record.get("id", f"record_{idx}"),
                "生成序号": gen_idx,
                "问题分类": record.get("问题分类", ""),
                "原始短句": short_sentence,
                "生成的完整句子": complete_sentence,
                "预期输出": expected_output,
                "temperature": round(temperature, 2),
                "top_p": round(top_p, 2),
                "duration": result["duration"],
                "tokens": {
                    "prompt": result["prompt_tokens"],
                    "completion": result["completion_tokens"],
                    "total": result["total_tokens"]
                }
            }
            results.append(result_data)
            
            print(f"✓ 第 {gen_idx} 次生成完成：{complete_sentence}")
        
        print(f"\n该短句共生成 {len(generated_sentences)} 个不同的完整句子")
    
    # 保存结果
    if save_results:
        if output_json:
            output_json_path = Path(output_json)
            output_json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n结果已保存到 JSON: {output_json_path}")
        
        if output_xls:
            output_xls_path = Path(output_xls)
            output_xls_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 准备 DataFrame 数据
            df_data = []
            for r in results:
                df_data.append({
                    "ID": r["id"],
                    "生成序号": r.get("生成序号", 1),
                    "问题分类": r["问题分类"],
                    "原始短句": r["原始短句"],
                    "生成的完整句子": r["生成的完整句子"],
                    "预期输出": r["预期输出"],
                    "Temperature": r.get("temperature", 0.5),
                    "Top_p": r.get("top_p", 0.9),
                    "耗时(秒)": round(r["duration"], 2),
                    "Token总数": r["tokens"]["total"]
                })
            
            df = pd.DataFrame(df_data)
            df.to_excel(output_xls_path, index=False, engine='openpyxl')
            print(f"结果已保存到 XLS: {output_xls_path}")
    
    return results


if __name__ == "__main__":
    # 配置路径
    excel_file = "../tree_docs/行为树数据hi.xls"
    jsonl_file = "../tree_docs/jsonl/hindi_extracted.jsonl"
    output_json = "hindi_complete_sentences.json"
    output_xls = "hindi_complete_sentences.xls"
    
    # 步骤1: 从 Excel 提取数据并保存为 JSONL
    print("="*60)
    print("步骤1: 从 Excel 提取数据")
    print("="*60)
    try:
        records = extract_from_excel(excel_file, jsonl_file)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("尝试直接读取 JSONL 文件...")
        records = load_from_jsonl(jsonl_file)
    
    if not records:
        print("警告: 没有找到任何记录")
        exit(1)
    
    # 步骤2: 生成完整句子
    print("\n" + "="*60)
    print("步骤2: 生成完整句子")
    print("="*60)
    
    # 可以限制处理数量用于测试，设置为 None 处理全部
    max_records = None  # 例如: 5 表示只处理前5条
    
    if max_records:
        records = records[:max_records]
        print(f"限制处理前 {max_records} 条记录")
    
    results = generate_complete_sentences(
        records=records,
        output_json=output_json,
        output_xls=output_xls,
        save_results=True
    )
    
    # 打印总结
    print(f"\n{'='*60}")
    print("处理完成！")
    print(f"{'='*60}")
    print(f"总共处理了 {len(results)} 条记录")
    if results:
        total_time = sum(r["duration"] for r in results)
        total_tokens = sum(r["tokens"]["total"] for r in results)
        avg_time = total_time / len(results)
        print(f"总耗时：{total_time:.2f}s")
        print(f"平均耗时：{avg_time:.2f}s/条")
        print(f"总 token 消耗：{total_tokens}")
        print(f"平均 token 消耗：{total_tokens // len(results)}/条")
    print(f"{'='*60}")
