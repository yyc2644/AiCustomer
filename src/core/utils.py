"""
工具函数模块
"""

import yaml
import json
from pathlib import Path


def load_config(file_path=None):
    """加载YAML配置文件"""
    if file_path is None:
        file_path = Path(__file__).parent.parent / "config" / "systems.yaml"
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_testcases(file_name="test_corpus.jsonl"):
    """加载测试用例"""
    # 优先从data/corpus目录加载
    corpus_path = Path(__file__).parent.parent / "data" / "corpus" / file_name
    
    if corpus_path.exists():
        file_path = corpus_path
    else:
        # 使用默认的测试语料
        file_path = Path(__file__).parent.parent / "data" / "corpus" / "test_corpus.jsonl"
    
    # 根据文件扩展名选择加载方式
    if file_path.suffix == '.jsonl':
        results = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        return results
    elif file_path.suffix == '.csv':
        # CSV格式需要转换为字典列表
        import csv
        results = []
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append(row)
        return results
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
