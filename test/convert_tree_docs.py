#!/usr/bin/env python
"""将 BehaviorTree 目录下的 XLS 转换为 JSONL。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
TREE_DOCS_DIR = PROJECT_ROOT / "BehaviorTree"
DEFAULT_OUTPUT_DIR = TREE_DOCS_DIR / "jsonl"

# 兼容不同命名方式的列名
COLUMN_CANDIDATES = {
    "category_id": ["问题分类ID", "分类ID", "category_id", "�������ID"],
    "category": ["问题分类", "分类", "category", "�������"],
    "question_id": [
        "问题ID",
        "问句ID",
        "分类子问题ID",
        "question_id",
        "����������ID",
    ],
    "question": ["问题", "问句", "分类子问题", "question", "����������"],
    "answer_id": [
        "答案ID",
        "回复ID",
        "子问题应答ID",
        "answer_id",
        "������Ӧ��ID",
    ],
    "answer": ["答案", "回复", "子问题应答", "answer", "������Ӧ��"],
    "context": ["上下文", "context"],
}


def pick(row: dict, candidates: Iterable[str]) -> str:
    """从候选列名中选择第一个非空值。"""
    for key in candidates:
        if key in row:
            value = row[key]
            if pd.notna(value):
                text = str(value).strip()
                if text and text.lower() != "nan":
                    return text
    return ""


def convert_file(xls_path: Path) -> List[dict]:
    df = pd.read_excel(xls_path, engine="openpyxl").fillna("")
    records = []
    for idx, row in enumerate(df.to_dict(orient="records"), start=1):
        record = {
            "id": pick(row, COLUMN_CANDIDATES["question_id"])
            or pick(row, COLUMN_CANDIDATES["answer_id"]),
            "问题分类": pick(row, COLUMN_CANDIDATES["category"]),
            "question": pick(row, COLUMN_CANDIDATES["question"]),
            "context": pick(row, COLUMN_CANDIDATES["context"]),
            "answer": pick(row, COLUMN_CANDIDATES["answer"]),
        }
        if not record["id"]:
            record["id"] = f"{xls_path.stem}_{idx}"
        if not record["context"]:
            record["context"] = record["问题分类"]
        records.append(record)
    return records


def write_jsonl(records: List[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for row in records:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert BehaviorTree XLS files to JSONL.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=TREE_DOCS_DIR,
        help="包含 XLS 的目录，默认 BehaviorTree。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="输出 JSONL 目录，默认 BehaviorTree/jsonl。",
    )
    args = parser.parse_args()

    xls_files = sorted(args.input_dir.glob("*.xls"))
    if not xls_files:
        raise FileNotFoundError(f"未在 {args.input_dir} 找到任何 .xls 文件")

    for xls_path in xls_files:
        records = convert_file(xls_path)
        output_path = args.output_dir / f"{xls_path.stem}.jsonl"
        write_jsonl(records, output_path)
        print(f"已生成 {output_path}，共 {len(records)} 条")


if __name__ == "__main__":
    main()

