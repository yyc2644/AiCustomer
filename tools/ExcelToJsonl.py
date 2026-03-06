import os
import re
import pandas as pd
import os
from datetime import datetime

# ========= 配置区 =========
# 存放excel的目录
EXCEL_DIR  =  r"C:\Users\ZhanYi\PycharmProjects\AiCustomer\BehaviorTree\base"
# 输出jsonl
OUTPUT_DIR = r"C:\Users\ZhanYi\PycharmProjects\AiCustomer\BehaviorTree\base\jsonl"
# =========================


def find_latest_excel(folder):
    """
    找到类似 20260112_162606_config_export.xlsx 的最新文件
    """
    pattern = re.compile(r"(\d{8}_\d{6})_config_export\.xlsx")

    candidates = []
    for f in os.listdir(folder):
        match = pattern.match(f)
        if match:
            candidates.append((match.group(1), f))

    if not candidates:
        raise FileNotFoundError("未找到任何 *_config_export.xlsx 文件")

    # 按时间戳排序
    candidates.sort(key=lambda x: x[0], reverse=True)

    return os.path.join(folder, candidates[0][1])


def excel_to_jsonl(excel_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    excel_file = os.path.basename(excel_path)

    # 从 20260112_162606_config_export.xlsx 提取 20260112_162606
    m = re.search(r"(\d{8}_\d{6})", excel_file)
    if not m:
        raise ValueError("无法从 Excel 文件名中解析时间戳")

    version = m.group(1)

    final_path = os.path.join(output_dir, f"{version}.jsonl")

    print(f"读取文件: {excel_path}")
    print(f"输出文件: {final_path}")

    df = pd.read_excel(excel_path)
    print(f"共 {len(df)} 行")

    with open(final_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            item = {}

            for col in df.columns:
                val = row[col]
                if pd.notna(val):
                    item[col] = str(val).strip()

            if item:
                f.write(pd.Series(item).to_json(force_ascii=False) + "\n")

    print("jsonl 生成完成")

if __name__ == "__main__":
    latest_excel = find_latest_excel(EXCEL_DIR)
    excel_to_jsonl(latest_excel, OUTPUT_DIR)
