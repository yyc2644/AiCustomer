import os
import re
import pandas as pd

# ========= 配置区 =========
EXCEL_DIR = r"./excel"       # 存放excel的目录
OUTPUT_FILE = "output.jsonl" # 输出jsonl
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


def excel_to_jsonl(excel_path, output_path):
    print(f"读取文件: {excel_path}")

    # 只读第一个 sheet
    df = pd.read_excel(excel_path)

    print(f"共 {len(df)} 行")

    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            item = {}

            for col in df.columns:
                value = row[col]
                if pd.notna(value):
                    item[col] = str(value).strip()

            if item:
                f.write(pd.Series(item).to_json(force_ascii=False) + "\n")

    print(f"jsonl 生成完成: {output_path}")


if __name__ == "__main__":
    latest_excel = find_latest_excel(EXCEL_DIR)
    excel_to_jsonl(latest_excel, OUTPUT_FILE)
