import pandas as pd
import json
from pathlib import Path

# ================= 基础配置 =================
EXCEL_FILE = "../test/v119/全球站客服语料树.xlsx"
SHEET_NAME = "充值问题"
QUESTION_TYPE = "Withdraw"  # 可选，不需要可删

LANG_CONFIG = {
    "en": {
        "question_col": "英语问题",
        "answer_col": "英语回答"
    },
    "hi": {
        "question_col": "印地语问题",
        "answer_col": "印地语回答"
    },
    "pt_br": {
        "question_col": "巴西葡语问题",
        "answer_col": "巴西葡语回答"
    }
}

OUTPUT_DIR  =  r"C:\Users\ZhanYi\PycharmProjects\AiCustomer\BehaviorTree\base"

# OUTPUT_DIR = Path("../test/v119/jsonl_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ================= 读取 Excel =================
df = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME)

print("📊 Excel 列名如下：")
print(list(df.columns))

# ================= 生成 jsonl =================
for lang, cfg in LANG_CONFIG.items():
    q_col = cfg["question_col"]
    a_col = cfg["answer_col"]

    # 列存在性校验
    if q_col not in df.columns or a_col not in df.columns:
        print(f"❌ {lang} 跳过：列名不存在 ({q_col}, {a_col})")
        continue

    output_file = OUTPUT_DIR / f"withdraw_{lang}.jsonl"
    record_id = 1

    with open(output_file, "w", encoding="utf-8") as f:
        for idx, row in df.iterrows():
            question = str(row[q_col]).strip()
            answer = str(row[a_col]).strip()

            if question in ("", "nan") or answer in ("", "nan"):
                continue

            record = {
                "id": f"record_{record_id}",
                "问题分类": QUESTION_TYPE,
                "预期输入": question,
                "预期输出": answer
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            record_id += 1

    print(f"✅ 已生成：{output_file}")

print("🎉 所有语言处理完成")
