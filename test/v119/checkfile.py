import os
os.environ['AWQ_NO_KERNELS'] = '1'  # 禁用AWQ内核警告
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. 加载模型（你本地已有就不会重新下载）
# model = SentenceTransformer(
#     "text-embedding-trotr-paraphrase-multilingual-minilm-l12-v2"
# )

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# 2. 读取 Excel（以“提现问题”为例）
file_path = "全球站客服语料树.xlsx"
df = pd.read_excel(file_path, sheet_name="提现问题")

# 3. 指定语言列
BASE_COL = "问题子类"
LANG_COLS = ["英语", "印地语", "巴西葡语"]

# 4. 只保留有效数据
df = df.dropna(subset=[BASE_COL])

results = []

for idx, row in df.iterrows():
    base_text = str(row[BASE_COL]).strip()
    if not base_text:
        continue

    base_emb = model.encode(base_text)

    for lang in LANG_COLS:
        if pd.isna(row.get(lang)):
            continue

        target_text = str(row[lang]).strip()
        target_emb = model.encode(target_text)

        score = cosine_similarity(
            [base_emb], [target_emb]
        )[0][0]

        results.append({
            "行号": idx + 2,
            "语言": lang,
            "相似度": round(float(score), 4),
            "是否通过": score >= 0.85,
            "中文": base_text,
            "翻译文本": target_text
        })

# 5. 导出检查结果
result_df = pd.DataFrame(results)
result_df.to_excel("翻译一致性检查结果.xlsx", index=False)

print("✅ 翻译一致性检查完成")
