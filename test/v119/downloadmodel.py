# 直接下载正确的模型，忽略本地有问题的版本
from sentence_transformers import SentenceTransformer
import os

# 设置环境变量跳过警告
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

# 下载并使用标准模型
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# 保存到新位置避免冲突
model.save("./model")

print("模型加载完成，已保存到 model 目录")