# AiCustomer - 智能客服测试框架

一个完整的智能客服自动化测试框架，支持后台API测试、页面UI测试、行为树对话测试和NLU语料评估。

## 项目架构

```
AiCustomer/
├── config/                 # 配置层
│   ├── env.yaml           # 环境配置（dev/test/prod）
│   ├── logging.conf       # 日志配置
│   ├── systems.yaml       # 系统配置（阈值、模型参数）
│   ├── config_loader.py  # 配置加载器
│   └── load_key.py       # 密钥加载模块
│
├── data/                  # 数据层
│   ├── corpus/           # 语料测试集（CSV/JSONL/XLSX）
│   ├── behavior_cases/    # 行为树测试用例（YAML/JSON）
│   ├── ui/               # UI元素定位器（YAML/JSON）
│   └── data_loader.py    # 数据加载器
│
├── core/                  # 核心层
│   ├── evaluator.py       # 评估器（意图识别、答案匹配、语义相似度）
│   ├── tree_parser.py     # 行为树解析器
│   ├── assert_helper.py   # 断言辅助工具
│   └── utils.py          # 工具函数
│
├── lib/                   # 基础库层
│   ├── api_client.py      # API客户端封装
│   ├── bot_simulator.py   # 机器人模拟器
│   ├── page_objects/      # Page Object模式
│   │   └── base.py       # 页面对象基类
│   └── db_manager.py     # 数据库管理工具
│
├── tests/                 # 测试用例层
│   ├── api/              # API接口测试
│   ├── ui/                # 页面UI测试
│   ├── dialogue/          # 行为树对话测试
│   ├── nlu/              # 语料知识库测试
│   ├── test_core.py       # 核心模块单元测试
│   ├── test_data.py       # 数据模块单元测试
│   └── test_lib.py        # 库模块单元测试
│
├── pytest.ini             # Pytest配置文件
├── conftest.py            # Pytest共享Fixtures
└── requirements.txt        # 项目依赖
```

## 功能特性

### 1. 后台API测试 (`tests/api/`)
- 知识库管理（增删改查）
- 会话管理（创建、查询、转人工）
- 消息发送与历史
- 统计数据获取

### 2. 页面UI测试 (`tests/ui/`)
- Page Object Model模式
- 聊天窗口元素定位
- 管理后台元素操作
- 截图功能

### 3. 行为树对话测试 (`tests/dialogue/`)
- 单轮/多轮对话模拟
- 行为树路径验证
- 节点跳转检查
- 槽位填充验证

### 4. NLU语料测试 (`tests/nlu/`)
- 意图识别准确率评估
- 语义相似度计算
- 答案匹配度测试
- 批量语料回归测试

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行测试

```bash
# 运行所有测试
pytest

# 按模块运行
pytest tests/api/ -v
pytest tests/dialogue/ -v
pytest tests/nlu/ -v

# 按标记运行
pytest -m api           # API测试
pytest -m dialogue       # 对话测试
pytest -m nlu           # NLU测试
pytest -m smoke         # 冒烟测试
pytest -m regression    # 回归测试

# 运行单元测试
pytest tests/test_core.py -v
pytest tests/test_data.py -v
pytest tests/test_lib.py -v

# 生成HTML报告
pytest --html=reports/report.html
```

## 配置说明

### 环境配置 (config/env.yaml)

```yaml
default: test
environments:
  dev:
    api:
      base_url: "http://localhost:8000"
  test:
    api:
      base_url: "https://test-api.example.com"
  prod:
    api:
      base_url: "https://api.example.com"
```

### 系统配置 (config/systems.yaml)

```yaml
evaluation:
  intent:
    top1_threshold: 0.85    # 意图准确率阈值
  answer:
    similarity_threshold: 0.80  # 答案相似度阈值

model:
  intent_model:
    type: "local"
    model_path: "test/v119/model"
```

## 测试数据格式

### 语料格式 (data/corpus/test_corpus.csv)

```csv
query,expected_intent,expected_answer,language,category
"我想退货",refund,"您好，请问有什么可以帮助？",zh-CN,售后服务
```

### 行为树用例格式 (data/behavior_cases/refund_flow.yaml)

```yaml
test_case:
  name: "退货流程测试"
  steps:
    - step: 1
      user_input: "我想退货"
      expected_intent: "refund"
      expected_node: "refund_start"
```

## 使用示例

### API客户端

```python
from lib.api_client import APIClient

client = APIClient(env="test")
client.login("admin", "password")

# 发送消息
response = client.send_message("我想退货")

# 知识库操作
client.add_knowledge(title="测试", content="答案")
```

### 机器人模拟器

```python
from lib.bot_simulator import BotSimulator

simulator = BotSimulator(api_client)

# 单轮对话
result = simulator.chat("我想退货")

# 多轮对话
result = simulator.chat_flow([
    "我想退货",
    "订单号123456"
])
```

### 评估器

```python
from core.evaluator import Evaluator

evaluator = Evaluator()

# 意图识别评估
match, score = evaluator.evaluate_intent("refund", "refund")

# 语义相似度
similarity = evaluator.semantic_similarity("我想退货", "我要退货")

# 批量评估
results = evaluator.batch_evaluate(test_cases, adapter)
```

### 行为树解析器

```python
from core.tree_parser import TreeParser

parser = TreeParser()
parser.load_tree("base", "en")
parser.validate_path(test_case)
```

## 标记说明

| 标记 | 说明 |
|------|------|
| `@pytest.mark.api` | API接口测试 |
| `@pytest.mark.ui` | 页面UI测试 |
| `@pytest.mark.dialogue` | 行为树对话测试 |
| `@pytest.mark.nlu` | 语料知识库测试 |
| `@pytest.mark.smoke` | 冒烟测试 |
| `@pytest.mark.regression` | 回归测试 |
| `@pytest.mark.slow` | 慢速测试 |

## 项目依赖

```
pytest>=7.0.0
requests>=2.28.0
pyyaml>=6.0
pandas>=1.5.0
pymysql>=1.0.0
sentence-transformers>=2.2.0
```

## 目录说明

| 目录 | 用途 |
|------|------|
| `config/` | 配置文件 |
| `data/` | 测试数据 |
| `core/` | 核心逻辑 |
| `lib/` | 基础库 |
| `tests/` | 测试用例 |
| `reports/` | 测试报告 |

## 贡献指南

欢迎提交Issue和Pull Request！

## 许可证

MIT License
