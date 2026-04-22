# AiCustomer 项目现状评估与可立即尝试清单

## 1. 结论速览

当前项目已经具备“测试框架骨架 + 关键模块代码 + 多类测试样例”的基础，结构完整度较高；
但运行环境与真实联调链路仍有断点，暂时属于“可评审、可局部运行、未完全打通”的阶段。

## 2. 当前搭建情况（已完成）

### 2.1 架构层面

- 已有分层结构：`config / data / core / lib / tests / runner`。
- 已有批量测试入口：`main.py -> runner/batch_test.py`。
- 已有核心能力代码：
  - 评估器：`core/evaluator.py`
  - 行为树解析：`core/tree_parser.py`
  - API封装：`lib/api_client.py`
  - 对话模拟器：`lib/bot_simulator.py`
- 已有多类型测试目录：
  - API：`tests/api/`
  - 对话：`tests/dialogue/`
  - NLU：`tests/nlu/`
  - 基础单测：`tests/test_core.py`、`tests/test_data.py`、`tests/test_lib.py`

### 2.2 数据与配置

- 已有测试语料与行为树样例：`data/corpus/`、`data/behavior_cases/`。
- 已有环境配置与系统参数：`config/env.yaml`、`config/systems.yaml`。
- 已有历史测试产物：`reports/logs/test.log`、`reports/summary.json`。

### 2.3 LLM/供应商接入

- 已新增 NVIDIA 验证脚本：`nvidia/demo.py`。
- 但 NVIDIA 目前仅在 demo 文件中使用，尚未接入 `lib/api_client.py` / `runner` / `tests` 主链路。

## 3. 主要优点

- 模块边界清晰，后续扩展成本低。
- 测试类型齐全（API/UI/对话/NLU），长期可沉淀成回归体系。
- `APIClient` 有重试与统一错误封装，工程化意识较好。
- `BotSimulator` 支持会话管理与多轮流程，便于模拟真实客服场景。
- 行为树与语料都已有样例文件，适合快速做端到端演练。

## 4. 主要问题与风险

### 4.1 运行环境尚未就绪（阻塞“立即全跑”）

实测当前环境中：

- `python3 main.py` 报错：`ModuleNotFoundError: No module named 'yaml'`。
- `./.venv/bin/python -m pytest ...` 报错：`No module named pytest`。

说明依赖尚未在当前解释器完整安装。

### 4.2 配置与真实接口可能不匹配

- `config/env.yaml` 中 `test.api.base_url` 为：
  `https://test-api.zhizhi168.com/#/home`
- 该地址包含前端路由片段 `#/home`，用于 API base_url 风险较高（应更像纯 API 域名/根路径）。

### 4.3 README 与现实存在偏差

- README 声称可以直接按 `pytest` 全量运行，但当前本机环境并未满足。
- README 的某些示例路径与实际实现细节有出入（例如运行入口/依赖前置条件未强调）。

### 4.4 部分代码处于“草稿态/占位态”

- `runner/single_test.py` 中 `OpenAIAdapter` 未定义即被调用。
- `tests` 内很多用例依赖 mock，可验证结构逻辑，但不等于真实后端已打通。

### 4.5 NVIDIA Key 尚未进入主流程

- 你现在的 NVIDIA key 仅用于 `nvidia/demo.py`。
- 主框架（API测试、批量评估、回归）并未直接消费该 key。

## 5. 哪些功能可以“立即尝试”

在当前代码状态下，按成功概率从高到低：

### A. 立即可做（不依赖真实后端）

1. 配置加载检查
- 验证 `config/config_loader.py` 能读取 `env.yaml + systems.yaml`。

2. 数据加载链路
- 跑 `data/data_loader.py` 的 CSV/JSONL/YAML 读取。

3. 纯本地评估逻辑
- 直接调用 `Evaluator.evaluate / fuzzy_match / evaluate_intent / evaluate_slots`。

4. BotSimulator 无API模式
- 不传 `api_client`，走 `_mock_response` 完成单轮/多轮流程验证。

5. NVIDIA 单文件连通测试
- 用 `nvidia/demo.py` 验证 key 是否有效、模型调用是否成功。

### B. 条件可做（需补环境或配置）

1. pytest 单元测试集
- `tests/test_core.py`、`tests/test_data.py`、`tests/test_lib.py`
- 前提：安装 pytest 和基础依赖。

2. API 集成链路
- 需修正/确认 `test.api.base_url` 和鉴权参数是否正确。

3. 行为树路径验证
- 依赖行为树数据完整性与期望节点命名一致。

### C. 暂不建议立即做

1. 全量 `pytest` 一次性跑完
- 当前依赖和外部环境未统一，失败噪音会较大。

2. 把 NVIDIA 直接替换到全部主流程
- 建议先增加“provider 适配层”再并入，避免破坏已有 API 测试结构。

## 6. 建议的最小打通顺序（你现在就能开始）

1. 先修环境（确保同一个 Python 解释器具备 `PyYAML + pytest`）。
2. 先跑本地无外部依赖链路：
   - `config` 加载
   - `data` 加载
   - `evaluator` 基础评估
   - `bot_simulator` mock 会话
3. 单独跑 `nvidia/demo.py` 验证 key 可用。
4. 再推进 API 真联调（base_url、token、接口路径逐个确认）。
5. 最后再做分组回归（core -> nlu -> dialogue -> api）。

## 7. 总体评价

- 完整度：`7/10`（框架齐全）
- 可运行度：`4/10`（环境与联调未完全就绪）
- 可扩展性：`8/10`（分层和测试组织较好）
- 当前阶段建议：先“打通最小闭环”，再追求“全量回归”。

