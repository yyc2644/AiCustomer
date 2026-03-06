"""
Pytest配置文件
提供全局共享的fixtures和配置
"""

import os
import pytest
import logging
from pathlib import Path


# 项目根目录
ROOT_DIR = Path(__file__).parent


def pytest_configure(config):
    """Pytest配置钩子"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)8s] %(name)s - %(message)s'
    )
    
    # 注册标记
    config.addinivalue_line("markers", "api: 后台API接口测试")
    config.addinivalue_line("markers", "ui: 页面UI测试")
    config.addinivalue_line("markers", "dialogue: 行为树对话测试")
    config.addinivalue_line("markers", "nlu: 语料和知识库测试")
    config.addinivalue_line("markers", "slow: 慢速测试")
    config.addinivalue_line("markers", "smoke: 冒烟测试")
    config.addinivalue_line("markers", "regression: 回归测试")


@pytest.fixture(scope="session")
def config():
    """全局配置fixture"""
    from config.config_loader import get_config
    return get_config()


@pytest.fixture(scope="session")
def api_client(config):
    """API客户端fixture"""
    from lib.api_client import APIClient
    client = APIClient(env="test")
    yield client
    client.close()


@pytest.fixture(scope="session")
def db_manager(config):
    """数据库管理器fixture"""
    from lib.db_manager import DBManager
    db = DBManager(env="test")
    yield db
    db.close()


@pytest.fixture
def bot_simulator(api_client):
    """机器人模拟器fixture"""
    from lib.bot_simulator import BotSimulator
    simulator = BotSimulator(api_client)
    yield simulator
    simulator.clear_sessions()


@pytest.fixture
def evaluator(config):
    """评估器fixture"""
    from core.evaluator import Evaluator
    return Evaluator(config)


@pytest.fixture
def tree_parser():
    """行为树解析器fixture"""
    from core.tree_parser import TreeParser
    return TreeParser()


@pytest.fixture
def test_corpus():
    """测试语料fixture"""
    from data.data_loader import load_corpus_by_name
    return load_corpus_by_name("test_corpus")


@pytest.fixture
def behavior_cases():
    """行为树测试用例fixture"""
    from data.data_loader import load_all_behavior_cases
    return load_all_behavior_cases()


@pytest.fixture
def temp_session():
    """临时会话ID"""
    import uuid
    return str(uuid.uuid4())[:12]


@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    """自动设置测试环境"""
    # 设置测试环境变量
    monkeypatch.setenv("TEST_ENV", "test")
    monkeypatch.setenv("LOG_LEVEL", "INFO")


@pytest.fixture
def mock_response():
    """模拟响应fixture"""
    def _mock(intent="greeting", message="您好，请问有什么可以帮助您的？", slots=None):
        return {
            "intent": intent,
            "response": message,
            "message": message,
            "slots": slots or {},
            "node_id": f"{intent}_node"
        }
    return _mock


# 报告相关fixtures

@pytest.fixture(scope="session")
def reports_dir():
    """报告目录"""
    reports_path = ROOT_DIR / "reports"
    reports_path.mkdir(exist_ok=True)
    return reports_path


@pytest.fixture
def output_file(reports_dir):
    """输出文件fixture"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return reports_dir / f"test_{timestamp}.json"


# 跳过条件

def pytest_collection_modifyitems(config, items):
    """修改测试项"""
    # 可以根据条件跳过某些测试
    pass


def pytest_runtest_setup(item):
    """测试运行前的设置"""
    # 每个测试运行前的钩子
    pass


def pytest_runtest_teardown(item, nextitem):
    """测试运行后的清理"""
    # 每个测试运行后的钩子
    pass
