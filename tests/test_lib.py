"""
Lib模块单元测试
测试 api_client, bot_simulator, db_manager 等库功能
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestAPIClient:
    """API客户端单元测试"""
    
    @patch('lib.api_client.requests.Session')
    def test_api_client_creation(self, mock_session):
        """测试API客户端创建"""
        from lib.api_client import APIClient
        
        client = APIClient("test")
        
        assert client is not None
        assert client.env == "test"
    
    @patch('lib.api_client.requests.Session')
    def test_api_client_default_env(self, mock_session):
        """测试API客户端默认环境"""
        from lib.api_client import APIClient
        
        client = APIClient()
        
        assert client.env == "test"
    
    @patch('lib.api_client.requests.Session')
    def test_get_full_url(self, mock_session):
        """测试获取完整URL"""
        from lib.api_client import APIClient
        
        client = APIClient("test")
        client.base_url = "https://api.example.com"
        
        url = client._get_full_url("/api/test")
        
        assert url == "https://api.example.com/api/test"
    
    @patch('lib.api_client.requests.Session')
    def test_get_full_url_absolute(self, mock_session):
        """测试获取完整URL - 绝对路径"""
        from lib.api_client import APIClient
        
        client = APIClient("test")
        
        url = client._get_full_url("https://other.com/api")
        
        assert url == "https://other.com/api"
    
    @patch('lib.api_client.requests.Session')
    def test_update_headers(self, mock_session):
        """测试更新请求头"""
        from lib.api_client import APIClient
        
        client = APIClient("test")
        client.token = "test_token"
        
        client._update_headers()
        
        assert "Authorization" in client.headers
        assert client.headers["Authorization"] == "Bearer test_token"
    
    @patch('lib.api_client.requests.Session')
    def test_session_creation(self, mock_session):
        """测试会话创建"""
        from lib.api_client import APIClient
        
        client = APIClient("test")
        
        assert client.session is not None


class TestBotSimulator:
    """机器人模拟器单元测试"""
    
    def test_simulator_creation(self):
        """测试模拟器创建"""
        from lib.bot_simulator import BotSimulator
        
        simulator = BotSimulator()
        
        assert simulator is not None
        assert len(simulator.sessions) == 0
    
    def test_create_session(self):
        """测试创建会话"""
        from lib.bot_simulator import BotSimulator
        
        simulator = BotSimulator()
        session = simulator.create_session(user_id="test_user")
        
        assert session is not None
        assert session.session_id is not None
        assert session.user_id == "test_user"
        assert len(simulator.sessions) == 1
    
    def test_get_session(self):
        """测试获取会话"""
        from lib.bot_simulator import BotSimulator
        
        simulator = BotSimulator()
        created_session = simulator.create_session("test_user")
        
        retrieved_session = simulator.get_session(created_session.session_id)
        
        assert retrieved_session is not None
        assert retrieved_session.session_id == created_session.session_id
    
    def test_switch_session(self):
        """测试切换会话"""
        from lib.bot_simulator import BotSimulator
        
        simulator = BotSimulator()
        session1 = simulator.create_session("user1")
        session2 = simulator.create_session("user2")
        
        result = simulator.switch_session(session1.session_id)
        
        assert result is True
        assert simulator.current_session.session_id == session1.session_id
    
    def test_chat_without_api(self):
        """测试无API的单轮对话"""
        from lib.bot_simulator import BotSimulator
        
        simulator = BotSimulator()  # 没有API客户端
        
        result = simulator.chat("你好")
        
        # 没有API时会返回模拟响应
        assert result is not None
        assert "response" in result or "error" in result
    
    def test_chat_flow(self):
        """测试多轮对话"""
        from lib.bot_simulator import BotSimulator
        
        simulator = BotSimulator()
        
        result = simulator.chat_flow([
            "你好",
            "我想退货"
        ])
        
        assert result is not None
        assert "total_turns" in result
        assert result["total_turns"] == 2
    
    def test_clear_sessions(self):
        """测试清除会话"""
        from lib.bot_simulator import BotSimulator
        
        simulator = BotSimulator()
        simulator.create_session("user1")
        simulator.create_session("user2")
        
        assert len(simulator.sessions) == 2
        
        simulator.clear_sessions()
        
        assert len(simulator.sessions) == 0


class TestDialogTurn:
    """对话轮次单元测试"""
    
    def test_dialog_turn_creation(self):
        """测试对话轮次创建"""
        from lib.bot_simulator import DialogTurn
        
        turn = DialogTurn(
            turn_index=1,
            user_input="你好",
            bot_response="您好"
        )
        
        assert turn.turn_index == 1
        assert turn.user_input == "你好"
        assert turn.bot_response == "您好"
    
    def test_dialog_turn_to_dict(self):
        """测试对话轮次转字典"""
        from lib.bot_simulator import DialogTurn
        
        turn = DialogTurn(
            turn_index=1,
            user_input="你好",
            bot_response="您好"
        )
        
        result = turn.to_dict()
        
        assert isinstance(result, dict)
        assert result["turn_index"] == 1
        assert result["user_input"] == "你好"


class TestChatSession:
    """对话会话单元测试"""
    
    def test_chat_session_creation(self):
        """测试对话会话创建"""
        from lib.bot_simulator import ChatSession
        
        session = ChatSession(
            session_id="test_001",
            user_id="user_001",
            language="zh-CN"
        )
        
        assert session.session_id == "test_001"
        assert session.user_id == "user_001"
        assert session.language == "zh-CN"
        assert len(session.turns) == 0
    
    def test_add_turn(self):
        """测试添加对话轮次"""
        from lib.bot_simulator import ChatSession, DialogTurn
        
        session = ChatSession(session_id="test_001", language="zh-CN")
        
        turn = DialogTurn(
            turn_index=1,
            user_input="你好",
            bot_response="您好"
        )
        
        session.add_turn(turn)
        
        assert len(session.turns) == 1
    
    def test_get_history(self):
        """测试获取历史"""
        from lib.bot_simulator import ChatSession, DialogTurn
        
        session = ChatSession(session_id="test_001", language="zh-CN")
        
        turn = DialogTurn(
            turn_index=1,
            user_input="你好",
            bot_response="您好"
        )
        session.add_turn(turn)
        
        history = session.get_history()
        
        assert isinstance(history, list)
        assert len(history) == 1
    
    def test_get_last_bot_response(self):
        """测试获取最后机器人回复"""
        from lib.bot_simulator import ChatSession, DialogTurn
        
        session = ChatSession(session_id="test_001", language="zh-CN")
        
        turn1 = DialogTurn(turn_index=1, user_input="你好", bot_response="您好")
        turn2 = DialogTurn(turn_index=2, user_input="我想退货", bot_response="请问什么问题？")
        
        session.add_turn(turn1)
        session.add_turn(turn2)
        
        last_response = session.get_last_bot_response()
        
        assert last_response == "请问什么问题？"
    
    def test_get_last_intent(self):
        """测试获取最后意图"""
        from lib.bot_simulator import ChatSession, DialogTurn
        
        session = ChatSession(session_id="test_001", language="zh-CN")
        
        turn = DialogTurn(
            turn_index=1,
            user_input="你好",
            bot_response="您好",
            intent="greeting"
        )
        session.add_turn(turn)
        
        last_intent = session.get_last_intent()
        
        assert last_intent == "greeting"


class TestDBManager:
    """数据库管理器单元测试"""
    
    @patch('lib.db_manager.DBManager._get_db_config')
    def test_db_manager_creation(self, mock_config):
        """测试数据库管理器创建"""
        from lib.db_manager import DBManager, DBConfig
        
        # 模拟配置
        mock_config.return_value = DBConfig(
            host="localhost",
            port=3306,
            database="test_db",
            user="root",
            password=""
        )
        
        db = DBManager("test")
        
        assert db is not None
        assert db.env == "test"


class TestLocatorHelper:
    """定位器辅助单元测试"""
    
    def test_css_locator(self):
        """测试CSS定位器"""
        from lib.page_objects.base import LocatorHelper
        
        locator = LocatorHelper.css("#input-box", "输入框")
        
        assert locator.type == "css"
        assert locator.value == "#input-box"
        assert locator.description == "输入框"
    
    def test_xpath_locator(self):
        """测试XPath定位器"""
        from lib.page_objects.base import LocatorHelper
        
        locator = LocatorHelper.xpath("//button[@id='send']", "发送按钮")
        
        assert locator.type == "xpath"
        assert "//button" in locator.value
    
    def test_id_locator(self):
        """测试ID定位器"""
        from lib.page_objects.base import LocatorHelper
        
        locator = LocatorHelper.id("username", "用户名输入框")
        
        assert locator.type == "id"
        assert locator.value == "username"
    
    def test_name_locator(self):
        """测试Name定位器"""
        from lib.page_objects.base import LocatorHelper
        
        locator = LocatorHelper.name("email", "邮箱输入框")
        
        assert locator.type == "name"
        assert locator.value == "email"
    
    def test_text_locator(self):
        """测试文本定位器"""
        from lib.page_objects.base import LocatorHelper
        
        locator = LocatorHelper.text("提交", "提交按钮")
        
        assert locator.type == "xpath"
        assert "提交" in locator.value
    
    def test_contains_locator(self):
        """测试包含文本定位器"""
        from lib.page_objects.base import LocatorHelper
        
        locator = LocatorHelper.contains("登录", "登录链接")
        
        assert locator.type == "xpath"
        assert "contains" in locator.value


class TestConvenienceFunctions:
    """便捷函数测试"""
    
    def test_create_client(self):
        """测试创建API客户端便捷函数"""
        from lib.api_client import create_client
        
        with patch('lib.api_client.requests.Session'):
            client = create_client("test")
        
        assert client is not None
    
    def test_create_simulator(self):
        """测试创建模拟器便捷函数"""
        from lib.bot_simulator import create_simulator
        
        simulator = create_simulator()
        
        assert simulator is not None
    
    def test_create_db_manager(self):
        """测试创建数据库管理器便捷函数"""
        from lib.db_manager import create_db_manager
        
        with patch('lib.db_manager.DBManager._get_db_config'):
            db = create_db_manager("test")
        
        assert db is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
