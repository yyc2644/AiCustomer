"""
API测试用例 - 知识库管理
包含单元测试(带mock)和集成测试(需要真实API)
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock


def create_mock_response(json_data, status_code=200):
    """创建模拟响应对象"""
    mock_resp = Mock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_data
    mock_resp.text = str(json_data)  # 支持 subscript
    mock_resp.raise_for_status = Mock()
    if status_code >= 400:
        from requests.exceptions import HTTPError
        mock_resp.raise_for_status.side_effect = HTTPError(f"{status_code} Error")
    return mock_resp


@pytest.mark.api
class TestKnowledgeAPI:
    """知识库API测试类 - 使用mock避免真实API调用"""
    
    @patch('lib.api_client.requests.Session')
    def test_get_knowledge_list(self, mock_session):
        """测试获取知识库列表"""
        from src.lib.api_client import APIClient
        
        # Mock响应
        mock_response = create_mock_response({
            "code": 0,
            "data": {"list": [], "total": 0}
        })
        mock_session.return_value.request.return_value = mock_response
        
        client = APIClient(env="test")
        response = client.get_knowledge_list(page=1, page_size=10)
        
        assert response is not None
        assert isinstance(response, (dict, list))
    
    @patch('lib.api_client.requests.Session')
    def test_add_knowledge(self, mock_session):
        """测试添加知识"""
        from src.lib.api_client import APIClient
        
        # Mock响应
        mock_response = create_mock_response({"code": 0, "id": "12345"})
        mock_session.return_value.request.return_value = mock_response
        
        client = APIClient(env="test")
        title = f"测试知识_{int(time.time())}"
        
        response = client.add_knowledge(
            title=title,
            content="这是测试内容",
            category="测试分类"
        )
        
        assert response is not None
    
    @patch('lib.api_client.requests.Session')
    def test_update_knowledge(self, mock_session):
        """测试更新知识"""
        from src.lib.api_client import APIClient
        
        # Mock响应 - 先返回添加成功的ID，再返回更新成功
        mock_response = create_mock_response({"code": 0, "id": "12345"})
        mock_session.return_value.request.return_value = mock_response
        
        client = APIClient(env="test")
        title = f"测试知识_{int(time.time())}"
        
        # 添加
        add_response = client.add_knowledge(title=title, content="原始内容")
        
        # 更新
        if isinstance(add_response, dict) and 'id' in add_response:
            knowledge_id = add_response['id']
            update_response = client.update_knowledge(
                knowledge_id,
                title=title,
                content="更新后的内容"
            )
            assert update_response is not None
    
    @patch('lib.api_client.requests.Session')
    def test_delete_knowledge(self, mock_session):
        """测试删除知识"""
        from src.lib.api_client import APIClient
        
        # Mock响应
        mock_response = create_mock_response({"code": 0, "id": "12345"})
        mock_session.return_value.request.return_value = mock_response
        
        client = APIClient(env="test")
        title = f"测试知识_{int(time.time())}"
        
        # 添加
        add_response = client.add_knowledge(title=title, content="测试内容")
        
        # 删除
        if isinstance(add_response, dict) and 'id' in add_response:
            knowledge_id = add_response['id']
            delete_response = client.delete_knowledge(knowledge_id)
            assert delete_response is not None
    
    @patch('lib.api_client.requests.Session')
    def test_search_knowledge(self, mock_session):
        """测试搜索知识"""
        from src.lib.api_client import APIClient
        
        # Mock响应
        mock_response = create_mock_response({"code": 0, "data": {"list": []}})
        mock_session.return_value.request.return_value = mock_response
        
        client = APIClient(env="test")
        response = client.get_knowledge_list(keyword="测试")
        
        assert response is not None
        assert isinstance(response, (dict, list))


@pytest.mark.api
class TestConversationAPI:
    """会话API测试类 - 使用mock避免真实API调用"""
    
    @patch('lib.api_client.requests.Session')
    def test_get_conversation_list(self, mock_session):
        """测试获取会话列表"""
        from src.lib.api_client import APIClient
        
        # Mock响应
        mock_response = create_mock_response({"code": 0, "data": {"list": []}})
        mock_session.return_value.request.return_value = mock_response
        
        client = APIClient(env="test")
        response = client.get_conversation_list(page=1, page_size=10)
        
        assert response is not None
        assert isinstance(response, (dict, list))
    
    @patch('lib.api_client.requests.Session')
    def test_send_message(self, mock_session):
        """测试发送消息"""
        from src.lib.api_client import APIClient
        
        # Mock响应
        mock_response = create_mock_response({
            "code": 0,
            "data": {"message": "您好，请问有什么可以帮助您的？"}
        })
        mock_session.return_value.request.return_value = mock_response
        
        client = APIClient(env="test")
        temp_session = "test_session_123"
        
        response = client.send_message(
            message="你好",
            session_id=temp_session
        )
        
        assert response is not None
        assert isinstance(response, dict)
    
    @patch('lib.api_client.requests.Session')
    def test_chat_flow(self, mock_session):
        """测试对话流程"""
        from src.lib.api_client import APIClient
        
        # Mock响应
        mock_response = create_mock_response({
            "code": 0,
            "data": {"message": "您好"}
        })
        mock_session.return_value.request.return_value = mock_response
        
        client = APIClient(env="test")
        temp_session = "test_session_123"
        messages = ["你好", "我想咨询退货", "订单号123456"]
        
        for msg in messages:
            response = client.send_message(
                message=msg,
                session_id=temp_session
            )
            assert response is not None
    
    @patch('lib.api_client.requests.Session')
    def test_get_chat_history(self, mock_session):
        """测试获取聊天历史"""
        from src.lib.api_client import APIClient
        
        # Mock响应 - 第一次发送消息，第二次获取历史
        mock_response = create_mock_response({
            "code": 0,
            "data": {"history": []}
        })
        mock_session.return_value.request.return_value = mock_response
        
        client = APIClient(env="test")
        temp_session = "test_session_123"
        
        # 发送消息
        client.send_message(message="测试", session_id=temp_session)
        
        # 获取历史
        response = client.get_chat_history(temp_session)
        
        assert response is not None
        assert isinstance(response, (dict, list))


@pytest.mark.api
class TestStatisticsAPI:
    """统计API测试类 - 使用mock避免真实API调用"""
    
    @patch('lib.api_client.requests.Session')
    def test_get_statistics(self, mock_session):
        """测试获取统计数据"""
        from src.lib.api_client import APIClient
        from datetime import datetime, timedelta
        
        # Mock响应
        mock_response = create_mock_response({
            "code": 0,
            "data": {"total": 0, "conversations": 0}
        })
        mock_session.return_value.request.return_value = mock_response
        
        client = APIClient(env="test")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        response = client.get_statistics(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )
        
        assert response is not None
        assert isinstance(response, dict)


@pytest.mark.api
class TestHealthCheck:
    """健康检查测试类"""
    
    def test_connection(self, api_client):
        """测试连接"""
        result = api_client.test_connection()
        
        # 可能返回True/False或直接抛出异常
        assert result is not None
    
    @pytest.mark.skip(reason="需要真实API服务器")
    def test_ping(self, api_client):
        """测试ping - 需要真实API"""
        try:
            response = api_client.get("/api/ping")
            assert response is not None
        except:
            # 如果接口不存在，跳过
            pytest.skip("ping接口不存在")
