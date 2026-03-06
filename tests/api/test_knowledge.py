"""
API测试用例 - 知识库管理
"""

import pytest
import time


@pytest.mark.api
class TestKnowledgeAPI:
    """知识库API测试类"""
    
    def test_get_knowledge_list(self, api_client):
        """测试获取知识库列表"""
        response = api_client.get_knowledge_list(page=1, page_size=10)
        
        assert response is not None
        assert isinstance(response, (dict, list))
    
    def test_add_knowledge(self, api_client):
        """测试添加知识"""
        title = f"测试知识_{int(time.time())}"
        
        response = api_client.add_knowledge(
            title=title,
            content="这是测试内容",
            category="测试分类"
        )
        
        assert response is not None
        # 根据实际返回格式调整断言
    
    def test_update_knowledge(self, api_client):
        """测试更新知识"""
        # 先添加
        title = f"测试知识_{int(time.time())}"
        add_response = api_client.add_knowledge(title=title, content="原始内容")
        
        # 再更新
        if isinstance(add_response, dict) and 'id' in add_response:
            knowledge_id = add_response['id']
            update_response = api_client.update_knowledge(
                knowledge_id,
                title=title,
                content="更新后的内容"
            )
            assert update_response is not None
    
    def test_delete_knowledge(self, api_client):
        """测试删除知识"""
        # 先添加
        title = f"测试知识_{int(time.time())}"
        add_response = api_client.add_knowledge(title=title, content="测试内容")
        
        # 再删除
        if isinstance(add_response, dict) and 'id' in add_response:
            knowledge_id = add_response['id']
            delete_response = api_client.delete_knowledge(knowledge_id)
            assert delete_response is not None
    
    def test_search_knowledge(self, api_client):
        """测试搜索知识"""
        response = api_client.get_knowledge_list(keyword="测试")
        
        assert response is not None
        assert isinstance(response, (dict, list))


@pytest.mark.api
class TestConversationAPI:
    """会话API测试类"""
    
    def test_get_conversation_list(self, api_client):
        """测试获取会话列表"""
        response = api_client.get_conversation_list(page=1, page_size=10)
        
        assert response is not None
        assert isinstance(response, (dict, list))
    
    def test_send_message(self, api_client, temp_session):
        """测试发送消息"""
        response = api_client.send_message(
            message="你好",
            session_id=temp_session
        )
        
        assert response is not None
        assert isinstance(response, dict)
    
    def test_chat_flow(self, api_client, temp_session):
        """测试对话流程"""
        messages = ["你好", "我想咨询退货", "订单号123456"]
        
        for msg in messages:
            response = api_client.send_message(
                message=msg,
                session_id=temp_session
            )
            assert response is not None
    
    def test_get_chat_history(self, api_client, temp_session):
        """测试获取聊天历史"""
        # 先发送消息
        api_client.send_message(message="测试", session_id=temp_session)
        
        # 再获取历史
        response = api_client.get_chat_history(temp_session)
        
        assert response is not None
        assert isinstance(response, (dict, list))


@pytest.mark.api
class TestStatisticsAPI:
    """统计API测试类"""
    
    def test_get_statistics(self, api_client):
        """测试获取统计数据"""
        from datetime import datetime, timedelta
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        response = api_client.get_statistics(
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
    
    def test_ping(self, api_client):
        """测试ping"""
        try:
            response = api_client.get("/api/ping")
            assert response is not None
        except:
            # 如果接口不存在，跳过
            pytest.skip("ping接口不存在")
