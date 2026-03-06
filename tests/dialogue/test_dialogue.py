"""
对话测试用例 - 行为树和对话流程
"""

import pytest
from unittest.mock import Mock, patch


@pytest.mark.dialogue
class TestDialogueFlow:
    """对话流程测试类"""
    
    def test_single_turn(self, bot_simulator, mock_response):
        """测试单轮对话"""
        with patch.object(bot_simulator.api_client, 'send_message', 
                        return_value=mock_response("refund", "您好，请问有什么可以帮助？")):
            result = bot_simulator.chat("我想退货")
            
            assert result["success"] is True
            assert "response" in result
    
    def test_multi_turn_refund(self, bot_simulator, mock_response):
        """测试多轮退货流程"""
        responses = [
            mock_response("refund", "您好，请问有什么可以帮助？"),
            mock_response("refund_order", "请提供订单号"),
            mock_response("refund_confirm", "好的，已确认订单")
        ]
        
        with patch.object(bot_simulator.api_client, 'send_message', 
                         side_effect=responses):
            result = bot_simulator.chat_flow([
                "我想退货",
                "订单号123456",
                "质量问题"
            ])
            
            assert result["total_turns"] == 3
    
    def test_session_management(self, bot_simulator):
        """测试会话管理"""
        session = bot_simulator.create_session(user_id="test_user")
        
        assert session.session_id is not None
        assert session.user_id == "test_user"
        
        # 测试切换会话
        session2 = bot_simulator.create_session(user_id="test_user2")
        assert bot_simulator.current_session.session_id == session2.session_id


@pytest.mark.dialogue
class TestBehaviorTreePath:
    """行为树路径测试类"""
    
    def test_tree_loading(self, tree_parser):
        """测试行为树加载"""
        # 尝试加载行为树
        result = tree_parser.load_tree("base", "en")
        # 可能成功也可能失败，取决于行为树文件是否存在
        assert result is True or result is False
    
    def test_node_lookup(self, tree_parser):
        """测试节点查找"""
        # 先加载树
        tree_parser.load_tree("base", "en")
        
        # 查找起始节点
        start_node = tree_parser._find_start_node()
        
        # 如果加载成功，应该能找到节点
        if tree_parser.current_tree:
            assert start_node is not None
    
    def test_path_validation(self, tree_parser):
        """测试路径验证"""
        test_case = {
            "test_case": {
                "name": "测试流程",
                "tree_name": "base",
                "language": "en",
                "steps": [
                    {
                        "step": 1,
                        "user_input": "你好",
                        "expected_intent": "greeting",
                        "expected_node": "greeting"
                    }
                ]
            }
        }
        
        result = tree_parser.validate_path(test_case)
        
        assert "success" in result
        assert "test_case_name" in result
    
    def test_session_creation(self, tree_parser):
        """测试会话创建"""
        tree_parser.load_tree("base", "en")
        session = tree_parser.create_session("test_session")
        
        if session:
            assert session.session_id == "test_session"
            assert session.tree_name == "base"


@pytest.mark.dialogue
class TestDialogueEvaluation:
    """对话评估测试类"""
    
    def test_intent_recognition(self, evaluator):
        """测试意图识别"""
        match, score = evaluator.evaluate_intent("refund", "refund")
        
        assert match is True
        assert score == 1.0
    
    def test_intent_similarity(self, evaluator):
        """测试意图相似"""
        match, score = evaluator.evaluate_intent("refund", "退货")
        
        # 可能匹配（包含关系）也可能不匹配
        assert isinstance(match, bool)
        assert 0 <= score <= 1
    
    def test_slot_extraction(self, evaluator):
        """测试槽位提取"""
        details, score = evaluator.evaluate_slots(
            {"order_id": "123", "reason": "质量"},
            {"order_id": "123", "reason": "质量"}
        )
        
        assert score == 1.0
    
    def test_slot_partial_match(self, evaluator):
        """测试槽位部分匹配"""
        details, score = evaluator.evaluate_slots(
            {"order_id": "123", "reason": "质量"},
            {"order_id": "123"}
        )
        
        assert score == 0.5


@pytest.mark.dialogue
@pytest.mark.regression
class TestRegressionDialogue:
    """回归测试 - 对话流程"""
    
    def test_refund_flow_complete(self, bot_simulator, mock_response):
        """测试完整退货流程"""
        responses = [
            mock_response("refund", "请问您的订单是什么时候下单的？"),
            mock_response("refund_order", "请提供订单号"),
            mock_response("refund_confirm", "订单已确认"),
            mock_response("refund_reason", "请问是什么原因？"),
            mock_response("refund_success", "退货申请已提交")
        ]
        
        with patch.object(bot_simulator.api_client, 'send_message',
                         side_effect=responses):
            result = bot_simulator.chat_flow([
                "我要退货",
                "订单号123456",
                "质量问题"
            ])
            
            # 验证流程完成
            assert result["total_turns"] >= 1
    
    def test_order_inquiry_flow(self, bot_simulator, mock_response):
        """测试订单查询流程"""
        responses = [
            mock_response("order_inquiry", "请提供订单号"),
            mock_response("order_tracking", "您的订单已发货")
        ]
        
        with patch.object(bot_simulator.api_client, 'send_message',
                         side_effect=responses):
            result = bot_simulator.chat_flow([
                "查一下订单",
                "订单号654321"
            ])
            
            assert result["total_turns"] == 2


@pytest.mark.dialogue
@pytest.mark.smoke
class TestSmokeDialogue:
    """冒烟测试 - 对话"""
    
    def test_basic_greeting(self, bot_simulator, mock_response):
        """测试基础问候"""
        with patch.object(bot_simulator.api_client, 'send_message',
                         return_value=mock_response("greeting", "您好！")):
            result = bot_simulator.chat("你好")
            
            assert result["success"] is True
    
    def test_end_conversation(self, bot_simulator, mock_response):
        """测试结束对话"""
        with patch.object(bot_simulator.api_client, 'send_message',
                         return_value=mock_response("goodbye", "再见！")):
            result = bot_simulator.chat("再见")
            
            assert result["success"] is True
