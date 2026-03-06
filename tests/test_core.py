"""
核心模块单元测试
测试 evaluator, tree_parser, assert_helper 等核心功能
"""

import pytest
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.evaluator import Evaluator, EvaluationResult
from core.tree_parser import TreeParser, TreeNode, DialogSession, DialogTurn
from core.assert_helper import AssertHelper, assert_intent, assert_similarity


class TestEvaluator:
    """评估器单元测试"""
    
    def test_evaluate_exact_match(self):
        """测试精确匹配"""
        score = Evaluator.evaluate("退货", "我想退货")
        assert score == 1.0
    
    def test_evaluate_no_match(self):
        """测试不匹配"""
        score = Evaluator.evaluate("退货", "查询订单")
        assert score == 0.0
    
    def test_fuzzy_match_identical(self):
        """测试模糊匹配 - 相同文本"""
        score = Evaluator.fuzzy_match("你好", "你好")
        assert score == 1.0
    
    def test_fuzzy_match_similar(self):
        """测试模糊匹配 - 相似文本"""
        score = Evaluator.fuzzy_match("我想退货", "我要退货")
        assert 0 < score < 1
    
    def test_fuzzy_match_different(self):
        """测试模糊匹配 - 不同文本"""
        score = Evaluator.fuzzy_match("退货", "订单")
        assert 0 <= score < 0.5
    
    def test_evaluate_intent_exact(self):
        """测试意图识别 - 精确匹配"""
        evaluator = Evaluator()
        match, score = evaluator.evaluate_intent("refund", "refund")
        assert match is True
        assert score == 1.0
    
    def test_evaluate_intent_contains(self):
        """测试意图识别 - 包含关系"""
        evaluator = Evaluator()
        match, score = evaluator.evaluate_intent("refund", "refund_process")
        assert match is True
        assert score == 0.8
    
    def test_evaluate_intent_mismatch(self):
        """测试意图识别 - 不匹配"""
        evaluator = Evaluator()
        match, score = evaluator.evaluate_intent("refund", "order")
        assert match is False
        assert score == 0.0
    
    def test_evaluate_slots_full_match(self):
        """测试槽位填充 - 完全匹配"""
        evaluator = Evaluator()
        details, score = evaluator.evaluate_slots(
            {"order_id": "123", "reason": "质量"},
            {"order_id": "123", "reason": "质量"}
        )
        assert score == 1.0
    
    def test_evaluate_slots_partial(self):
        """测试槽位填充 - 部分匹配"""
        evaluator = Evaluator()
        details, score = evaluator.evaluate_slots(
            {"order_id": "123", "reason": "质量"},
            {"order_id": "123"}
        )
        assert score == 0.5
    
    def test_evaluate_slots_no_match(self):
        """测试槽位填充 - 不匹配"""
        evaluator = Evaluator()
        details, score = evaluator.evaluate_slots(
            {"order_id": "123"},
            {"order_id": "456"}
        )
        assert score == 0.0
    
    def test_evaluate_answer_exact(self):
        """测试答案匹配 - 完全匹配"""
        evaluator = Evaluator()
        score = evaluator.evaluate_answer("您好，请问有什么可以帮助？", "您好，请问有什么可以帮助？")
        assert score == 1.0
    
    def test_evaluate_answer_fuzzy(self):
        """测试答案匹配 - 模糊匹配"""
        evaluator = Evaluator()
        score = evaluator.evaluate_answer("退货政策", "7天无理由退货")
        assert 0 <= score <= 1
    
    def test_evaluate_single_complete(self):
        """测试完整单条评估"""
        evaluator = Evaluator()
        result = evaluator.evaluate_single(
            query="我想退货",
            expected_intent="refund",
            actual_intent="refund",
            expected_answer="您好，请问有什么可以帮助？",
            actual_answer="您好，请问有什么可以帮助？",
            case_id="test_001"
        )
        
        assert result.case_id == "test_001"
        assert result.intent_match is True
        assert result.intent_score == 1.0
        assert result.overall_score == 1.0
        assert result.passed is True
    
    def test_calculate_stats(self):
        """测试统计计算"""
        results = [
            EvaluationResult(
                case_id="1", query="", expected="", actual="",
                intent_expected="refund", intent_actual="refund",
                intent_score=1.0, answer_score=0.9, overall_score=0.95,
                passed=True
            ),
            EvaluationResult(
                case_id="2", query="", expected="", actual="",
                intent_expected="refund", intent_actual="order",
                intent_score=0.0, answer_score=0.0, overall_score=0.0,
                passed=False
            )
        ]
        
        stats = Evaluator.calculate_stats(results)
        
        assert stats["total"] == 2
        assert stats["passed"] == 1
        assert stats["pass_rate"] == 0.5
        assert stats["intent_accuracy"] == 0.5


class TestTreeParser:
    """行为树解析器单元测试"""
    
    def test_tree_node_creation(self):
        """测试树节点创建"""
        node = TreeNode(
            node_id="refund_start",
            node_type="dialog",
            intent="refund",
            response="您好，请问有什么可以帮助？",
            next_nodes=["refund_confirm"]
        )
        
        assert node.node_id == "refund_start"
        assert node.intent == "refund"
        assert "refund_confirm" in node.next_nodes
    
    def test_dialog_session_creation(self):
        """测试对话会话创建"""
        session = DialogSession(
            session_id="test_001",
            tree_name="base",
            language="zh-CN"
        )
        
        assert session.session_id == "test_001"
        assert session.tree_name == "base"
        assert len(session.history) == 0
    
    def test_dialog_session_add_turn(self):
        """测试添加对话轮次"""
        session = DialogSession(
            session_id="test_001",
            tree_name="base",
            language="zh-CN"
        )
        
        turn = DialogTurn(
            turn_index=1,
            user_input="我想退货",
            detected_intent="refund",
            extracted_slots={},
            current_node_id="refund_start",
            bot_response="您好，请问有什么可以帮助？"
        )
        
        session.add_turn(turn)
        
        assert len(session.history) == 1
        assert session.slots == {}
    
    def test_dialog_session_slots_update(self):
        """测试槽位更新"""
        session = DialogSession(
            session_id="test_001",
            tree_name="base",
            language="zh-CN"
        )
        
        turn = DialogTurn(
            turn_index=1,
            user_input="订单号123456",
            detected_intent="order_number",
            extracted_slots={"order_id": "123456"},
            current_node_id="order_confirm",
            bot_response="好的，已确认订单"
        )
        
        session.add_turn(turn)
        
        assert session.slots["order_id"] == "123456"


class TestAssertHelper:
    """断言辅助单元测试"""
    
    def test_assert_intent_match(self):
        """测试意图断言 - 匹配"""
        result = assert_intent("refund", "refund")
        assert result is True
    
    def test_assert_intent_no_match(self):
        """测试意图断言 - 不匹配"""
        with pytest.raises(Exception):
            assert_intent("refund", "order")
    
    def test_assert_similarity_high(self):
        """测试相似度断言 - 高相似度"""
        helper = AssertHelper()
        score = helper.assert_similarity("我想退货", "我想退货", threshold=0.8)
        assert score >= 0.8
    
    def test_assert_similarity_low(self):
        """测试相似度断言 - 低相似度"""
        helper = AssertHelper()
        with pytest.raises(Exception):
            helper.assert_similarity("退货", "订单", threshold=0.8)
    
    def test_assert_slots_match(self):
        """测试槽位断言 - 匹配"""
        helper = AssertHelper()
        result = helper.assert_slots(
            {"order_id": "123", "reason": "质量"},
            {"order_id": "123", "reason": "质量"}
        )
        assert result is True
    
    def test_assert_slots_missing(self):
        """测试槽位断言 - 缺失"""
        helper = AssertHelper()
        with pytest.raises(Exception):
            helper.assert_slots({}, {"order_id": "123"})
    
    def test_assert_not_empty(self):
        """测试非空断言"""
        helper = AssertHelper()
        result = helper.assert_not_empty("test", "测试字段")
        assert result is True
    
    def test_assert_not_empty_fail(self):
        """测试非空断言 - 失败"""
        helper = AssertHelper()
        with pytest.raises(Exception):
            helper.assert_not_empty("", "测试字段")
    
    def test_assert_response_time(self):
        """测试响应时间断言"""
        helper = AssertHelper()
        result = helper.assert_response_time(0.5, 1.0)
        assert result is True
    
    def test_assert_response_time_fail(self):
        """测试响应时间断言 - 失败"""
        helper = AssertHelper()
        with pytest.raises(Exception):
            helper.assert_response_time(2.0, 1.0)
    
    def test_assert_all(self):
        """测试批量断言"""
        helper = AssertHelper()
        
        def assertion1():
            return True
        
        def assertion2():
            return True
        
        results = helper.assert_all([assertion1, assertion2])
        
        assert results["passed"] == 2
        assert results["failed"] == 0


class TestIntegration:
    """集成测试"""
    
    def test_evaluator_with_threshold(self):
        """测试评估器阈值"""
        config = {
            "evaluation": {
                "intent": {"top1_threshold": 0.9},
                "answer": {"similarity_threshold": 0.85}
            }
        }
        
        evaluator = Evaluator(config)
        
        assert evaluator.intent_threshold == 0.9
        assert evaluator.similarity_threshold == 0.85
    
    def test_full_evaluation_pipeline(self):
        """测试完整评估流程"""
        evaluator = Evaluator()
        
        # 评估意图
        match, score = evaluator.evaluate_intent("refund", "refund")
        
        # 评估答案
        answer_score = evaluator.evaluate_answer(
            "您好，请问有什么可以帮助？",
            "您好，请问有什么可以帮助？"
        )
        
        # 评估槽位
        slot_detail, slot_score = evaluator.evaluate_slots(
            {"order_id": "123"},
            {"order_id": "123"}
        )
        
        # 验证结果
        assert match is True
        assert answer_score == 1.0
        assert slot_score == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
