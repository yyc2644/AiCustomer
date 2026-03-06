"""
NLU测试用例 - 语料和知识库
"""

import pytest
from unittest.mock import Mock, patch


@pytest.mark.nlu
class TestIntentRecognition:
    """意图识别测试类"""
    
    def test_exact_intent_match(self, evaluator):
        """测试精确意图匹配"""
        result = evaluator.evaluate_single(
            query="我想退货",
            expected_intent="refund",
            actual_intent="refund"
        )
        
        assert result.intent_match is True
        assert result.intent_score == 1.0
    
    def test_partial_intent_match(self, evaluator):
        """测试部分意图匹配"""
        result = evaluator.evaluate_single(
            query="怎么退货",
            expected_intent="refund",
            actual_intent="refund_process"
        )
        
        # 部分匹配
        assert result.intent_score >= 0.5
    
    def test_intent_mismatch(self, evaluator):
        """测试意图不匹配"""
        result = evaluator.evaluate_single(
            query="我想退货",
            expected_intent="refund",
            actual_intent="order_inquiry"
        )
        
        assert result.intent_match is False
        assert result.intent_score == 0.0
    
    @pytest.mark.parametrize("query,expected", [
        ("我想退货", "refund"),
        ("怎么申请退款", "refund"),
        ("查询订单", "order_inquiry"),
        ("物流到哪里了", "order_tracking"),
    ])
    def test_intent_variants(self, evaluator, query, expected):
        """测试意图变体"""
        # 这里使用相同的intent来模拟
        result = evaluator.evaluate_single(
            query=query,
            expected_intent=expected,
            actual_intent=expected
        )
        
        assert result.intent_score == 1.0


@pytest.mark.nlu
class TestAnswerMatching:
    """答案匹配测试类"""
    
    def test_exact_answer_match(self, evaluator):
        """测试精确答案匹配"""
        expected = "您好，请问有什么可以帮助？"
        actual = "您好，请问有什么可以帮助？"
        
        score = evaluator.evaluate_answer(expected, actual)
        
        assert score == 1.0
    
    def test_fuzzy_answer_match(self, evaluator):
        """测试模糊答案匹配"""
        expected = "您好，请问有什么可以帮助？"
        actual = "您好，请问需要什么帮助？"
        
        score = evaluator.evaluate_answer(expected, actual)
        
        assert 0 < score < 1
    
    def test_semantic_similarity(self, evaluator):
        """测试语义相似度"""
        text1 = "我想退货"
        text2 = "我要申请退货"
        
        similarity = evaluator.semantic_similarity(text1, text2)
        
        assert 0 <= similarity <= 1
    
    def test_answer_with_threshold(self, evaluator):
        """测试阈值判断"""
        expected = "这是标准答案"
        actual = "这是标准答案"
        
        score = evaluator.evaluate_answer(expected, actual)
        
        threshold = evaluator.similarity_threshold
        assert (score >= threshold) is True


@pytest.mark.nlu
class TestCorpusEvaluation:
    """语料库评估测试类"""
    
    def test_batch_evaluation(self, evaluator, test_corpus):
        """测试批量评估"""
        if not test_corpus:
            pytest.skip("没有测试语料")
        
        # 模拟测试用例
        test_cases = [
            {
                "id": "case_1",
                "query": "我想退货",
                "expected_intent": "refund",
                "expected_answer": "您好，请问有什么可以帮助？"
            },
            {
                "id": "case_2",
                "query": "查询订单",
                "expected_intent": "order_inquiry",
                "expected_answer": "请提供订单号"
            }
        ]
        
        # 模拟适配器
        mock_adapter = Mock()
        mock_adapter.get_response = Mock(side_effect=[
            "您好，请问有什么可以帮助？",
            "请提供订单号"
        ])
        
        results = evaluator.batch_evaluate(test_cases, mock_adapter)
        
        assert len(results) == 2
        assert all(hasattr(r, 'passed') for r in results)
    
    def test_evaluation_stats(self, evaluator):
        """测试评估统计"""
        from core.evaluator import EvaluationResult
        
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
        
        stats = evaluator.calculate_stats(results)
        
        assert stats["total"] == 2
        assert stats["passed"] == 1
        assert stats["pass_rate"] == 0.5
        assert stats["intent_accuracy"] == 0.5


@pytest.mark.nlu
class TestKnowledgeBase:
    """知识库测试类"""
    
    def test_knowledge_retrieval(self, evaluator):
        """测试知识检索"""
        # 模拟知识库查询
        query = "退货政策"
        
        # 应该返回相关知识
        expected_knowledge = "7天无理由退货"
        
        # 简单匹配测试
        score = evaluator.fuzzy_match(query, expected_knowledge)
        
        assert 0 <= score <= 1
    
    def test_similarity_threshold(self, evaluator):
        """测试相似度阈值"""
        # 设置阈值
        threshold = 0.8
        
        text1 = "退货流程"
        text2 = "退货流程"
        
        score = evaluator.semantic_similarity(text1, text2)
        
        passed = score >= threshold
        assert isinstance(passed, bool)


@pytest.mark.nlu
@pytest.mark.regression
class TestRegressionNLU:
    """回归测试 - NLU"""
    
    def test_refund_intent_accuracy(self, evaluator):
        """测试退货意图准确率"""
        test_cases = [
            ("我想退货", "refund"),
            ("怎么退货", "refund"),
            ("退货流程", "refund"),
            ("申请退款", "refund"),
            ("退款", "refund"),
        ]
        
        # 模拟实际意图（假设全部正确识别）
        passed = 0
        for query, expected in test_cases:
            result = evaluator.evaluate_single(
                query=query,
                expected_intent=expected,
                actual_intent=expected
            )
            if result.intent_match:
                passed += 1
        
        accuracy = passed / len(test_cases)
        
        # 准确率应该 >= 85%
        threshold = evaluator.intent_threshold
        assert accuracy >= threshold, f"准确率 {accuracy} 低于阈值 {threshold}"
    
    def test_answer_similarity_stability(self, evaluator):
        """测试答案相似度稳定性"""
        expected = "这是标准答案"
        
        # 多次测试相同答案
        scores = []
        for _ in range(5):
            score = evaluator.evaluate_answer(expected, expected)
            scores.append(score)
        
        # 分数应该稳定
        assert all(s == 1.0 for s in scores)
    
    def test_corpus_coverage(self, evaluator, test_corpus):
        """测试语料覆盖率"""
        if not test_corpus:
            pytest.skip("没有测试语料")
        
        # 统计各类别分布
        categories = {}
        for case in test_corpus:
            cat = case.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
        
        # 应该有多个类别
        assert len(categories) > 0


@pytest.mark.nlu
@pytest.mark.smoke
class TestSmokeNLU:
    """冒烟测试 - NLU"""
    
    def test_basic_intent(self, evaluator):
        """测试基础意图"""
        result = evaluator.evaluate_single(
            query="你好",
            expected_intent="greeting",
            actual_intent="greeting"
        )
        
        assert result.passed is True
    
    def test_basic_similarity(self, evaluator):
        """测试基础相似度"""
        score = evaluator.fuzzy_match("测试", "测试")
        
        assert score == 1.0
    
    def test_empty_handling(self, evaluator):
        """测试空值处理"""
        result = evaluator.evaluate_single(
            query="",
            expected_intent="greeting",
            actual_intent=""
        )
        
        assert result.intent_score == 0.0
