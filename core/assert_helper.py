"""
断言辅助模块
用于智能客服测试的自定义断言

功能：
1. 意图识别断言
2. 答案匹配断言
3. 语义相似度断言
4. 槽位填充断言
5. 行为树节点跳转断言
6. 自定义断言失败消息

使用方法：
    from core.assert_helper import AssertHelper, assert_intent, assert_similarity
    
    # 断言意图匹配
    assert_intent(actual_intent, expected_intent)
    
    # 断言语义相似度
    assert_similarity(actual_answer, expected_answer, threshold=0.8)
    
    # 断言槽位填充
    assert_slots(actual_slots, expected_slots)
    
    # 批量断言
    results = assert_batch(results_list)
"""

import logging
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class AssertError(AssertionError):
    """自定义断言错误"""
    message: str = ""
    expected: Any = None
    actual: Any = None
    details: Dict = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}
    
    def __str__(self):
        return self.message


class AssertHelper:
    """断言辅助类"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化断言辅助类
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        
        # 默认阈值
        self.intent_threshold = self.config.get('evaluation', {}).get('intent', {}).get('top1_threshold', 0.85)
        self.similarity_threshold = self.config.get('evaluation', {}).get('answer', {}).get('similarity_threshold', 0.80)
        self.slot_threshold = self.config.get('evaluation', {}).get('slot', {}).get('fill_threshold', 0.90)
    
    # ============================================
    # 意图相关断言
    # ============================================
    
    def assert_intent(self, actual: str, expected: str, message: str = None) -> bool:
        """
        断言意图匹配
        
        Args:
            actual: 实际意图
            expected: 期望意图
            message: 自定义错误消息
            
        Raises:
            AssertError: 断言失败
        """
        if not actual or not expected:
            raise AssertError(
                message=message or "意图不能为空",
                expected=expected,
                actual=actual
            )
        
        if actual.lower() == expected.lower():
            return True
        
        raise AssertError(
            message=message or f"意图不匹配: 期望 '{expected}', 实际 '{actual}'",
            expected=expected,
            actual=actual
        )
    
    def assert_intent_similar(self, actual: str, expected: str, threshold: float = None) -> bool:
        """
        断言意图相似（支持包含关系）
        
        Args:
            actual: 实际意图
            expected: 期望意图
            threshold: 相似度阈值
            
        Raises:
            AssertError: 断言失败
        """
        threshold = threshold or self.intent_threshold
        
        if not actual or not expected:
            raise AssertError("意图不能为空")
        
        # 完全匹配
        if actual.lower() == expected.lower():
            return True
        
        # 包含匹配
        if expected.lower() in actual.lower() or actual.lower() in expected.lower():
            return True
        
        # 计算相似度
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, expected.lower(), actual.lower()).ratio()
        
        if similarity >= threshold:
            return True
        
        raise AssertError(
            message=f"意图相似度不足: 期望 '{expected}', 实际 '{actual}', 相似度 {similarity:.2f}",
            expected=expected,
            actual=actual,
            details={"similarity": similarity, "threshold": threshold}
        )
    
    # ============================================
    # 答案相关断言
    # ============================================
    
    def assert_answer_contains(self, actual: str, expected: str, message: str = None) -> bool:
        """
        断言答案包含关键词
        
        Args:
            actual: 实际答案
            expected: 期望包含的关键词
            message: 自定义错误消息
            
        Raises:
            AssertError: 断言失败
        """
        if not actual:
            raise AssertError(
                message=message or "实际答案为空",
                expected=expected,
                actual=actual
            )
        
        if expected.lower() in actual.lower():
            return True
        
        raise AssertError(
            message=message or f"答案未包含关键词: 期望包含 '{expected}', 实际 '{actual[:100]}...'",
            expected=expected,
            actual=actual[:100] if len(actual) > 100 else actual
        )
    
    def assert_answer_exact(self, actual: str, expected: str, message: str = None) -> bool:
        """
        断言答案完全匹配
        
        Args:
            actual: 实际答案
            expected: 期望答案
            message: 自定义错误消息
            
        Raises:
            AssertError: 断言失败
        """
        if not actual or not expected:
            raise AssertError("答案不能为空")
        
        if actual.strip() == expected.strip():
            return True
        
        raise AssertError(
            message=message or f"答案不匹配",
            expected=expected,
            actual=actual
        )
    
    def assert_similarity(self, actual: str, expected: str, threshold: float = None, 
                        use_semantic: bool = False, message: str = None) -> float:
        """
        断言语义相似度
        
        Args:
            actual: 实际答案
            expected: 期望答案
            threshold: 相似度阈值
            use_semantic: 是否使用语义相似度
            message: 自定义错误消息
            
        Returns:
            相似度得分
            
        Raises:
            AssertError: 断言失败
        """
        threshold = threshold or self.similarity_threshold
        
        if not actual or not expected:
            raise AssertError("答案不能为空")
        
        # 计算相似度
        if use_semantic:
            from core.evaluator import Evaluator
            score = Evaluator.semantic_similarity(expected, actual)
        else:
            from difflib import SequenceMatcher
            score = SequenceMatcher(None, expected.lower(), actual.lower()).ratio()
        
        if score >= threshold:
            return score
        
        raise AssertError(
            message=message or f"答案相似度不足: 期望 '{expected[:50]}...', 实际 '{actual[:50]}...', 相似度 {score:.2f}",
            expected=expected,
            actual=actual,
            details={"similarity": score, "threshold": threshold}
        )
    
    # ============================================
    # 槽位相关断言
    # ============================================
    
    def assert_slots(self, actual: Dict, expected: Dict, message: str = None) -> bool:
        """
        断言槽位匹配
        
        Args:
            actual: 实际槽位
            expected: 期望槽位
            message: 自定义错误消息
            
        Raises:
            AssertError: 断言失败
        """
        if not expected:
            return True
        
        if not actual:
            raise AssertError(
                message=message or f"槽位缺失: 期望 {list(expected.keys())}, 实际 {}",
                expected=expected,
                actual=actual
            )
        
        missing = []
        mismatched = []
        
        for key, value in expected.items():
            if key not in actual:
                missing.append(key)
            elif str(value).lower() != str(actual[key]).lower():
                mismatched.append(f"{key}: 期望 '{value}', 实际 '{actual[key]}'")
        
        if missing or mismatched:
            error_msg = "槽位不匹配: "
            if missing:
                error_msg += f"缺失 {missing}; "
            if mismatched:
                error_msg += f"不匹配 {mismatched}"
            
            raise AssertError(
                message=message or error_msg,
                expected=expected,
                actual=actual
            )
        
        return True
    
    def assert_slot_value(self, slots: Dict, slot_name: str, expected_value: str, 
                         message: str = None) -> bool:
        """
        断言单个槽位值
        
        Args:
            slots: 槽位字典
            slot_name: 槽位名称
            expected_value: 期望值
            message: 自定义错误消息
            
        Raises:
            AssertError: 断言失败
        """
        if slot_name not in slots:
            raise AssertError(
                message=message or f"槽位不存在: {slot_name}",
                expected=expected_value,
                actual=None
            )
        
        actual_value = slots[slot_name]
        
        if str(actual_value).lower() == str(expected_value).lower():
            return True
        
        raise AssertError(
            message=message or f"槽位值不匹配: {slot_name}, 期望 '{expected_value}', 实际 '{actual_value}'",
            expected=expected_value,
            actual=actual_value
        )
    
    # ============================================
    # 行为树相关断言
    # ============================================
    
    def assert_node(self, actual_node: str, expected_node: str, message: str = None) -> bool:
        """
        断言节点匹配
        
        Args:
            actual_node: 实际节点ID
            expected_node: 期望节点ID
            message: 自定义错误消息
            
        Raises:
            AssertError: 断言失败
        """
        if not actual_node or not expected_node:
            raise AssertError("节点ID不能为空")
        
        if actual_node == expected_node:
            return True
        
        raise AssertError(
            message=message or f"节点不匹配: 期望 '{expected_node}', 实际 '{actual_node}'",
            expected=expected_node,
            actual=actual_node
        )
    
    def assert_node_transition(self, from_node: str, to_node: str, 
                             valid_transitions: List[str], message: str = None) -> bool:
        """
        断言节点跳转是否合法
        
        Args:
            from_node: 起始节点
            to_node: 目标节点
            valid_transitions: 合法的跳转目标列表
            message: 自定义错误消息
            
        Raises:
            AssertError: 断言失败
        """
        if to_node in valid_transitions:
            return True
        
        raise AssertError(
            message=message or f"非法节点跳转: 从 '{from_node}' 到 '{to_node}', 有效跳转: {valid_transitions}",
            expected=valid_transitions,
            actual=to_node
        )
    
    # ============================================
    # 响应相关断言
    # ============================================
    
    def assert_response_time(self, response_time: float, max_time: float, 
                           message: str = None) -> bool:
        """
        断言响应时间
        
        Args:
            response_time: 实际响应时间（秒）
            max_time: 最大允许时间
            message: 自定义错误消息
            
        Raises:
            AssertError: 断言失败
        """
        if response_time <= max_time:
            return True
        
        raise AssertError(
            message=message or f"响应时间过长: {response_time:.3f}s > {max_time}s",
            expected=f"<={max_time}s",
            actual=f"{response_time:.3f}s"
        )
    
    def assert_not_empty(self, value: Any, field_name: str = "字段", message: str = None) -> bool:
        """
        断言非空
        
        Args:
            value: 要检查的值
            field_name: 字段名称
            message: 自定义错误消息
            
        Raises:
            AssertError: 断言失败
        """
        if value:
            return True
        
        raise AssertError(
            message=message or f"{field_name}不能为空",
            expected="非空",
            actual=None
        )
    
    def assert_status_code(self, actual: int, expected: int, message: str = None) -> bool:
        """
        断言HTTP状态码
        
        Args:
            actual: 实际状态码
            expected: 期望状态码
            message: 自定义错误消息
            
        Raises:
            AssertError: 断言失败
        """
        if actual == expected:
            return True
        
        raise AssertError(
            message=message or f"状态码不匹配: 期望 {expected}, 实际 {actual}",
            expected=expected,
            actual=actual
        )
    
    # ============================================
    # 批量断言
    # ============================================
    
    def assert_all(self, assertions: List[Callable[[], bool]]) -> Dict:
        """
        执行批量断言
        
        Args:
            assertions: 断言函数列表
            
        Returns:
            执行结果字典
        """
        results = []
        passed = 0
        failed = 0
        
        for i, assertion in enumerate(assertions):
            try:
                assertion()
                passed += 1
                results.append({"index": i, "passed": True})
            except AssertError as e:
                failed += 1
                results.append({
                    "index": i,
                    "passed": False,
                    "error": str(e),
                    "expected": e.expected,
                    "actual": e.actual
                })
            except AssertionError as e:
                failed += 1
                results.append({
                    "index": i,
                    "passed": False,
                    "error": str(e)
                })
        
        return {
            "total": len(assertions),
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / len(assertions) if assertions else 0,
            "results": results
        }
    
    def assert_batch_results(self, results: List[Dict], threshold: float = None) -> Dict:
        """
        批量断言测试结果
        
        Args:
            results: 测试结果列表
            threshold: 通过阈值
            
        Returns:
            汇总结果
        """
        threshold = threshold or self.intent_threshold
        
        passed = 0
        failed_results = []
        
        for result in results:
            score = result.get('overall_score', result.get('intent_score', 0))
            if score >= threshold:
                passed += 1
            else:
                failed_results.append(result)
        
        total = len(results)
        
        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total > 0 else 0,
            "failed_results": failed_results[:10]  # 最多返回10个失败案例
        }


# ============================================
# 便捷函数
# ============================================

# 创建默认实例
_default_helper = AssertHelper()


def assert_intent(actual: str, expected: str, message: str = None) -> bool:
    """断言意图匹配"""
    return _default_helper.assert_intent(actual, expected, message)


def assert_intent_similar(actual: str, expected: str, threshold: float = None) -> bool:
    """断言意图相似"""
    return _default_helper.assert_intent_similar(actual, expected, threshold)


def assert_answer_contains(actual: str, expected: str, message: str = None) -> bool:
    """断言答案包含关键词"""
    return _default_helper.assert_answer_contains(actual, expected, message)


def assert_similarity(actual: str, expected: str, threshold: float = None, 
                     use_semantic: bool = False, message: str = None) -> float:
    """断言语义相似度"""
    return _default_helper.assert_similarity(actual, expected, threshold, use_semantic, message)


def assert_slots(actual: Dict, expected: Dict, message: str = None) -> bool:
    """断言槽位匹配"""
    return _default_helper.assert_slots(actual, expected, message)


def assert_node(actual_node: str, expected_node: str, message: str = None) -> bool:
    """断言节点匹配"""
    return _default_helper.assert_node(actual_node, expected_node, message)


def assert_response_time(response_time: float, max_time: float, message: str = None) -> bool:
    """断言响应时间"""
    return _default_helper.assert_response_time(response_time, max_time, message)


def assert_not_empty(value: Any, field_name: str = "字段", message: str = None) -> bool:
    """断言非空"""
    return _default_helper.assert_not_empty(value, field_name, message)


def assert_batch_results(results: List[Dict], threshold: float = None) -> Dict:
    """批量断言测试结果"""
    return _default_helper.assert_batch_results(results, threshold)
