"""
评估器模块
用于评估智能客服的回复质量

功能：
1. 意图识别准确率评估
2. 语义相似度计算
3. 答案匹配度评估
4. 槽位填充准确率评估
5. 批量测试评估

使用方法：
    from core.evaluator import Evaluator, EvaluationResult
    
    # 简单关键词匹配评估
    score = Evaluator.evaluate("我想退货", "您可以申请退货")
    
    # 语义相似度评估（需要embedding模型）
    similarity = Evaluator.semantic_similarity(
        "我想退货",
        "您可以申请7天无理由退货",
        model_path="test/v119/model"
    )
    
    # 批量评估
    results = Evaluator.batch_evaluate(test_cases, adapter)
"""

import time
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher


logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """评估结果数据类"""
    case_id: str
    query: str
    expected: str
    actual: str
    intent_expected: str
    intent_actual: str
    
    # 评分
    intent_score: float = 0.0      # 意图识别得分
    answer_score: float = 0.0       # 答案匹配得分
    similarity_score: float = 0.0   # 语义相似度得分
    overall_score: float = 0.0     # 综合得分
    
    # 详细信息
    intent_match: bool = False       # 意图是否匹配
    slots_filled: Dict = None        # 槽位填充情况
    slots_expected: Dict = None      # 预期槽位
    response_time: float = 0.0       # 响应时间（秒）
    
    # 状态
    passed: bool = False
    error: str = ""
    
    def __post_init__(self):
        if self.slots_filled is None:
            self.slots_filled = {}
        if self.slots_expected is None:
            self.slots_expected = {}


class Evaluator:
    """评估器类"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化评估器
        
        Args:
            config: 配置字典，包含阈值等参数
        """
        self.config = config or {}
        
        # 默认阈值
        self.intent_threshold = self.config.get('evaluation', {}).get('intent', {}).get('top1_threshold', 0.85)
        self.similarity_threshold = self.config.get('evaluation', {}).get('answer', {}).get('similarity_threshold', 0.80)
        self.slot_threshold = self.config.get('evaluation', {}).get('slot', {}).get('fill_threshold', 0.90)
        
        # embedding模型（延迟加载）
        self._embedding_model = None
    
    @property
    def embedding_model(self):
        """延迟加载embedding模型"""
        if self._embedding_model is None:
            self._embedding_model = self._load_embedding_model()
        return self._embedding_model
    
    def _load_embedding_model(self):
        """加载embedding模型"""
        try:
            from sentence_transformers import SentenceTransformer
            
            model_config = self.config.get('model', {}).get('embedding_model', {})
            model_path = model_config.get('model_path')
            
            if model_path:
                logger.info(f"加载embedding模型: {model_path}")
                return SentenceTransformer(model_path)
            else:
                logger.warning("未配置embedding模型路径")
                return None
        except ImportError:
            logger.warning("sentence-transformers未安装，使用备选方案")
            return None
        except Exception as e:
            logger.error(f"加载embedding模型失败: {e}")
            return None
    
    # ============================================
    # 基础评估方法
    # ============================================
    
    @staticmethod
    def evaluate(expected: str, actual: str) -> float:
        """
        简单关键词匹配评估
        
        如果期望内容出现在实际回复中，返回1.0，否则返回0.0
        
        Args:
            expected: 期望内容
            actual: 实际回复
            
        Returns:
            得分（0.0 或 1.0）
        """
        if not expected or not actual:
            return 0.0
        
        if expected.lower() in actual.lower():
            return 1.0
        return 0.0
    
    @staticmethod
    def fuzzy_match(expected: str, actual: str) -> float:
        """
        模糊匹配评估
        
        使用SequenceMatcher计算相似度
        
        Args:
            expected: 期望内容
            actual: 实际回复
            
        Returns:
            相似度得分（0.0 - 1.0）
        """
        if not expected or not actual:
            return 0.0
        
        return SequenceMatcher(None, expected.lower(), actual.lower()).ratio()
    
    def semantic_similarity(self, text1: str, text2: str) -> float:
        """
        语义相似度评估
        
        使用embedding模型计算语义相似度
        
        Args:
            text1: 文本1
            text2: 文本2
            
        Returns:
            相似度得分（0.0 - 1.0）
        """
        if not text1 or not text2:
            return 0.0
        
        # 如果没有embedding模型，使用模糊匹配
        if self.embedding_model is None:
            logger.debug("使用模糊匹配作为备选方案")
            return self.fuzzy_match(text1, text2)
        
        try:
            # 编码文本
            embeddings = self.embedding_model.encode([text1, text2])
            
            # 计算余弦相似度
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            
            return float(similarity)
        except Exception as e:
            logger.error(f"语义相似度计算失败: {e}")
            return self.fuzzy_match(text1, text2)
    
    def evaluate_intent(self, expected_intent: str, actual_intent: str) -> Tuple[bool, float]:
        """
        意图识别评估
        
        Args:
            expected_intent: 期望意图
            actual_intent: 实际意图
            
        Returns:
            (是否匹配, 得分)
        """
        if not expected_intent or not actual_intent:
            return False, 0.0
        
        # 完全匹配
        if expected_intent.lower() == actual_intent.lower():
            return True, 1.0
        
        # 部分匹配（包含关系）
        if expected_intent.lower() in actual_intent.lower() or \
           actual_intent.lower() in expected_intent.lower():
            return True, 0.8
        
        return False, 0.0
    
    def evaluate_slots(self, expected_slots: Dict, actual_slots: Dict) -> Tuple[Dict, float]:
        """
        槽位填充评估
        
        Args:
            expected_slots: 期望槽位
            actual_slots: 实际槽位
            
        Returns:
            (槽位匹配详情, 得分)
        """
        if not expected_slots:
            return {}, 1.0
        
        if not actual_slots:
            return {k: False for k in expected_slots}, 0.0
        
        slot_details = {}
        match_count = 0
        
        for slot_name, expected_value in expected_slots.items():
            actual_value = actual_slots.get(slot_name)
            
            if actual_value is not None:
                # 值匹配
                if str(expected_value).lower() == str(actual_value).lower():
                    slot_details[slot_name] = {
                        "expected": expected_value,
                        "actual": actual_value,
                        "match": True
                    }
                    match_count += 1
                else:
                    slot_details[slot_name] = {
                        "expected": expected_value,
                        "actual": actual_value,
                        "match": False
                    }
            else:
                slot_details[slot_name] = {
                    "expected": expected_value,
                    "actual": None,
                    "match": False
                }
        
        score = match_count / len(expected_slots) if expected_slots else 1.0
        return slot_details, score
    
    def evaluate_answer(self, expected_answer: str, actual_answer: str) -> float:
        """
        答案匹配度评估
        
        综合使用关键词匹配、模糊匹配和语义相似度
        
        Args:
            expected_answer: 期望答案
            actual_answer: 实际答案
            
        Returns:
            匹配度得分（0.0 - 1.0）
        """
        if not expected_answer or not actual_answer:
            return 0.0
        
        # 1. 精确匹配
        if expected_answer.lower() == actual_answer.lower():
            return 1.0
        
        # 2. 计算多种相似度
        fuzzy_score = self.fuzzy_match(expected_answer, actual_answer)
        
        # 3. 语义相似度（有模型时）
        semantic_score = self.semantic_similarity(expected_answer, actual_answer)
        
        # 4. 综合得分（加权平均）
        # 如果有语义模型，给语义相似度更高权重
        if self.embedding_model is not None:
            answer_score = 0.3 * fuzzy_score + 0.7 * semantic_score
        else:
            answer_score = fuzzy_score
        
        return answer_score
    
    # ============================================
    # 完整评估
    # ============================================
    
    def evaluate_single(self, 
                       query: str,
                       expected_intent: str,
                       actual_intent: str,
                       expected_answer: str = None,
                       actual_answer: str = None,
                       expected_slots: Dict = None,
                       actual_slots: Dict = None,
                       case_id: str = None,
                       response_time: float = 0.0) -> EvaluationResult:
        """
        单条测试用例完整评估
        
        Args:
            query: 用户问题
            expected_intent: 期望意图
            actual_intent: 实际意图
            expected_answer: 期望答案
            actual_answer: 实际答案
            expected_slots: 期望槽位
            actual_slots: 实际槽位
            case_id: 用例ID
            response_time: 响应时间
            
        Returns:
            EvaluationResult 评估结果
        """
        result = EvaluationResult(
            case_id=case_id or "",
            query=query,
            expected=expected_answer or "",
            actual=actual_answer or "",
            intent_expected=expected_intent,
            intent_actual=actual_intent,
            response_time=response_time
        )
        
        # 评估意图
        intent_match, intent_score = self.evaluate_intent(result.intent_expected, result.intent_actual)
        result.intent_match = intent_match
        result.intent_score = intent_score
        
        # 评估答案（如果有）
        if expected_answer and actual_answer:
            result.similarity_score = self.semantic_similarity(expected_answer, actual_answer)
            result.answer_score = self.evaluate_answer(expected_answer, actual_answer)
        else:
            result.answer_score = 0.0
            result.similarity_score = 0.0
        
        # 评估槽位（如果有）
        if expected_slots:
            slot_details, slot_score = self.evaluate_slots(expected_slots, actual_slots or {})
            result.slots_filled = slot_details
            result.slots_expected = expected_slots
            # 槽位得分暂时不计入总分
        else:
            result.slots_filled = actual_slots or {}
        
        # 计算综合得分
        # 意图占比60%，答案占比40%
        if expected_answer and actual_answer:
            result.overall_score = 0.6 * result.intent_score + 0.4 * result.answer_score
        else:
            result.overall_score = result.intent_score
        
        # 判断是否通过
        result.passed = (
            result.intent_score >= self.intent_threshold and
            result.overall_score >= self.similarity_threshold
        )
        
        return result
    
    def batch_evaluate(self, 
                      test_cases: List[Dict],
                      adapter = None,
                      get_response_func = None) -> List[EvaluationResult]:
        """
        批量评估
        
        Args:
            test_cases: 测试用例列表
            adapter: 对话适配器（有get_response方法）
            get_response_func: 响应获取函数（优先级高于adapter）
            
        Returns:
            评估结果列表
        """
        results = []
        
        for i, case in enumerate(test_cases):
            case_id = case.get('id', f"case_{i}")
            query = case.get('query') or case.get('input', '')
            expected_intent = case.get('expected_intent', '')
            expected_answer = case.get('expected_answer', case.get('expected', ''))
            expected_slots = case.get('expected_slots', {})
            
            try:
                start_time = time.time()
                
                # 获取实际回复
                if get_response_func:
                    actual_response = get_response_func(query)
                elif adapter and hasattr(adapter, 'get_response'):
                    actual_response = adapter.get_response(query)
                else:
                    actual_response = case.get('actual_response', '')
                
                response_time = time.time() - start_time
                
                # 解析实际响应（需要根据实际接口返回格式调整）
                actual_intent = self._extract_intent(actual_response)
                actual_answer = self._extract_answer(actual_response)
                actual_slots = self._extract_slots(actual_response)
                
                # 评估
                result = self.evaluate_single(
                    query=query,
                    expected_intent=expected_intent,
                    actual_intent=actual_intent,
                    expected_answer=expected_answer,
                    actual_answer=actual_answer,
                    expected_slots=expected_slots,
                    actual_slots=actual_slots,
                    case_id=case_id,
                    response_time=response_time
                )
                
            except Exception as e:
                logger.error(f"评估用例 {case_id} 时出错: {e}")
                result = EvaluationResult(
                    case_id=case_id,
                    query=query,
                    expected=expected_answer,
                    actual="",
                    intent_expected=expected_intent,
                    intent_actual="",
                    error=str(e),
                    passed=False
                )
            
            results.append(result)
        
        return results
    
    def _extract_intent(self, response: Any) -> str:
        """从响应中提取意图"""
        if isinstance(response, dict):
            return response.get('intent', '') or response.get('action', '')
        if isinstance(response, str):
            # 可以添加解析逻辑
            return ""
        return str(response) if response else ""
    
    def _extract_answer(self, response: Any) -> str:
        """从响应中提取答案"""
        if isinstance(response, dict):
            return response.get('answer', '') or response.get('message', '') or response.get('reply', '')
        if isinstance(response, str):
            return response
        return str(response) if response else ""
    
    def _extract_slots(self, response: Any) -> Dict:
        """从响应中提取槽位"""
        if isinstance(response, dict):
            return response.get('slots', {}) or response.get('entities', {})
        return {}
    
    # ============================================
    # 统计方法
    # ============================================
    
    @staticmethod
    def calculate_stats(results: List[EvaluationResult]) -> Dict:
        """
        计算评估统计信息
        
        Args:
            results: 评估结果列表
            
        Returns:
            统计信息字典
        """
        if not results:
            return {}
        
        total = len(results)
        passed = sum(1for r in results if r.passed)
        
        intent_scores = [r.intent_score for r in results]
        answer_scores = [r.answer_score for r in results]
        overall_scores = [r.overall_score for r in results]
        response_times = [r.response_time for r in results]
        
        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total > 0 else 0,
            
            "intent_accuracy": sum(intent_scores) / len(intent_scores) if intent_scores else 0,
            "answer_match_rate": sum(answer_scores) / len(answer_scores) if answer_scores else 0,
            "overall_score": sum(overall_scores) / len(overall_scores) if overall_scores else 0,
            
            "avg_response_time": sum(response_times) / len(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0,
            "min_response_time": min(response_times) if response_times else 0,
            
            "intent_scores": intent_scores,
            "answer_scores": answer_scores,
            "overall_scores": overall_scores,
        }
    
    @staticmethod
    def export_results(results: List[EvaluationResult], output_path: str, format: str = 'json') -> bool:
        """
        导出评估结果
        
        Args:
            results: 评估结果列表
            output_path: 输出路径
            format: 输出格式（json, csv）
            
        Returns:
            是否成功
        """
        try:
            if format == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)
            elif format == 'csv':
                import csv
                with open(output_path, 'w', encoding='utf-8', newline='') as f:
                    if results:
                        fieldnames = ['case_id', 'query', 'expected', 'actual', 
                                    'intent_expected', 'intent_actual', 'intent_score',
                                    'answer_score', 'similarity_score', 'overall_score',
                                    'passed', 'response_time', 'error']
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        for r in results:
                            writer.writerow({
                                'case_id': r.case_id,
                                'query': r.query,
                                'expected': r.expected,
                                'actual': r.actual,
                                'intent_expected': r.intent_expected,
                                'intent_actual': r.intent_actual,
                                'intent_score': r.intent_score,
                                'answer_score': r.answer_score,
                                'similarity_score': r.similarity_score,
                                'overall_score': r.overall_score,
                                'passed': r.passed,
                                'response_time': r.response_time,
                                'error': r.error
                            })
            else:
                logger.error(f"不支持的导出格式: {format}")
                return False
            
            logger.info(f"评估结果已导出到: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"导出评估结果失败: {e}")
            return False


# 便捷函数
evaluate = Evaluator.evaluate
fuzzy_match = Evaluator.fuzzy_match
semantic_similarity = Evaluator.semantic_similarity
batch_evaluate = Evaluator.batch_evaluate
calculate_stats = Evaluator.calculate_stats
