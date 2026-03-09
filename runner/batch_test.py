"""
批量测试运行器
"""

from core.utils import load_testcases
from core.evaluator import Evaluator
from pathlib import Path
import json


class SimpleAdapter:
    """简单的对话适配器"""
    
    def __init__(self, api_client=None):
        self.api_client = api_client
    
    def get_response(self, query: str) -> dict:
        """
        获取对话响应
        
        Args:
            query: 用户输入
            
        Returns:
            包含intent和response的字典
        """
        if self.api_client:
            try:
                # 如果有API客户端，调用真实API
                response = self.api_client.send_message(message=query)
                return {
                    "intent": response.get("intent", ""),
                    "response": response.get("message", response.get("response", "")),
                    "slots": response.get("slots", {})
                }
            except Exception as e:
                return {"intent": "", "response": f"API调用失败: {e}", "slots": {}}
        else:
            # 没有API客户端，返回模拟响应
            return {
                "intent": "greeting",
                "response": "您好，请问有什么可以帮助您的？",
                "slots": {}
            }


def run_batch():
    """运行批量测试"""
    # 加载测试用例
    testcases = load_testcases()
    
    # 创建适配器
    adapter = SimpleAdapter()
    
    # 创建评估器
    evaluator = Evaluator()
    
    # 批量评估
    results = evaluator.batch_evaluate(testcases, adapter)
    
    # 转换结果为字典
    results_data = []
    for result in results:
        results_data.append({
            "case_id": result.case_id,
            "query": result.query,
            "expected": result.expected,
            "actual": result.actual,
            "intent_expected": result.intent_expected,
            "intent_actual": result.intent_actual,
            "intent_score": result.intent_score,
            "answer_score": result.answer_score,
            "similarity_score": result.similarity_score,
            "overall_score": result.overall_score,
            "intent_match": result.intent_match,
            "slots_filled": result.slots_filled,
            "slots_expected": result.slots_expected,
            "passed": result.passed,
            "response_time": result.response_time,
            "error": result.error
        })
    
    # 保存报告
    report_file = Path(__file__).parent.parent / "reports" / "summary.json"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)
    
    # 打印统计
    total = len(results_data)
    passed = sum(1 for r in results_data if r["passed"])
    print(f"测试完成: 共 {total} 条用例, 通过 {passed} 条, 失败 {total - passed} 条")
    print(f"报告已保存至: {report_file}")
    
    return results


if __name__ == "__main__":
    run_batch()
