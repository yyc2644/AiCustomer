from core.utils import load_testcases
from core.adapter.openai_adapter import OpenAIAdapter
from core.evaluator import Evaluator
from pathlib import Path
import json


def run_batch():
    adapter = OpenAIAdapter()
    testcases = load_testcases()
    results = Evaluator.batch_evaluate(testcases, adapter)

    # 保存报告
    report_file = Path(__file__).parent.parent / "reports" / "summary.json"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"测试完成，共 {len(results)} 条用例，报告已保存至 {report_file}")
    return results


if __name__ == "__main__":
    run_batch()
