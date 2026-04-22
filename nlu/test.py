"""
意图识别模型测试脚本
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlu.intent_recognizer import IntentRecognizer, load_model
from pathlib import Path


def test_recognize():
    """测试单条识别"""
    
    # 加载模型
    model_path = Path(__file__).parent.parent / "nlu" / "models" / "intent_model.json"
    
    if not model_path.exists():
        print("错误: 模型文件不存在，请先运行 train.py 训练模型")
        return
    
    recognizer = load_model(str(model_path))
    
    print("=" * 60)
    print("意图识别测试")
    print("=" * 60)
    
    # 测试用例
    test_queries = [
        "我想退货",
        "怎么申请退款",
        "查询订单物流",
        "这个产品有货吗",
        "密码忘记了怎么办",
        "什么时候能送到",
        "如何修改收货地址",
        "我想换成其他颜色",
        "订单号123456想退货",
        "我要查一下订单状态",
    ]
    
    print("\n识别结果:")
    print("-" * 60)
    
    for query in test_queries:
        intent, confidence, details = recognizer.recognize(query)
        
        print(f"\n输入: {query}")
        print(f"意图: {intent}")
        print(f"置信度: {confidence:.2%}")
        
        # 显示各意图得分
        if "scores" in details:
            scores = details["scores"]
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            print("得分排行:")
            for intent_name, score in sorted_scores[:3]:
                if score > 0:
                    print(f"  - {intent_name}: {score:.2%}")


def test_evaluate():
    """评估模型准确率"""
    
    # 加载模型
    model_path = Path(__file__).parent.parent / "nlu" / "models" / "intent_model.json"
    
    if not model_path.exists():
        print("错误: 模型文件不存在，请先运行 train.py 训练模型")
        return
    
    recognizer = load_model(str(model_path))
    
    # 加载测试语料
    corpus_path = Path(__file__).parent.parent / "data" / "corpus" / "test_corpus.jsonl"
    
    if not corpus_path.exists():
        print("错误: 语料文件不存在")
        return
    
    # 加载语料
    test_corpus = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                test_corpus.append(json.loads(line))
    
    # 评估
    print("=" * 60)
    print("模型评估")
    print("=" * 60)
    
    result = recognizer.evaluate(test_corpus)
    
    print(f"\n准确率: {result['accuracy']:.2%}")
    print(f"正确: {result['correct']} / {result['total']}")
    
    # 显示混淆矩阵
    print("\n混淆矩阵 (预期 -> 实际: 次数):")
    confusion = result.get('confusion', {})
    sorted_confusion = sorted(confusion.items(), key=lambda x: x[1], reverse=True)
    
    for key, count in sorted_confusion[:10]:
        expected, predicted = key.split("->")
        if expected != predicted:  # 只显示错误分类
            print(f"  {expected} -> {predicted}: {count}次")


def interactive_mode():
    """交互模式"""
    
    # 加载模型
    model_path = Path(__file__).parent.parent / "nlu" / "models" / "intent_model.json"
    
    if not model_path.exists():
        print("错误: 模型文件不存在，请先运行 train.py 训练模型")
        return
    
    recognizer = load_model(str(model_path))
    
    print("=" * 60)
    print("意图识别交互模式")
    print("输入 'quit' 或 'exit' 退出")
    print("=" * 60)
    
    while True:
        try:
            query = input("\n请输入: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("再见!")
                break
            
            intent, confidence, _ = recognizer.recognize(query)
            
            print(f"意图: {intent}")
            print(f"置信度: {confidence:.2%}")
            
        except KeyboardInterrupt:
            print("\n再见!")
            break
        except Exception as e:
            print(f"错误: {e}")


if __name__ == "__main__":
    import json
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "recognize":
            test_recognize()
        elif command == "evaluate":
            test_evaluate()
        elif command == "interactive":
            interactive_mode()
        else:
            print(f"未知命令: {command}")
            print("可用命令: recognize, evaluate, interactive")
    else:
        # 默认运行识别测试
        test_recognize()
