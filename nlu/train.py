"""
意图识别模型训练脚本
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlu.intent_recognizer import IntentRecognizer
from pathlib import Path


def train():
    """训练意图识别模型"""
    
    # 创建识别器
    recognizer = IntentRecognizer()
    
    # 1. 加载语料
    corpus_path = Path(__file__).parent.parent / "data" / "corpus" / "test_corpus.jsonl"
    
    if not corpus_path.exists():
        print(f"错误: 语料文件不存在: {corpus_path}")
        return
    
    print("=" * 50)
    print("步骤1: 加载语料")
    print("=" * 50)
    recognizer.load_corpus(str(corpus_path))
    
    # 2. 构建词汇表
    print("\n" + "=" * 50)
    print("步骤2: 构建词汇表")
    print("=" * 50)
    recognizer.build_vocab()
    
    # 3. 训练模型
    print("\n" + "=" * 50)
    print("步骤3: 训练模型")
    print("=" * 50)
    recognizer.train_from_corpus()
    
    # 4. 添加一些常用关键词（可选，增强识别效果）
    print("\n" + "=" * 50)
    print("步骤4: 添加关键词规则")
    print("=" * 50)
    _add_keyword_rules(recognizer)
    
    # 5. 保存模型
    print("\n" + "=" * 50)
    print("步骤5: 保存模型")
    print("=" * 50)
    model_dir = Path(__file__).parent.parent / "nlu" / "models"
    model_path = model_dir / "intent_model.json"
    recognizer.save(str(model_path))
    
    print("\n" + "=" * 50)
    print("训练完成!")
    print("=" * 50)
    
    # 6. 展示意图统计
    print("\n意图统计:")
    for intent_name, intent_data in recognizer.intents.items():
        sample_count = len(intent_data.get('samples', []))
        print(f"  - {intent_name}: {sample_count} 个样本")


def _add_keyword_rules(recognizer: IntentRecognizer):
    """添加关键词规则（可选）"""
    
    # 退货相关
    recognizer.add_intent("refund", 
        keywords=["退货", "退款", "退钱", "退货款", "还钱", "return", "refund", "退"],
        patterns=[r"想.*退货", r"要.*退货", r"怎么.*退货", r"如何.*退货", r".*退货.*吗"]
    )
    
    # 换货相关
    recognizer.add_intent("exchange",
        keywords=["换货", "换", "换成", "换颜色", "换款式", "exchange", "换"],
        patterns=[r"想.*换", r"要.*换货", r"换.*颜色"]
    )
    
    # 物流查询
    recognizer.add_intent("order_tracking",
        keywords=["物流", "快递", "发货", "到哪", "到哪里", "物流信息", "tracking", "shipping", "快递"],
        patterns=[r"物流.*", r".*物流.*", r"快递.*", r"发货.*", r"到哪.*"]
    )
    
    # 订单查询
    recognizer.add_intent("order_inquiry",
        keywords=["订单", "查询订单", "订单号", "order", "订单"],
        patterns=[r"查.*订单", r"订单.*查", r"订单号.*"]
    )
    
    # 产品咨询
    recognizer.add_intent("product_inquiry",
        keywords=["产品", "商品", "有没有", "有货", "库存", "颜色", "款式", "尺寸", "product", "颜色"],
        patterns=[r"有.*货", r"有没有.*", r".*颜色.*", r"什么.*颜色"]
    )
    
    # 账户问题
    recognizer.add_intent("account",
        keywords=["密码", "账户", "账号", "登录", "注册", "忘记", "account", "password", "登录"],
        patterns=[r"密码.*忘记", r"忘记.*密码", r"登录.*", r".*登录.*"]
    )
    
    # 配送问题
    recognizer.add_intent("delivery_time",
        keywords=["配送", "送达", "送货", "什么时候", "几天", "delivery", "shipping", "送"],
        patterns=[r"什么.*时候.*送", r"多久.*送", r".*时候.*到"]
    )
    
    # 地址相关
    recognizer.add_intent("address",
        keywords=["地址", "收货地址", "修改地址", "address"],
        patterns=[r"修改.*地址", r"地址.*修改", r"收货地址"]
    )
    
    # 确认退货
    recognizer.add_intent("refund_confirm",
        keywords=["订单号", "确认退货", "可以退货"],
        patterns=[r"订单号.*退货", r".*订单.*退货"]
    )
    
    print("已添加关键词规则")


if __name__ == "__main__":
    train()
