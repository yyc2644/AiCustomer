"""
意图识别器模块
从零开始实现的简单意图识别系统
使用关键词匹配 + TF-IDF 余弦相似度
"""

import json
import re
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
import math


class IntentRecognizer:
    """意图识别器"""
    
    def __init__(self):
        """初始化意图识别器"""
        self.intents = {}  # 意图字典: {intent_name: {keywords: [], patterns: [], responses: []}}
        self.corpus = []   # 训练语料: [{query, intent, language}, ...]
        self.vocab = set() # 词汇表
        self.idf = {}      # IDF权重
        
    # ============================================
    # 基础方法
    # ============================================
    
    def load_corpus(self, corpus_path: str) -> int:
        """
        加载训练语料
        
        Args:
            corpus_path: 语料文件路径 (JSONL格式)
            
        Returns:
            加载的语料数量
        """
        self.corpus = []
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    self.corpus.append(item)
        
        print(f"已加载 {len(self.corpus)} 条语料")
        return len(self.corpus)
    
    def build_vocab(self):
        """构建词汇表"""
        self.vocab = set()
        
        for item in self.corpus:
            query = self._preprocess(item.get('query', ''))
            self.vocab.update(query)
        
        # 构建IDF
        self._compute_idf()
        
        print(f"词汇表大小: {len(self.vocab)}")
    
    def _preprocess(self, text: str) -> List[str]:
        """
        文本预处理
        
        Args:
            text: 输入文本
            
        Returns:
            分词后的词列表
        """
        if not text:
            return []
        
        # 转小写
        text = text.lower()
        
        # 去除标点符号，保留中文、英文、数字
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
        
        # 分词（简单按空格分词 + 单字）
        words = text.split()
        
        # 添加单字（对中文有帮助）
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                words.append(char)
        
        # 去除停用词
        stopwords = {'的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        words = [w for w in words if w not in stopwords and len(w) > 0]
        
        return words
    
    def _compute_idf(self):
        """计算IDF权重"""
        if not self.corpus:
            return
            
        doc_count = len(self.corpus)
        word_doc_count = Counter()
        
        # 统计每个词出现在多少文档中
        for item in self.corpus:
            query = self._preprocess(item.get('query', ''))
            unique_words = set(query)
            for word in unique_words:
                word_doc_count[word] += 1
        
        # 计算IDF
        for word, count in word_doc_count.items():
            self.idf[word] = math.log((doc_count + 1) / (count + 1)) + 1
    
    # ============================================
    # 意图管理
    # ============================================
    
    def add_intent(self, intent_name: str, keywords: List[str] = None, 
                   patterns: List[str] = None, response: str = ""):
        """
        添加意图
        
        Args:
            intent_name: 意图名称
            keywords: 关键词列表
            patterns: 模式列表（正则表达式）
            response: 默认回复
        """
        self.intents[intent_name] = {
            'keywords': keywords or [],
            'patterns': patterns or [],
            'response': response,
            'samples': []  # 样本列表
        }
    
    def add_sample(self, intent_name: str, query: str):
        """
        添加训练样本
        
        Args:
            intent_name: 意图名称
            query: 用户查询
        """
        if intent_name not in self.intents:
            self.add_intent(intent_name)
        
        self.intents[intent_name]['samples'].append(query)
    
    def train_from_corpus(self):
        """从语料训练"""
        # 按意图分组样本
        intent_samples = {}
        
        for item in self.corpus:
            intent = item.get('expected_intent', '')
            query = item.get('query', '')
            
            if intent and query:
                if intent not in intent_samples:
                    intent_samples[intent] = []
                intent_samples[intent].append(query)
        
        # 添加到意图
        for intent_name, samples in intent_samples.items():
            for sample in samples:
                self.add_sample(intent_name, sample)
        
        print(f"训练完成，共 {len(intent_samples)} 个意图")
    
    # ============================================
    # 识别方法
    # ============================================
    
    def recognize(self, query: str) -> Tuple[str, float, Dict]:
        """
        识别意图
        
        Args:
            query: 用户查询
            
        Returns:
            (意图名称, 置信度, 详细信息)
        """
        if not self.intents:
            return ("unknown", 0.0, {"error": "模型未训练"})
        
        # 1. 关键词匹配
        keyword_result = self._match_keywords(query)
        
        # 2. 模式匹配
        pattern_result = self._match_patterns(query)
        
        # 3. 相似度匹配
        similarity_result = self._match_similarity(query)
        
        # 综合评分
        scores = {}
        details = {}
        
        for intent_name in self.intents:
            score = 0.0
            
            # 关键词匹配得分 (权重: 0.3)
            if intent_name in keyword_result:
                score += keyword_result[intent_name] * 0.3
                details[f"{intent_name}_keyword"] = keyword_result[intent_name]
            
            # 模式匹配得分 (权重: 0.3)
            if intent_name in pattern_result:
                score += pattern_result[intent_name] * 0.3
                details[f"{intent_name}_pattern"] = pattern_result[intent_name]
            
            # 相似度匹配得分 (权重: 0.4)
            if intent_name in similarity_result:
                score += similarity_result[intent_name] * 0.4
                details[f"{intent_name}_similarity"] = similarity_result[intent_name]
            
            scores[intent_name] = score
        
        # 选择最高分
        if not scores:
            return ("unknown", 0.0, {"error": "无法识别"})
        
        best_intent = max(scores, key=scores.get)
        best_score = scores[best_intent]
        
        return (best_intent, best_score, {
            "scores": scores,
            "details": details,
            "keyword_match": keyword_result,
            "pattern_match": pattern_result,
            "similarity_match": similarity_result
        })
    
    def _match_keywords(self, query: str) -> Dict[str, float]:
        """关键词匹配"""
        query = query.lower()
        results = {}
        
        for intent_name, intent_data in self.intents.items():
            keywords = intent_data.get('keywords', [])
            matched = 0
            
            for keyword in keywords:
                if keyword.lower() in query:
                    matched += 1
            
            if keywords:
                results[intent_name] = matched / len(keywords)
            else:
                results[intent_name] = 0.0
        
        return results
    
    def _match_patterns(self, query: str) -> Dict[str, float]:
        """模式匹配"""
        results = {}
        
        for intent_name, intent_data in self.intents.items():
            patterns = intent_data.get('patterns', [])
            matched = 0
            
            for pattern in patterns:
                if re.search(pattern, query):
                    matched += 1
            
            if patterns:
                results[intent_name] = matched / len(patterns)
            else:
                results[intent_name] = 0.0
        
        return results
    
    def _match_similarity(self, query: str) -> Dict[str, float]:
        """相似度匹配 (TF-IDF余弦相似度)"""
        query_vec = self._text_to_vector(query)
        
        if not query_vec:
            return {intent_name: 0.0 for intent_name in self.intents}
        
        results = {}
        
        for intent_name, intent_data in self.intents.items():
            samples = intent_data.get('samples', [])
            
            if not samples:
                results[intent_name] = 0.0
                continue
            
            # 计算查询与所有样本的相似度
            similarities = []
            for sample in samples:
                sample_vec = self._text_to_vector(sample)
                sim = self._cosine_similarity(query_vec, sample_vec)
                similarities.append(sim)
            
            # 取最大相似度
            results[intent_name] = max(similarities) if similarities else 0.0
        
        return results
    
    def _text_to_vector(self, text: str) -> Dict[str, float]:
        """将文本转换为TF-IDF向量"""
        words = self._preprocess(text)
        
        if not words:
            return {}
        
        # 计算TF
        tf = Counter(words)
        total = len(words)
        
        # 转换为TF-IDF
        vector = {}
        for word, count in tf.items():
            tf_value = count / total
            idf_value = self.idf.get(word, 1.0)
            vector[word] = tf_value * idf_value
        
        return vector
    
    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """计算余弦相似度"""
        if not vec1 or not vec2:
            return 0.0
        
        # 计算点积
        dot_product = 0
        for word, weight in vec1.items():
            if word in vec2:
                dot_product += weight * vec2[word]
        
        # 计算模长
        norm1 = math.sqrt(sum(w**2 for w in vec1.values()))
        norm2 = math.sqrt(sum(w**2 for w in vec2.values()))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    # ============================================
    # 模型保存和加载
    # ============================================
    
    def save(self, model_path: str):
        """
        保存模型
        
        Args:
            model_path: 模型保存路径
        """
        model_data = {
            'intents': self.intents,
            'vocab': list(self.vocab),
            'idf': self.idf
        }
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        with open(model_path, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)
        
        print(f"模型已保存至: {model_path}")
    
    def load(self, model_path: str):
        """
        加载模型
        
        Args:
            model_path: 模型路径
        """
        with open(model_path, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
        
        self.intents = model_data.get('intents', {})
        self.vocab = set(model_data.get('vocab', []))
        self.idf = model_data.get('idf', {})
        
        print(f"模型已加载: {len(self.intents)} 个意图")
    
    # ============================================
    # 评估方法
    # ============================================
    
    def evaluate(self, test_corpus: List[Dict]) -> Dict:
        """
        评估模型
        
        Args:
            test_corpus: 测试语料
            
        Returns:
            评估结果
        """
        correct = 0
        total = 0
        confusion = {}  # 混淆矩阵
        
        for item in test_corpus:
            query = item.get('query', '')
            expected = item.get('expected_intent', '')
            
            if not query or not expected:
                continue
            
            predicted, confidence, _ = self.recognize(query)
            
            total += 1
            if predicted == expected:
                correct += 1
            
            # 记录混淆矩阵
            key = f"{expected}->{predicted}"
            confusion[key] = confusion.get(key, 0) + 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'confusion': confusion
        }


# 便捷函数
def create_recognizer() -> IntentRecognizer:
    """创建意图识别器"""
    return IntentRecognizer()


def load_model(model_path: str) -> IntentRecognizer:
    """加载模型"""
    recognizer = IntentRecognizer()
    recognizer.load(model_path)
    return recognizer
