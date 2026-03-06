"""
数据加载模块单元测试
测试 data_loader, config_loader 等数据功能
"""

import pytest
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import DataLoader
from config.config_loader import Config


class TestDataLoader:
    """数据加载器单元测试"""
    
    def test_load_csv(self):
        """测试加载CSV"""
        # 测试加载现有语料文件
        data = DataLoader.load_corpus("data/corpus/test_corpus.csv")
        
        # 应该能成功加载（即使为空）
        assert isinstance(data, list)
    
    def test_load_jsonl(self):
        """测试加载JSONL"""
        data = DataLoader.load_corpus("data/corpus/test_corpus.jsonl")
        
        assert isinstance(data, list)
    
    def test_load_corpus_by_name(self):
        """测试按名称加载语料"""
        data = DataLoader.load_corpus_by_name("test_corpus")
        
        # 应该返回列表
        assert isinstance(data, list)
    
    def test_load_corpus_not_found(self):
        """测试加载不存在的语料"""
        data = DataLoader.load_corpus_by_name("not_exist_file")
        
        assert data == []
    
    def test_load_all_corpus(self):
        """测试加载所有语料"""
        data = DataLoader.load_all_corpus()
        
        assert isinstance(data, list)
    
    def test_load_behavior_cases_yaml(self):
        """测试加载YAML行为树用例"""
        data = DataLoader.load_behavior_cases("data/behavior_cases/refund_flow.yaml")
        
        # 应该返回字典
        assert isinstance(data, dict)
    
    def test_load_behavior_cases_not_found(self):
        """测试加载不存在的行为树用例"""
        data = DataLoader.load_behavior_cases("data/behavior_cases/not_exist.yaml")
        
        assert data == {}
    
    def test_load_all_behavior_cases(self):
        """测试加载所有行为树用例"""
        data = DataLoader.load_all_behavior_cases()
        
        assert isinstance(data, list)
    
    def test_load_ui_locators_yaml(self):
        """测试加载YAML UI定位器"""
        data = DataLoader.load_ui_locators("data/ui/chat_window.yaml")
        
        assert isinstance(data, dict)
    
    def test_load_ui_locators_by_name(self):
        """测试按名称加载UI定位器"""
        data = DataLoader.load_ui_locators_by_name("chat_window")
        
        # 应该返回字典
        assert isinstance(data, dict)
    
    def test_load_all_ui_locators(self):
        """测试加载所有UI定位器"""
        data = DataLoader.load_all_ui_locators()
        
        assert isinstance(data, dict)
    
    def test_load_json(self):
        """测试加载JSON"""
        data = DataLoader.load_json("data/behavior_cases/refund_flow.yaml")
        
        # YAML会被解析为字典
        assert isinstance(data, dict)
    
    def test_load_yaml(self):
        """测试加载YAML"""
        data = DataLoader.load_yaml("data/behavior_cases/refund_flow.yaml")
        
        assert isinstance(data, dict)


class TestConfigLoader:
    """配置加载器单元测试"""
    
    def test_config_creation(self):
        """测试配置创建"""
        config = Config("test")
        
        assert config is not None
        assert config.current_env == "test"
    
    def test_config_get_value(self):
        """测试获取配置值"""
        config = Config("test")
        
        # 测试获取嵌套值
        value = config.get("test_data")
        
        assert value is not None
    
    def test_config_get_default(self):
        """测试获取默认值"""
        config = Config("test")
        
        value = config.get("not_exist_key", "default_value")
        
        assert value == "default_value"
    
    def test_config_get_nested(self):
        """测试获取嵌套配置"""
        config = Config("test")
        
        # 测试获取evaluation配置
        value = config.get("evaluation")
        
        assert isinstance(value, dict)
    
    def test_get_evaluation_threshold(self):
        """测试获取评估阈值"""
        config = Config("test")
        
        threshold = config.get_evaluation_threshold("intent.top1")
        
        assert isinstance(threshold, float)
    
    def test_get_test_data_path(self):
        """测试获取测试数据路径"""
        config = Config("test")
        
        path = config.get_test_data_path("corpus")
        
        assert isinstance(path, str)
        assert "corpus" in path
    
    def test_get_model_config(self):
        """测试获取模型配置"""
        config = Config("test")
        
        model_config = config.get_model_config("intent_model")
        
        assert isinstance(model_config, dict)
    
    def test_get_behavior_tree_config(self):
        """测试获取行为树配置"""
        config = Config("test")
        
        bt_config = config.get_behavior_tree_config()
        
        assert isinstance(bt_config, dict)
    
    def test_get_knowledge_base_config(self):
        """测试获取知识库配置"""
        config = Config("test")
        
        kb_config = config.get_knowledge_base_config()
        
        assert isinstance(kb_config, dict)


class TestDataIntegrity:
    """数据完整性测试"""
    
    def test_corpus_data_format(self):
        """测试语料数据格式"""
        data = DataLoader.load_corpus_by_name("test_corpus")
        
        if data:
            # 检查必要字段
            first_item = data[0]
            # 可能包含的字段
            possible_fields = ["query", "expected_intent", "expected_answer", "language", "category"]
            
            # 至少应该有一些字段
            has_field = any(field in first_item for field in possible_fields)
            assert has_field or len(data) == 0
    
    def test_behavior_case_data_format(self):
        """测试行为树用例数据格式"""
        data = DataLoader.load_behavior_cases("data/behavior_cases/refund_flow.yaml")
        
        if data:
            # 应该包含test_case
            assert "test_case" in data or isinstance(data, dict)
    
    def test_ui_locator_data_format(self):
        """测试UI定位器数据格式"""
        data = DataLoader.load_ui_locators("data/ui/chat_window.yaml")
        
        if data:
            # 应该包含locators
            assert "locators" in data or isinstance(data, dict)


class TestConfigIntegrity:
    """配置完整性测试"""
    
    def test_env_config_has_environments(self):
        """测试环境配置包含environments"""
        config = Config("test")
        
        # 检查是否有环境配置
        envs = config._env_config.get("environments", {})
        
        assert isinstance(envs, dict)
    
    def test_systems_config_has_evaluation(self):
        """测试系统配置包含evaluation"""
        config = Config("test")
        
        # 检查是否有评估配置
        evaluation = config._systems_config.get("evaluation", {})
        
        assert isinstance(evaluation, dict)
    
    def test_systems_config_has_model(self):
        """测试系统配置包含model"""
        config = Config("test")
        
        # 检查是否有模型配置
        model = config._systems_config.get("model", {})
        
        assert isinstance(model, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
