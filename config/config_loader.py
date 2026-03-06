"""
配置加载模块
用于统一加载和访问测试项目的所有配置文件

使用方法：
    from config.config_loader import Config
    
    # 加载所有配置
    config = Config()
    
    # 访问环境配置
    base_url = config.get('api.base_url')
    db_host = config.get('database.host')
    
    # 获取特定环境的配置
    test_env = config.get_env('test')
    
    # 获取评估阈值
    intent_threshold = config.get_evaluation_threshold('intent.top1')
"""

import os
import yaml
import logging
import logging.config
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv


class Config:
    """配置加载器类"""
    
    def __init__(self, env: Optional[str] = None):
        """
        初始化配置加载器
        
        Args:
            env: 指定环境名称，如 'dev', 'test', 'prod'。如果不指定，则从 env.yaml 读取 default 字段
        """
        # 获取项目根目录
        self.base_dir = Path(__file__).parent.parent
        self.config_dir = Path(__file__).parent
        
        # 加载环境变量
        load_dotenv()
        
        # 加载配置文件
        self._env_config = self._load_yaml('env.yaml')
        self._systems_config = self._load_yaml('systems.yaml')
        
        # 确定当前环境
        self.current_env = env or self._env_config.get('default', 'test')
        
        # 获取当前环境的配置
        self.env_config = self._env_config.get('environments', {}).get(self.current_env, {})
        
        # 初始化日志
        self._init_logging()
        
        # 合并所有配置
        self._config = {
            **self.env_config,
            'evaluation': self._systems_config.get('evaluation', {}),
            'model': self._systems_config.get('model', {}),
            'behavior_tree': self._systems_config.get('behavior_tree', {}),
            'knowledge_base': self._systems_config.get('knowledge_base', {}),
            'test_data': self._systems_config.get('test_data', {}),
            'reports': self._systems_config.get('reports', {}),
            'concurrency': self._env_config.get('concurrency', {}),
            'debug': self._env_config.get('debug', {}),
        }
        
        logging.info(f"配置加载完成，当前环境: {self.current_env}")
    
    def _load_yaml(self, filename: str) -> Dict:
        """加载 YAML 配置文件"""
        file_path = self.config_dir / filename
        if not file_path.exists():
            logging.warning(f"配置文件不存在: {file_path}")
            return {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logging.error(f"加载配置文件失败 {filename}: {e}")
            return {}
    
    def _init_logging(self):
        """初始化日志配置"""
        log_config_file = self.config_dir / 'logging.conf'
        if log_config_file.exists():
            try:
                logging.config.fileConfig(log_config_file, disable_existing_loggers=False)
            except Exception as e:
                # 如果日志配置失败，使用默认配置
                logging.basicConfig(
                    level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                logging.warning(f"日志配置加载失败，使用默认配置: {e}")
        else:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值，支持点号分隔的键
        
        Args:
            key: 配置键，如 'api.base_url'
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        # 处理环境变量引用，如 ${DB_PASSWORD}
        if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
            env_var = value[2:-1]
            return os.getenv(env_var, default)
        
        return value
    
    def get_env(self, env_name: str) -> Dict:
        """
        获取指定环境的配置
        
        Args:
            env_name: 环境名称，如 'dev', 'test', 'prod'
            
        Returns:
            环境配置字典
        """
        return self._env_config.get('environments', {}).get(env_name, {})
    
    def get_evaluation_threshold(self, threshold_type: str) -> float:
        """
        获取评估阈值
        
        Args:
            threshold_type: 阈值类型，如 'intent.top1', 'answer.similarity', 'slot.fill'
            
        Returns:
            阈值数值
        """
        return self.get(f'evaluation.{threshold_type}', 0.0)
    
    def get_test_data_path(self, data_type: str) -> str:
        """
        获取测试数据路径
        
        Args:
            data_type: 数据类型，如 'corpus', 'behavior_cases', 'ui_locators', 'reports'
            
        Returns:
            完整路径字符串
        """
        relative_path = self.get(f'test_data.{data_type}_dir', f'data/{data_type}')
        # 转换为绝对路径
        if not os.path.isabs(relative_path):
            return str(self.base_dir / relative_path)
        return relative_path
    
    def get_model_config(self, model_type: str = 'intent_model') -> Dict:
        """
        获取模型配置
        
        Args:
            model_type: 模型类型，如 'intent_model', 'embedding_model', 'llm'
            
        Returns:
            模型配置字典
        """
        return self.get(f'model.{model_type}', {})
    
    def get_behavior_tree_config(self) -> Dict:
        """获取行为树配置"""
        return self.get('behavior_tree', {})
    
    def get_knowledge_base_config(self) -> Dict:
        """获取知识库配置"""
        return self.get('knowledge_base', {})


# 创建全局配置实例
_config_instance: Optional[Config] = None


def get_config(env: Optional[str] = None) -> Config:
    """
    获取配置实例（单例模式）
    
    Args:
        env: 环境名称
        
    Returns:
        Config 实例
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(env)
    return _config_instance


# 便于直接导入使用
default_config = Config()
