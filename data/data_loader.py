"""
数据加载模块
用于统一加载测试项目中的各类测试数据

支持的格式：
- 语料：CSV, JSONL, XLSX
- 行为树用例：YAML, JSON
- UI定位器：YAML, JSON

使用方法：
    from data.data_loader import DataLoader
    
    # 加载语料测试集
    corpus = DataLoader.load_corpus("data/corpus/test_corpus.csv")
    
    # 加载行为树测试用例
    cases = DataLoader.load_behavior_cases("data/behavior_cases/refund_flow.yaml")
    
    # 加载UI定位器
    locators = DataLoader.load_ui_locators("data/ui/chat_window.yaml")
"""

import os
import json
import csv
import yaml
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from logging import getLogger


logger = getLogger(__name__)


class DataLoader:
    """数据加载器类"""
    
    # 项目根目录
    BASE_DIR = Path(__file__).parent.parent
    
    # 数据目录
    CORPUS_DIR = BASE_DIR / "data" / "corpus"
    BEHAVIOR_CASES_DIR = BASE_DIR / "data" / "behavior_cases"
    UI_DIR = BASE_DIR / "data" / "ui"
    
    @classmethod
    def _ensure_dir(cls, dir_path: Path) -> None:
        """确保目录存在"""
        if not dir_path.exists():
            logger.warning(f"数据目录不存在: {dir_path}")
    
    @classmethod
    def load_csv(cls, file_path: str) -> List[Dict[str, str]]:
        """
        加载CSV文件
        
        Args:
            file_path: CSV文件路径
            
        Returns:
            字典列表
        """
        file_path = cls.BASE_DIR / file_path if not os.path.isabs(file_path) else file_path
        
        if not Path(file_path).exists():
            logger.error(f"文件不存在: {file_path}")
            return []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                return list(reader)
        except Exception as e:
            logger.error(f"加载CSV文件失败: {e}")
            return []
    
    @classmethod
    def load_jsonl(cls, file_path: str) -> List[Dict[str, Any]]:
        """
        加载JSONL文件（每行一个JSON对象）
        
        Args:
            file_path: JSONL文件路径
            
        Returns:
            字典列表
        """
        file_path = cls.BASE_DIR / file_path if not os.path.isabs(file_path) else file_path
        
        if not Path(file_path).exists():
            logger.error(f"文件不存在: {file_path}")
            return []
        
        results = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        results.append(json.loads(line))
            return results
        except Exception as e:
            logger.error(f"加载JSONL文件失败: {e}")
            return []
    
    @classmethod
    def load_json(cls, file_path: str) -> Dict[str, Any]:
        """
        加载JSON文件
        
        Args:
            file_path: JSON文件路径
            
        Returns:
            字典对象
        """
        file_path = cls.BASE_DIR / file_path if not os.path.isabs(file_path) else file_path
        
        if not Path(file_path).exists():
            logger.error(f"文件不存在: {file_path}")
            return {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载JSON文件失败: {e}")
            return {}
    
    @classmethod
    def load_yaml(cls, file_path: str) -> Dict[str, Any]:
        """
        加载YAML文件
        
        Args:
            file_path: YAML文件路径
            
        Returns:
            字典对象
        """
        file_path = cls.BASE_DIR / file_path if not os.path.isabs(file_path) else file_path
        
        if not Path(file_path).exists():
            logger.error(f"文件不存在: {file_path}")
            return {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"加载YAML文件失败: {e}")
            return {}
    
    @classmethod
    def load_excel(cls, file_path: str, sheet_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        加载Excel文件
        
        Args:
            file_path: Excel文件路径
            sheet_name: 工作表名称，默认第一个
            
        Returns:
            字典列表
        """
        file_path = cls.BASE_DIR / file_path if not os.path.isabs(file_path) else file_path
        
        if not Path(file_path).exists():
            logger.error(f"文件不存在: {file_path}")
            return []
        
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            return df.to_dict('records')
        except Exception as e:
            logger.error(f"加载Excel文件失败: {e}")
            return []
    
    # ============================================
    # 高级加载方法
    # ============================================
    
    @classmethod
    def load_corpus(cls, file_path: str) -> List[Dict[str, Any]]:
        """
        加载语料测试集（自动识别格式）
        
        Args:
            file_path: 语料文件路径
            
        Returns:
            语料列表
        """
        ext = Path(file_path).suffix.lower()
        
        if ext == '.csv':
            return cls.load_csv(file_path)
        elif ext == '.jsonl':
            return cls.load_jsonl(file_path)
        elif ext in ['.xlsx', '.xls']:
            return cls.load_excel(file_path)
        else:
            logger.error(f"不支持的语料格式: {ext}")
            return []
    
    @classmethod
    def load_corpus_by_name(cls, name: str) -> List[Dict[str, Any]]:
        """
        根据名称加载语料文件
        
        Args:
            name: 文件名（不含路径和扩展名）
            
        Returns:
            语料列表
        """
        # 尝试多种扩展名
        for ext in ['.csv', '.jsonl', '.xlsx', '.xls']:
            file_path = f"data/corpus/{name}{ext}"
            data = cls.load_corpus(file_path)
            if data:
                logger.info(f"加载语料文件: {file_path}, 共 {len(data)} 条")
                return data
        
        logger.error(f"未找到语料文件: {name}")
        return []
    
    @classmethod
    def load_all_corpus(cls) -> List[Dict[str, Any]]:
        """
        加载所有语料文件
        
        Returns:
            所有语料的合并列表
        """
        all_corpus = []
        
        if not cls.CORPUS_DIR.exists():
            logger.warning(f"语料目录不存在: {cls.CORPUS_DIR}")
            return all_corpus
        
        # 遍历目录下的所有支持的文件
        for ext in ['*.csv', '*.jsonl', '*.xlsx', '*.xls']:
            for file_path in cls.CORPUS_DIR.glob(ext):
                data = cls.load_corpus(str(file_path.relative_to(cls.BASE_DIR)))
                all_corpus.extend(data)
        
        logger.info(f"共加载语料 {len(all_corpus)} 条")
        return all_corpus
    
    @classmethod
    def load_behavior_cases(cls, file_path: str) -> Dict[str, Any]:
        """
        加载行为树测试用例
        
        Args:
            file_path: 用例文件路径
            
        Returns:
            用例字典
        """
        ext = Path(file_path).suffix.lower()
        
        if ext in ['.yaml', '.yml']:
            return cls.load_yaml(file_path)
        elif ext == '.json':
            return cls.load_json(file_path)
        else:
            logger.error(f"不支持的行为树用例格式: {ext}")
            return {}
    
    @classmethod
    def load_behavior_case_by_name(cls, name: str) -> Dict[str, Any]:
        """
        根据名称加载行为树测试用例
        
        Args:
            name: 用例名称（不含路径和扩展名）
            
        Returns:
            用例字典
        """
        for ext in ['.yaml', '.yml', '.json']:
            file_path = f"data/behavior_cases/{name}{ext}"
            data = cls.load_behavior_cases(file_path)
            if data:
                logger.info(f"加载行为树用例: {file_path}")
                return data
        
        logger.error(f"未找到行为树用例: {name}")
        return {}
    
    @classmethod
    def load_all_behavior_cases(cls) -> List[Dict[str, Any]]:
        """
        加载所有行为树测试用例
        
        Returns:
            用例列表
        """
        all_cases = []
        
        if not cls.BEHAVIOR_CASES_DIR.exists():
            logger.warning(f"行为树用例目录不存在: {cls.BEHAVIOR_CASES_DIR}")
            return all_cases
        
        # 遍历目录下的所有支持的文件
        for ext in ['*.yaml', '*.yml', '*.json']:
            for file_path in cls.BEHAVIOR_CASES_DIR.glob(ext):
                data = cls.load_behavior_cases(str(file_path.relative_to(cls.BASE_DIR)))
                if data:
                    all_cases.append(data)
        
        logger.info(f"共加载行为树用例 {len(all_cases)} 个")
        return all_cases
    
    @classmethod
    def load_ui_locators(cls, file_path: str) -> Dict[str, Any]:
        """
        加载UI元素定位器
        
        Args:
            file_path: 定位器文件路径
            
        Returns:
            定位器字典
        """
        ext = Path(file_path).suffix.lower()
        
        if ext in ['.yaml', '.yml']:
            return cls.load_yaml(file_path)
        elif ext == '.json':
            return cls.load_json(file_path)
        else:
            logger.error(f"不支持的UI定位器格式: {ext}")
            return {}
    
    @classmethod
    def load_ui_locators_by_name(cls, name: str) -> Dict[str, Any]:
        """
        根据页面名称加载UI定位器
        
        Args:
            name: 页面名称（不含路径和扩展名）
            
        Returns:
            定位器字典
        """
        for ext in ['.yaml', '.yml', '.json']:
            file_path = f"data/ui/{name}{ext}"
            data = cls.load_ui_locators(file_path)
            if data:
                logger.info(f"加载UI定位器: {file_path}")
                return data
        
        logger.error(f"未找到UI定位器: {name}")
        return {}
    
    @classmethod
    def load_all_ui_locators(cls) -> Dict[str, Dict[str, Any]]:
        """
        加载所有UI定位器
        
        Returns:
            页面名称到定位器的映射字典
        """
        all_locators = {}
        
        if not cls.UI_DIR.exists():
            logger.warning(f"UI定位器目录不存在: {cls.UI_DIR}")
            return all_locators
        
        # 遍历目录下的所有支持的文件
        for ext in ['*.yaml', '*.yml', '*.json']:
            for file_path in cls.UI_DIR.glob(ext):
                data = cls.load_ui_locators(str(file_path.relative_to(cls.BASE_DIR)))
                if data and 'page_name' in data:
                    all_locators[data['page_name']] = data
        
        logger.info(f"共加载UI定位器 {len(all_locators)} 个页面")
        return all_locators


# 便捷函数
load_corpus = DataLoader.load_corpus
load_corpus_by_name = DataLoader.load_corpus_by_name
load_all_corpus = DataLoader.load_all_corpus
load_behavior_cases = DataLoader.load_behavior_cases
load_behavior_case_by_name = DataLoader.load_behavior_case_by_name
load_all_behavior_cases = DataLoader.load_all_behavior_cases
load_ui_locators = DataLoader.load_ui_locators
load_ui_locators_by_name = DataLoader.load_ui_locators_by_name
load_all_ui_locators = DataLoader.load_all_ui_locators
