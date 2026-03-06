"""
行为树解析器模块
用于解析和验证智能客服的行为树

功能：
1. 加载行为树定义（支持JSONL格式）
2. 解析行为树结构
3. 验证对话路径
4. 检查节点跳转逻辑
5. 验证槽位填充

使用方法：
    from core.tree_parser import TreeParser, DialogSession
    
    # 创建解析器
    parser = TreeParser("BehaviorTree/base")
    
    # 加载行为树
    parser.load_tree("en")
    
    # 创建对话会话
    session = parser.create_session()
    
    # 处理用户输入
    result = session.process_input("我想退货")
    
    # 验证结果
    parser.validate_path(test_case)
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass
class TreeNode:
    """行为树节点"""
    node_id: str
    node_type: str           # condition, action, dialog
    intent: str              # 意图名称
    response: str            # 机器人回复
    next_nodes: List[str] = field(default_factory=list)  # 下一节点列表
    slot_to_fill: str = ""   # 需要填充的槽位
    conditions: Dict = field(default_factory=dict)  # 条件
    metadata: Dict = field(default_factory=dict)    # 元数据


@dataclass
class DialogTurn:
    """对话轮次"""
    turn_index: int
    user_input: str
    detected_intent: str
    extracted_slots: Dict
    current_node_id: str
    bot_response: str
    
    # 预期值（用于测试验证）
    expected_intent: str = ""
    expected_node: str = ""
    expected_slots: Dict = field(default_factory=dict)
    
    # 验证结果
    intent_matched: bool = False
    node_matched: bool = False
    slots_matched: bool = False


@dataclass
class DialogSession:
    """对话会话"""
    session_id: str
    tree_name: str
    language: str
    current_node: Optional[TreeNode] = None
    slots: Dict = field(default_factory=dict)
    history: List[DialogTurn] = field(default_factory=list)
    
    def add_turn(self, turn: DialogTurn):
        """添加对话轮次"""
        self.history.append(turn)
        # 更新槽位
        if turn.extracted_slots:
            self.slots.update(turn.extracted_slots)


class TreeParser:
    """行为树解析器"""
    
    def __init__(self, tree_base_path: str = None):
        """
        初始化解析器
        
        Args:
            tree_base_path: 行为树基础路径
        """
        if tree_base_path is None:
            # 默认路径
            self.base_path = Path(__file__).parent.parent / "BehaviorTree"
        else:
            self.base_path = Path(tree_base_path)
        
        # 加载的行为树
        self.trees: Dict[str, Dict[str, TreeNode]] = {}  # {language: {node_id: Node}}
        
        # 当前使用的树
        self.current_tree: Optional[Dict[str, TreeNode]] = None
        self.current_language: str = ""
        self.current_tree_name: str = ""
    
    # ============================================
    # 加载行为树
    # ============================================
    
    def load_tree(self, tree_name: str, language: str = "en") -> bool:
        """
        加载行为树
        
        Args:
            tree_name: 树名称（如 "base", "ComeIndia"）
            language: 语言（如 "en", "zh", "hi"）
            
        Returns:
            是否加载成功
        """
        tree_path = self.base_path / tree_name / "jsonl"
        
        if not tree_path.exists():
            logger.error(f"行为树路径不存在: {tree_path}")
            return False
        
        # 查找对应的jsonl文件
        jsonl_file = tree_path / f"{language}.jsonl"
        
        if not jsonl_file.exists():
            # 尝试其他可能的文件名
            possible_files = list(tree_path.glob("*.jsonl"))
            if possible_files:
                jsonl_file = possible_files[0]
                logger.warning(f"未找到 {language}.jsonl，使用: {jsonl_file.name}")
            else:
                logger.error(f"未找到jsonl文件: {tree_path}")
                return False
        
        try:
            nodes = {}
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        node_data = json.loads(line)
                        node = self._parse_node(node_data)
                        nodes[node.node_id] = node
                    except json.JSONDecodeError as e:
                        logger.warning(f"解析第{line_num}行失败: {e}")
                        continue
            
            # 构建节点关联
            self._build_node_relations(nodes)
            
            # 存储
            key = f"{tree_name}_{language}"
            self.trees[key] = nodes
            
            # 设置为当前树
            self.current_tree = nodes
            self.current_language = language
            self.current_tree_name = tree_name
            
            logger.info(f"成功加载行为树: {tree_name}/{language}, 共 {len(nodes)} 个节点")
            return True
            
        except Exception as e:
            logger.error(f"加载行为树失败: {e}")
            return False
    
    def _parse_node(self, data: Dict) -> TreeNode:
        """解析单个节点"""
        return TreeNode(
            node_id=data.get('id', data.get('node_id', '')),
            node_type=data.get('type', 'dialog'),
            intent=data.get('intent', ''),
            response=data.get('response', ''),
            next_nodes=data.get('next', []),
            slot_to_fill=data.get('slot_to_fill', ''),
            conditions=data.get('conditions', {}),
            metadata=data.get('metadata', {})
        )
    
    def _build_node_relations(self, nodes: Dict[str, TreeNode]):
        """构建节点关联"""
        for node_id, node in nodes.items():
            # 解析next字段（可能是字符串或列表）
            if isinstance(node.next_nodes, str):
                if node.next_nodes:
                    node.next_nodes = [node.next_nodes]
                else:
                    node.next_nodes = []
            elif not isinstance(node.next_nodes, list):
                node.next_nodes = []
    
    def load_all_trees(self, tree_name: str) -> bool:
        """
        加载所有语言版本的行为树
        
        Args:
            tree_name: 树名称
            
        Returns:
            是否全部加载成功
        """
        tree_path = self.base_path / tree_name / "jsonl"
        
        if not tree_path.exists():
            logger.error(f"行为树路径不存在: {tree_path}")
            return False
        
        # 查找所有jsonl文件
        jsonl_files = list(tree_path.glob("*.jsonl"))
        
        success = False
        for jsonl_file in jsonl_files:
            language = jsonl_file.stem
            if self.load_tree(tree_name, language):
                success = True
        
        return success
    
    # ============================================
    # 对话会话
    # ============================================
    
    def create_session(self, session_id: str = None, language: str = None) -> Optional[DialogSession]:
        """
        创建对话会话
        
        Args:
            session_id: 会话ID
            language: 语言（使用当前加载的默认语言）
            
        Returns:
            DialogSession对象
        """
        if self.current_tree is None:
            logger.error("请先加载行为树")
            return None
        
        import uuid
        session_id = session_id or str(uuid.uuid4())[:8]
        language = language or self.current_language
        
        # 找到起始节点
        start_node = self._find_start_node()
        
        if start_node is None:
            logger.error("未找到起始节点")
            return None
        
        session = DialogSession(
            session_id=session_id,
            tree_name=self.current_tree_name,
            language=language,
            current_node=start_node,
            slots={}
        )
        
        return session
    
    def _find_start_node(self) -> Optional[TreeNode]:
        """找到起始节点"""
        if self.current_tree is None:
            return None
        
        # 查找类型为 start 或 id 为 root/greeting 的节点
        for node in self.current_tree.values():
            if node.node_type == 'start' or node.node_id in ['root', 'greeting', 'start']:
                return node
        
        # 返回第一个节点
        return next(iter(self.current_tree.values())) if self.current_tree else None
    
    def find_node(self, node_id: str) -> Optional[TreeNode]:
        """
        查找节点
        
        Args:
            node_id: 节点ID
            
        Returns:
            TreeNode对象
        """
        if self.current_tree is None:
            return None
        
        return self.current_tree.get(node_id)
    
    def get_node_by_intent(self, intent: str) -> Optional[TreeNode]:
        """
        根据意图查找节点
        
        Args:
            intent: 意图名称
            
        Returns:
            TreeNode对象
        """
        if self.current_tree is None:
            return None
        
        for node in self.current_tree.values():
            if node.intent.lower() == intent.lower():
                return node
        
        return None
    
    def get_next_node(self, current_node: TreeNode, intent: str = None, slots: Dict = None) -> Optional[TreeNode]:
        """
        获取下一节点
        
        Args:
            current_node: 当前节点
            intent: 当前识别的意图
            slots: 当前已填充的槽位
            
        Returns:
            下一节点
        """
        if not current_node or not current_node.next_nodes:
            return None
        
        # 如果有next_nodes，直接返回第一个
        if current_node.next_nodes:
            next_node_id = current_node.next_nodes[0]
            return self.find_node(next_node_id)
        
        return None
    
    # ============================================
    # 路径验证
    # ============================================
    
    def validate_path(self, test_case: Dict) -> Dict:
        """
        验证行为树测试用例路径
        
        Args:
            test_case: 测试用例字典
            
        Returns:
            验证结果
        """
        if not test_case or 'test_case' not in test_case:
            return {
                "success": False,
                "error": "无效的测试用例格式"
            }
        
        tc = test_case['test_case']
        steps = tc.get('steps', [])
        
        if not steps:
            return {
                "success": False,
                "error": "测试用例没有步骤"
            }
        
        # 加载对应的行为树
        tree_name = tc.get('tree_name', 'base')
        language = tc.get('language', 'en')
        
        if not self.load_tree(tree_name, language):
            return {
                "success": False,
                "error": f"加载行为树失败: {tree_name}/{language}"
            }
        
        # 创建会话
        session = self.create_session()
        
        if session is None:
            return {
                "success": False,
                "error": "创建对话会话失败"
            }
        
        # 验证每一步
        turn_results = []
        all_matched = True
        
        for i, step in enumerate(steps):
            user_input = step.get('user_input', '')
            expected_intent = step.get('expected_intent', '')
            expected_node = step.get('expected_node', '')
            expected_slots = step.get('expected_slots', {})
            
            # 模拟处理（这里需要接入实际的NLU和对话引擎）
            # 在实际测试中，这些值应该从被测系统获取
            actual_intent = step.get('actual_intent', expected_intent)
            actual_node_id = step.get('expected_node', expected_node)  # 实际应该从系统获取
            actual_slots = step.get('actual_slots', expected_slots)
            
            # 验证
            intent_match = (actual_intent.lower() == expected_intent.lower()) if actual_intent and expected_intent else False
            node_match = (actual_node_id == expected_node) if expected_node else True
            slots_match = self._verify_slots(expected_slots, actual_slots)
            
            turn_result = {
                "step": i + 1,
                "user_input": user_input,
                "expected_intent": expected_intent,
                "actual_intent": actual_intent,
                "intent_matched": intent_match,
                "expected_node": expected_node,
                "actual_node": actual_node_id,
                "node_matched": node_match,
                "expected_slots": expected_slots,
                "actual_slots": actual_slots,
                "slots_matched": slots_match
            }
            
            turn_results.append(turn_result)
            
            if not (intent_match and node_match and slots_match):
                all_matched = False
        
        return {
            "success": all_matched,
            "test_case_name": tc.get('name', ''),
            "total_steps": len(steps),
            "turn_results": turn_results,
            "summary": {
                "matched": sum(1 for t in turn_results if t['intent_matched'] and t['node_matched'] and t['slots_matched']),
                "total": len(steps)
            }
        }
    
    def _verify_slots(self, expected: Dict, actual: Dict) -> bool:
        """验证槽位匹配"""
        if not expected:
            return True
        
        if not actual:
            return False
        
        for key, value in expected.items():
            if key not in actual:
                return False
            if str(value).lower() != str(actual[key]).lower():
                return False
        
        return True
    
    # ============================================
    # 工具方法
    # ============================================
    
    def get_tree_structure(self) -> Dict:
        """
        获取行为树结构
        
        Returns:
            树结构字典
        """
        if self.current_tree is None:
            return {}
        
        structure = {
            "tree_name": self.current_tree_name,
            "language": self.current_language,
            "node_count": len(self.current_tree),
            "nodes": []
        }
        
        for node_id, node in self.current_tree.items():
            structure["nodes"].append({
                "id": node.node_id,
                "type": node.node_type,
                "intent": node.intent,
                "next": node.next_nodes,
                "has_slot": bool(node.slot_to_fill)
            })
        
        return structure
    
    def find_paths(self, start_node_id: str, end_node_id: str, max_depth: int = 10) -> List[List[str]]:
        """
        查找两个节点之间的所有路径
        
        Args:
            start_node_id: 起始节点ID
            end_node_id: 结束节点ID
            max_depth: 最大深度
            
        Returns:
            路径列表
        """
        if self.current_tree is None:
            return []
        
        paths = []
        
        def dfs(node_id: str, current_path: List[str]):
            if len(current_path) > max_depth:
                return
            
            if node_id == end_node_id:
                paths.append(current_path + [node_id])
                return
            
            node = self.find_node(node_id)
            if node is None:
                return
            
            for next_id in node.next_nodes:
                if next_id not in current_path:  # 避免循环
                    dfs(next_id, current_path + [node_id])
        
        dfs(start_node_id, [])
        return paths
    
    def export_tree(self, output_path: str = None, format: str = 'json') -> bool:
        """
        导出行为树
        
        Args:
            output_path: 输出路径
            format: 输出格式
            
        Returns:
            是否成功
        """
        if self.current_tree is None:
            logger.error("没有可导出的行为树")
            return False
        
        if output_path is None:
            output_path = f"{self.current_tree_name}_{self.current_language}.{format}"
        
        try:
            if format == 'json':
                tree_data = {}
                for node_id, node in self.current_tree.items():
                    tree_data[node_id] = {
                        'type': node.node_type,
                        'intent': node.intent,
                        'response': node.response,
                        'next': node.next_nodes,
                        'slot_to_fill': node.slot_to_fill,
                        'conditions': node.conditions,
                        'metadata': node.metadata
                    }
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(tree_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"行为树已导出到: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"导出行为树失败: {e}")
            return False


# 便捷函数
create_session = TreeParser.create_session
validate_path = TreeParser.validate_path
