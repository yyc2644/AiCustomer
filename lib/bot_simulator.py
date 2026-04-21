"""
机器人模拟器模块
用于模拟用户与智能客服的对话

功能：
1. 模拟单轮对话
2. 模拟多轮对话流程
3. 支持行为树路径测试
4. 支持语料批量测试
5. 记录对话历史

使用方法：
    from lib.bot_simulator import BotSimulator, DialogTurn
    
    # 创建模拟器
    simulator = BotSimulator(api_client)
    
    # 单轮对话
    response = simulator.chat("我想退货")
    
    # 多轮对话
    history = simulator.chat_flow([
        "我想退货",
        "订单号是123456",
        "质量问题"
    ])
    
    # 使用语料库测试
    results = simulator.test_corpus(corpus_data)
"""

import time
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime


logger = logging.getLogger(__name__)


@dataclass
class DialogTurn:
    """对话轮次"""
    turn_index: int
    user_input: str
    bot_response: str
    intent: str = ""
    slots: Dict = field(default_factory=dict)
    node_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    response_time: float = 0.0
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "turn_index": self.turn_index,
            "user_input": self.user_input,
            "bot_response": self.bot_response,
            "intent": self.intent,
            "slots": self.slots,
            "node_id": self.node_id,
            "timestamp": self.timestamp.isoformat(),
            "response_time": self.response_time
        }


@dataclass
class ChatSession:
    """对话会话"""
    session_id: str
    user_id: str = ""
    language: str = "zh-CN"
    turns: List[DialogTurn] = field(default_factory=list)
    context: Dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    ended_at: Optional[datetime] = None
    
    def add_turn(self, turn: DialogTurn):
        """添加对话轮次"""
        self.turns.append(turn)
    
    def get_history(self) -> List[Dict]:
        """获取对话历史"""
        return [turn.to_dict() for turn in self.turns]
    
    def get_last_bot_response(self) -> str:
        """获取最后一条机器人回复"""
        if self.turns:
            return self.turns[-1].bot_response
        return ""
    
    def get_last_intent(self) -> str:
        """获取最后识别的意图"""
        if self.turns:
            return self.turns[-1].intent
        return ""
    
    def get_context(self) -> Dict:
        """获取上下文"""
        return self.context.copy()


class BotSimulator:
    """机器人模拟器"""
    
    def __init__(self, api_client=None, config: Dict | None = None):
        """
        初始化模拟器
        
        Args:
            api_client: API客户端实例
            config: 配置字典
        """
        self.api_client = api_client
        self.config = config or {}
        
        # 会话管理
        self.sessions: Dict[str, ChatSession] = {}
        
        # 当前会话
        self.current_session: Optional[ChatSession] = None
        
        logger.info("机器人模拟器初始化完成")
    
    def create_session(self, session_id: str| None = None, user_id: str = "",
                     language: str = "zh-CN") -> ChatSession:
        """
        创建新会话
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            language: 语言
            
        Returns:
            ChatSession对象
        """
        import uuid
        session_id = session_id or str(uuid.uuid4())[:12]
        
        session = ChatSession(
            session_id=session_id,
            user_id=user_id,
            language=language
        )
        
        self.sessions[session_id] = session
        self.current_session = session
        
        logger.info(f"创建会话: {session_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """获取会话"""
        return self.sessions.get(session_id)
    
    def switch_session(self, session_id: str) -> bool:
        """切换当前会话"""
        session = self.sessions.get(session_id)
        if session:
            self.current_session = session
            return True
        return False
    
    def chat(self, message: str, session_id: str | None= None,
            context: Dict| None = None) -> Dict:
        """
        单轮对话
        
        Args:
            message: 用户消息
            session_id: 会话ID
            context: 上下文
            
        Returns:
            对话响应字典
        """
        # 获取或创建会话
        if session_id:
            session = self.get_session(session_id)
            if not session:
                session = self.create_session(session_id)
            self.current_session = session
        elif self.current_session:
            session = self.current_session
        else:
            session = self.create_session()
        
        # 记录开始时间
        start_time = time.time()
        
        try:
            # 调用API
            if self.api_client:
                response = self.api_client.send_message(
                    message=message,
                    session_id=session.session_id,
                    context=context or session.get_context()
                )
            else:
                # 模拟响应（如果没有API客户端）
                response = self._mock_response(message, session)
            
            response_time = time.time() - start_time
            
            # 解析响应
            bot_response = self._extract_response(response)
            intent = self._extract_intent(response)
            slots = self._extract_slots(response)
            node_id = self._extract_node_id(response)
            
            # 创建对话轮次
            turn = DialogTurn(
                turn_index=len(session.turns) + 1,
                user_input=message,
                bot_response=bot_response,
                intent=intent,
                slots=slots,
                node_id=node_id,
                response_time=response_time
            )
            
            # 添加到会话
            session.add_turn(turn)
            
            # 更新上下文
            if slots:
                session.context.update(slots)
            
            return {
                "success": True,
                "session_id": session.session_id,
                "turn_index": turn.turn_index,
                "response": bot_response,
                "intent": intent,
                "slots": slots,
                "node_id": node_id,
                "response_time": response_time,
                "history": session.get_history()
            }
            
        except Exception as e:
            logger.error(f"对话异常: {e}")
            return {
                "success": False,
                "session_id": session.session_id,
                "error": str(e)
            }
    
    def chat_flow(self, messages: List[str], session_id: str | None= None,
                 context: Dict| None = None) -> Dict:
        """
        多轮对话流程
        
        Args:
            messages: 消息列表
            session_id: 会话ID
            context: 初始上下文
            
        Returns:
            对话结果
        """
        # 创建会话
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
        else:
            session = self.create_session(session_id)
        
        if context:
            session.context.update(context)
        
        results = []
        
        for i, message in enumerate(messages):
            result = self.chat(message)
            results.append(result)
            
            if not result.get("success"):
                break
        
        return {
            "session_id": session.session_id,
            "total_turns": len(results),
            "turns": results,
            "history": session.get_history(),
            "final_context": session.get_context()
        }
    
    def test_corpus(self, corpus_data: List[Dict],
                   progress_callback: Callable | None= None) -> List[Dict]:
        """
        使用语料库进行批量测试
        
        Args:
            corpus_data: 语料数据列表
            progress_callback: 进度回调函数
            
        Returns:
            测试结果列表
        """
        results = []
        total = len(corpus_data)
        
        for i, case in enumerate(corpus_data):
            query = case.get("query") or case.get("input", "")
            expected_intent = case.get("expected_intent", "")
            expected_answer = case.get("expected_answer", case.get("expected", ""))
            
            # 执行对话
            result = self.chat(query)
            
            # 评估结果
            from core.evaluator import Evaluator
            evaluator = Evaluator(self.config)
            
            eval_result = evaluator.evaluate_single(
                query=query,
                expected_intent=expected_intent,
                actual_intent=result.get("intent", ""),
                expected_answer=expected_answer,
                actual_answer=result.get("response", ""),
                case_id=case.get("id", f"case_{i}")
            )
            
            test_result = {
                "case_id": case.get("id", f"case_{i}"),
                "query": query,
                "expected_intent": expected_intent,
                "actual_intent": result.get("intent", ""),
                "intent_match": eval_result.intent_match,
                "expected_answer": expected_answer,
                "actual_answer": result.get("response", ""),
                "similarity": eval_result.similarity_score,
                "passed": eval_result.passed,
                "response_time": result.get("response_time", 0)
            }
            
            results.append(test_result)
            
            # 进度回调
            if progress_callback:
                progress_callback(i + 1, total, test_result)
        
        return results
    
    def test_behavior_tree(self, test_case: Dict,
                          validate: bool = True) -> Dict:
        """
        测试行为树路径
        
        Args:
            test_case: 测试用例（行为树格式）
            validate: 是否验证路径
            
        Returns:
            测试结果
        """
        from core.tree_parser import TreeParser
        
        # 加载行为树
        parser = TreeParser()
        
        tc = test_case.get("test_case", {})
        tree_name = tc.get("tree_name", "base")
        language = tc.get("language", "zh-CN")
        
        if not parser.load_tree(tree_name, language):
            return {
                "success": False,
                "error": f"加载行为树失败: {tree_name}/{language}"
            }
        
        # 创建会话
        session = parser.create_session()
        if not session:
            return {
                "success": False,
                "error": "创建对话会话失败"
            }
        
        # 执行每一步
        steps = tc.get("steps", [])
        results = []
        
        for step in steps:
            user_input = step.get("user_input", "")
            
            # 发送消息
            chat_result = self.chat(user_input, session.session_id)
            
            # 验证
            expected_intent = step.get("expected_intent", "")
            expected_node = step.get("expected_node", "")
            expected_slots = step.get("expected_slots", {})
            
            actual_intent = chat_result.get("intent", "")
            actual_node = chat_result.get("node_id", "")
            actual_slots = chat_result.get("slots", {})
            
            step_result = {
                "step": step.get("step", 0),
                "user_input": user_input,
                "expected_intent": expected_intent,
                "actual_intent": actual_intent,
                "intent_match": expected_intent.lower() == actual_intent.lower() if expected_intent and actual_intent else False,
                "expected_node": expected_node,
                "actual_node": actual_node,
                "node_match": expected_node == actual_node if expected_node else True,
                "expected_slots": expected_slots,
                "actual_slots": actual_slots,
                "slots_match": expected_slots == actual_slots if expected_slots else True
            }
            
            results.append(step_result)
        
        # 计算通过率
        matched = sum(1 for r in results if r["intent_match"] and r["node_match"])
        
        return {
            "success": matched == len(results),
            "test_case_name": tc.get("name", ""),
            "total_steps": len(results),
            "matched_steps": matched,
            "pass_rate": matched / len(results) if results else 0,
            "steps": results
        }
    
    # ============================================
    # 内部方法
    # ============================================
    
    def _mock_response(self, message: str, session: ChatSession) -> Dict:
        """模拟响应（当没有API客户端时使用）"""
        # 简单的模拟逻辑
        message_lower = message.lower()
        
        if "退货" in message_lower or "return" in message_lower:
            return {
                "response": "您好，请问您的订单是什么时候下单的呢？",
                "intent": "refund",
                "slots": {},
                "node_id": "refund_start"
            }
        elif "订单" in message_lower or "order" in message_lower:
            return {
                "response": "请提供您的订单号",
                "intent": "order_inquiry",
                "slots": {},
                "node_id": "order_ask_number"
            }
        else:
            return {
                "response": "您好，请问有什么可以帮助您的？",
                "intent": "greeting",
                "slots": {},
                "node_id": "greeting"
            }
    
    def _extract_response(self, response: Any) -> str:
        """提取回复内容"""
        if isinstance(response, dict):
            return response.get("response") or response.get("answer") or \
                   response.get("message") or response.get("reply", "")
        return str(response) if response else ""
    
    def _extract_intent(self, response: Any) -> str:
        """提取意图"""
        if isinstance(response, dict):
            return response.get("intent", "") or response.get("action", "")
        return ""
    
    def _extract_slots(self, response: Any) -> Dict:
        """提取槽位"""
        if isinstance(response, dict):
            return response.get("slots", {}) or response.get("entities", {})
        return {}
    
    def _extract_node_id(self, response: Any) -> str:
        """提取节点ID"""
        if isinstance(response, dict):
            return response.get("node_id", "") or response.get("current_node", "")
        return ""
    
    # ============================================
    # 工具方法
    # ============================================
    
    def export_history(self, session_id: str | None = None,
                      format: str = "json") -> str:
        """
        导出会话历史
        
        Args:
            session_id: 会话ID（默认当前会话）
            format: 导出格式
            
        Returns:
            导出内容
        """
        if session_id:
            session = self.sessions.get(session_id)
        else:
            session = self.current_session
        
        if not session:
            return ""
        
        if format == "json":
            import json
            return json.dumps(session.get_history(), ensure_ascii=False, indent=2)
        
        return str(session.get_history())
    
    def clear_sessions(self):
        """清除所有会话"""
        self.sessions.clear()
        self.current_session = None
        logger.info("所有会话已清除")


# 便捷函数
def create_simulator(api_client=None) -> BotSimulator:
    """创建模拟器"""
    return BotSimulator(api_client)
