"""
API客户端模块
用于封装智能客服后台API调用

功能：
1. 统一处理认证（Token/Cookie）
2. 封装常用API接口
3. 自动重试和错误处理
4. 请求/响应日志记录

使用方法：
    from lib.api_client import APIClient
    
    # 创建客户端
    client = APIClient(env="test")
    
    # 登录获取Token
    client.login("admin", "password")
    
    # 调用API
    response = client.get("/api/knowledge/list")
    
    # 添加知识
    client.post("/api/knowledge/add", data={"title": "测试", "answer": "答案"})
"""

import os
import time
import json
import logging
from typing import Dict, Any, Optional, List
from urllib.parse import urljoin
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


logger = logging.getLogger(__name__)


class APIError(Exception):
    """API调用错误"""
    def __init__(self, message: str, status_code: int | None = None, response: Any | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response = response


class APIClient:
    """API客户端类"""
    
    def __init__(self, env: str = "test", config: Dict| None  = None):
        """
        初始化API客户端
        
        Args:
            env: 环境名称 (dev/test/prod)
            config: 配置字典
        """
        self.env = env
        self.config = config or {}
        
        # 获取环境配置
        from src.config.config_loader import get_config
        self.config_obj = get_config(env)
        
        # API配置
        self.base_url = self.config_obj.get('api.base_url', '')
        self.timeout = self.config_obj.get('api.timeout', 30)
        
        # 认证信息
        self.token = None
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # 会话
        self.session = self._create_session()
        
        logger.info(f"API客户端初始化完成，环境: {env}, URL: {self.base_url}")
    
    def _create_session(self) -> requests.Session:
        """创建带重试机制的会话"""
        session = requests.Session()
        
        # 配置重试策略
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def _get_full_url(self, endpoint: str) -> str:
        """获取完整的URL"""
        if endpoint.startswith('http'):
            return endpoint
        return urljoin(self.base_url, endpoint)
    
    def _update_headers(self):
        """更新请求头"""
        if self.token:
            self.headers["Authorization"] = f"Bearer {self.token}"
    
    # ============================================
    # 认证相关
    # ============================================
    
    def login(self, username: str, password: str) -> bool:
        """
        登录获取Token
        
        Args:
            username: 用户名
            password: 密码
            
        Returns:
            是否登录成功
        """
        try:
            login_url = self._get_full_url("/api/auth/login")
            
            response = self.session.post(
                login_url,
                json={"username": username, "password": password},
                headers=self.headers,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == 0:
                    self.token = data.get('data', {}).get('token')
                    self._update_headers()
                    logger.info(f"登录成功，用户: {username}")
                    return True
            
            logger.error(f"登录失败: {response.text}")
            return False
            
        except Exception as e:
            logger.error(f"登录异常: {e}")
            return False
    
    def logout(self):
        """退出登录"""
        self.token = None
        if "Authorization" in self.headers:
            del self.headers["Authorization"]
        logger.info("已退出登录")
    
    def refresh_token(self) -> bool:
        """
        刷新Token
        
        Returns:
            是否刷新成功
        """
        if not self.token:
            logger.warning("未登录，无法刷新Token")
            return False
        
        try:
            refresh_url = self._get_full_url("/api/auth/refresh")
            
            response = self.session.post(
                refresh_url,
                headers=self.headers,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == 0:
                    self.token = data.get('data', {}).get('token')
                    self._update_headers()
                    logger.info("Token刷新成功")
                    return True
            
            logger.error(f"Token刷新失败: {response.text}")
            return False
            
        except Exception as e:
            logger.error(f"Token刷新异常: {e}")
            return False
    
    # ============================================
    # 通用请求方法
    # ============================================
    
    def request(self, method: str, endpoint: str, params: Dict | None = None, 
                data: Any = None, headers: Dict | None = None, retry: int = 3) -> Dict:
        """
        通用请求方法
        
        Args:
            method: HTTP方法 (GET/POST/PUT/DELETE)
            endpoint: 接口路径
            params: URL参数
            data: 请求体数据
            headers: 额外请求头
            retry: 重试次数
            
        Returns:
            响应数据字典
            
        Raises:
            APIError: API调用错误
        """
        url = self._get_full_url(endpoint)
        
        # 合并请求头
        request_headers = {**self.headers}
        if headers:
            request_headers.update(headers)
        
        # 记录请求
        logger.debug(f"请求: {method} {url}")
        
        for attempt in range(retry):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=data,
                    headers=request_headers,
                    timeout=self.timeout
                )
                
                # 记录响应
                logger.debug(f"响应: {response.status_code} {response.text[:200]}")
                
                # 检查HTTP状态码
                response.raise_for_status()
                
                # 解析响应
                try:
                    result = response.json()
                except:
                    result = {"raw": response.text}
                
                # 检查业务状态码
                if isinstance(result, dict):
                    if result.get('code') == 0:
                        return result.get('data', result)
                    else:
                        raise APIError(
                            message=result.get('message', 'Unknown error'),
                            status_code=response.status_code,
                            response=result
                        )
                
                return result
                
            except requests.exceptions.Timeout:
                logger.warning(f"请求超时 (尝试 {attempt + 1}/{retry})")
                if attempt == retry - 1:
                    raise APIError("请求超时", status_code=408)
                    
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"连接错误 (尝试 {attempt + 1}/{retry}): {e}")
                if attempt == retry - 1:
                    raise APIError(f"连接失败: {e}", status_code=500)
                    
            except requests.exceptions.HTTPError as e:
                raise APIError(
                    message=str(e),
                    status_code=e.response.status_code if e.response else 500,
                    response=e.response.text if e.response else None
                )
            
            # 等待后重试
            if attempt < retry - 1:
                time.sleep(1 * (attempt + 1))
        
        raise APIError("请求失败", status_code=500)
    
    def get(self, endpoint: str, params: Dict| None  = None, **kwargs) -> Dict:
        """GET请求"""
        return self.request("GET", endpoint, params=params, **kwargs)
    
    def post(self, endpoint: str, data: Any = None, **kwargs) -> Dict:
        """POST请求"""
        return self.request("POST", endpoint, data=data, **kwargs)
    
    def put(self, endpoint: str, data: Any = None, **kwargs) -> Dict:
        """PUT请求"""
        return self.request("PUT", endpoint, data=data, **kwargs)
    
    def delete(self, endpoint: str, **kwargs) -> Dict:
        """DELETE请求"""
        return self.request("DELETE", endpoint, **kwargs)
    
    # ============================================
    # 业务API封装
    # ============================================
    
    # 知识库相关
    def get_knowledge_list(self, page: int = 1, page_size: int = 20, 
                          category: str | None = None, keyword: str | None = None) -> Dict:
        """获取知识库列表"""
        params = {"page": page, "page_size": page_size}
        if category:
            params["category"] = category
        if keyword:
            params["keyword"] = keyword
        
        return self.get("/api/knowledge/list", params)
    
    def get_knowledge_detail(self, knowledge_id: str) -> Dict:
        """获取知识详情"""
        return self.get(f"/api/knowledge/{knowledge_id}")
    
    def add_knowledge(self, title: str, content: str, category: str | None = None,
                     tags: List[str] | None = None) -> Dict:
        """添加知识"""
        data = {"title": title, "content": content}
        if category:
            data["category"] = category
        if tags:
            data["tags"] = tags
        
        return self.post("/api/knowledge/add", data=data)
    
    def update_knowledge(self, knowledge_id: str, **kwargs) -> Dict:
        """更新知识"""
        return self.put(f"/api/knowledge/{knowledge_id}", data=kwargs)
    
    def delete_knowledge(self, knowledge_id: str) -> Dict:
        """删除知识"""
        return self.delete(f"/api/knowledge/{knowledge_id}")
    
    # 会话相关
    def get_conversation_list(self, page: int = 1, page_size: int = 20,
                             status: str | None = None, start_date: str | None = None,
                             end_date: str | None = None) -> Dict:
        """获取会话列表"""
        params = {"page": page, "page_size": page_size}
        if status:
            params["status"] = status
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        
        return self.get("/api/conversation/list", params)
    
    def get_conversation_detail(self, conversation_id: str) -> Dict:
        """获取会话详情"""
        return self.get(f"/api/conversation/{conversation_id}")
    
    def transfer_to_human(self, conversation_id: str, reason: str | None = None) -> Dict:
        """转人工"""
        data = {"conversation_id": conversation_id}
        if reason:
            data["reason"] = reason
        
        return self.post("/api/conversation/transfer", data=data)
    
    def end_conversation(self, conversation_id: str) -> Dict:
        """结束会话"""
        return self.post(f"/api/conversation/{conversation_id}/end")
    
    # 客服相关
    def get_online_agents(self) -> Dict:
        """获取在线客服列表"""
        return self.get("/api/agents/online")
    
    def assign_agent(self, conversation_id: str, agent_id: str) -> Dict:
        """分配客服"""
        data = {"conversation_id": conversation_id, "agent_id": agent_id}
        return self.post("/api/agents/assign", data=data)
    
    # 统计相关
    def get_statistics(self, start_date: str, end_date: str,
                      metrics: List[str] | None = None) -> Dict:
        """获取统计数据"""
        params = {"start_date": start_date, "end_date": end_date}
        if metrics:
            params["metrics"] = ",".join(metrics)
        
        return self.get("/api/statistics", params)
    
    # 对话接口（核心）
    def send_message(self, message: str, session_id: str | None = None,
                    context: Dict | None = None) -> Dict:
        """
        发送消息（对话接口）
        
        Args:
            message: 用户消息
            session_id: 会话ID
            context: 上下文信息
            
        Returns:
            对话响应
        """
        data = {"message": message}
        if session_id:
            data["session_id"] = session_id
        if context:
            data["context"] = context
        
        return self.post("/api/chat/send", data=data)
    
    def get_chat_history(self, session_id: str, page: int = 1,
page_size: int = 50) -> Dict:
        """获取聊天历史"""
        params = {"session_id": session_id, "page": page, "page_size": page_size}
        return self.get("/api/chat/history", params)
    
    # ============================================
    # 工具方法
    # ============================================
    
    def test_connection(self) -> bool:
        """测试连接"""
        try:
            response = self.get("/api/health")
            return True
        except:
            return False
    
    def get_session_id(self) -> str:
        """创建新会话并返回session_id"""
        try:
            result = self.post("/api/chat/session/create")
            return result.get('session_id', '')
        except:
            return ''
    
    def close(self):
        """关闭会话"""
        self.session.close()
        logger.info("API客户端会话已关闭")


# 便捷函数
def create_client(env: str = "test", **kwargs) -> APIClient:
    """创建API客户端"""
    return APIClient(env=env, **kwargs)


# 默认客户端
_default_client = None


def get_client(env: str = "test") -> APIClient:
    """获取默认客户端（单例）"""
    global _default_client
    if _default_client is None:
        _default_client = APIClient(env)
    return _default_client
