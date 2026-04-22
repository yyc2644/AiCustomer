"""
数据库管理模块
用于连接和操作测试数据库

功能：
1. 数据库连接管理
2. 常用SQL操作封装
3. 测试数据准备和清理
4. 查询结果转换

使用方法：
    from lib.db_manager import DBManager
    
    # 创建数据库管理器
    db = DBManager(env="test")
    
    # 执行查询
    results = db.query("SELECT * FROM users WHERE id = %s", (1,))
    
    # 执行更新
    db.execute("UPDATE users SET name = %s WHERE id = %s", ("测试", 1))
    
    # 批量插入
    db.batch_insert("users", [{"name": "用户1"}, {"name": "用户2"}])
    
    # 获取会话记录
    conversations = db.get_conversations_by_user(user_id)
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from contextlib import contextmanager
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class DBConfig:
    """数据库配置"""
    host: str
    port: int
    database: str
    user: str
    password: str
    charset: str = "utf8mb4"


class DBManager:
    """数据库管理器"""
    
    def __init__(self, env: str = "test", config: Dict | None = None):
        """
        初始化数据库管理器
        
        Args:
            env: 环境名称
            config: 配置文件（可选）
        """
        self.env = env
        self.config = config
        
        # 获取数据库配置
        self.db_config = self._get_db_config()
        
        # 数据库连接
        self._connection = None
        self._cursor = None
        
        logger.info(f"数据库管理器初始化完成，环境: {env}")
    
    def _get_db_config(self) -> DBConfig:
        """获取数据库配置"""
        from src.config.config_loader import get_config
        
        config_obj = get_config(self.env)
        
        db_config = config_obj.get('database', {})
        
        return DBConfig(
            host=db_config.get('host', 'localhost'),
            port=db_config.get('port', 3306),
            database=db_config.get('name', 'customer_service_test'),
            user=db_config.get('user', 'root'),
            password=db_config.get('password', ''),
            charset=db_config.get('charset', 'utf8mb4')
        )
    
    def connect(self):
        """建立数据库连接"""
        try:
            import pymysql
            self._connection = pymysql.connect(
                host=self.db_config.host,
                port=self.db_config.port,
                user=self.db_config.user,
                password=self.db_config.password,
                database=self.db_config.database,
                charset=self.db_config.charset,
                cursorclass=pymysql.cursors.DictCursor
            )
            self._cursor = self._connection.cursor()
            logger.info(f"数据库连接成功: {self.db_config.database}")
        except ImportError:
            logger.warning("pymysql未安装，使用sqlite模式")
            self._use_sqlite = True
            self._connect_sqlite()
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            raise
    
    def _connect_sqlite(self):
        """连接SQLite（备选方案）"""
        import sqlite3
        db_path = f"{self.db_config.database}.db"
        self._connection = sqlite3.connect(db_path)
        self._connection.row_factory = sqlite3.Row
        self._cursor = self._connection.cursor()
        logger.info(f"SQLite连接成功: {db_path}")
    
    def close(self):
        """关闭数据库连接"""
        if self._cursor:
            self._cursor.close()
        if self._connection:
            self._connection.close()
        logger.info("数据库连接已关闭")
    
    @contextmanager
    def get_connection(self):
        """获取数据库连接（上下文管理器）"""
        try:
            if not self._connection or not self._connection.open:# type: ignore
                self.connect()
            yield self
        finally:
            pass
    
    # ============================================
    # 基础操作
    # ============================================
    
    def execute(self, sql: str, params: Tuple| None = None) -> int:
        """
        执行SQL语句
        
        Args:
            sql: SQL语句
            params: 参数元组
            
        Returns:
            影响行数
        """
        if not self._connection:
            self.connect()
        
        try:
            if params:
                self._cursor.execute(sql, params) # type: ignore
            else:
                self._cursor.execute(sql)# type: ignore
            
            self._connection.commit()# type: ignore
            return self._cursor.rowcount# type: ignore
            
        except Exception as e:
            self._connection.rollback()# type: ignore
            logger.error(f"执行SQL失败: {e}, SQL: {sql}")
            raise
    
    def query(self, sql: str, params: Tuple| None = None) -> List[Dict]:
        """
        查询数据
        w s w
        Args:
            sql: SQL语句
            params: 参数元组
            
        Returns:
            查询结果列表
        """
        if not self._connection:
            self.connect()
        
        try:
            if params:
                self._cursor.execute(sql, params)# type: ignore
            else:
                self._cursor.execute(sql)# type: ignore
            
            results = self._cursor.fetchall()# type: ignore
            
            # 转换为字典
            if results and hasattr(results[0], 'keys'):
                return [dict(row) for row in results]
            return results# type: ignore
            
        except Exception as e:
            logger.error(f"查询失败: {e}, SQL: {sql}")
            raise
    
    def query_one(self, sql: str, params: Tuple| None = None) -> Optional[Dict]:
        """查询单条数据"""
        results = self.query(sql, params)
        return results[0] if results else None
    
    def batch_execute(self, sql: str, params_list: List[Tuple]) -> int:
        """
        批量执行
        
        Args:
            sql: SQL语句
            params_list: 参数列表
            
        Returns:
            影响行数
        """
        if not self._connection:
            self.connect()
        
        try:
            self._cursor.executemany(sql, params_list)# type: ignore
            self._connection.commit()# type: ignore
            return self._cursor.rowcount# type: ignore
            
        except Exception as e:
            self._connection.rollback()# type: ignore
            logger.error(f"批量执行失败: {e}")
            raise
    
    # ============================================
    # 高级操作
    # ============================================
    
    def insert(self, table: str, data: Dict) -> int:
        """
        插入数据
        
        Args:
            table: 表名
            data: 数据字典
            
        Returns:
            插入ID
        """
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['%s'] * len(data))
        sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        
        self.execute(sql, tuple(data.values()))
        
        # 获取插入ID
        if hasattr(self._cursor, 'lastrowid'):
            return self._cursor.lastrowid# type: ignore
        return 0
    
    def batch_insert(self, table: str, data_list: List[Dict]) -> int:
        """
        批量插入
        
        Args:
            table: 表名
            data_list: 数据字典列表
            
        Returns:
            插入行数
        """
        if not data_list:
            return 0
        
        columns = ', '.join(data_list[0].keys())
        placeholders = ', '.join(['%s'] * len(data_list[0]))
        sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        
        params_list = [tuple(d.values()) for d in data_list]
        return self.batch_execute(sql, params_list)
    
    def update(self, table: str, data: Dict, where: str, 
              where_params: Tuple| None = None) -> int:
        """
        更新数据

        Args:
            table: 表名
            data: 更新数据
            where: WHERE条件
            where_params: 条件参数
            
        Returns:
            影响行数
        """
        set_clause = ', '.join([f"{k} = %s" for k in data.keys()])
        sql = f"UPDATE {table} SET {set_clause} WHERE {where}"
        
        params = tuple(data.values())
        if where_params:
            params = params + where_params
        
        return self.execute(sql, params)
    
    def delete(self, table: str, where: str, 
              where_params: Tuple| None = None) -> int:
        """
        删除数据
        
        Args:
            table: 表名
            where: WHERE条件
            where_params: 条件参数
            
        Returns:
            影响行数
        """
        sql = f"DELETE FROM {table} WHERE {where}"
        return self.execute(sql, where_params)
    
    # ============================================
    # 业务相关方法
    # ============================================
    
    def get_conversations(self, user_id: str | None = None, status: str | None = None,
                         limit: int = 100) -> List[Dict]:
        """
        获取会话列表
        
        Args:
            user_id: 用户ID
            status: 会话状态
            limit: 返回数量
            
        Returns:
            会话列表
        """
        sql = "SELECT * FROM conversations WHERE 1=1"
        params = []
        
        if user_id:
            sql += " AND user_id = %s"
            params.append(user_id)
        
        if status:
            sql += " AND status = %s"
            params.append(status)
        
        sql += f" ORDER BY created_at DESC LIMIT {limit}"
        
        return self.query(sql, tuple(params) if params else None) # type: ignore
    
    def get_conversation_messages(self, conversation_id: str) -> List[Dict]:
        """获取会话消息"""
        sql = """
            SELECT * FROM messages 
WHERE conversation_id = %s 
            ORDER BY created_at ASC
        """
        return self.query(sql, (conversation_id,))
    
    def get_knowledge_list(self, category: str | None = None, keyword: str | None = None, limit: int = 100) -> List[Dict]:
        """获取知识库列表"""
        sql = "SELECT * FROM knowledge_base WHERE 1=1"
        params = []
        
        if category:
            sql += " AND category = %s"
            params.append(category)
        
        if keyword:
            sql += " AND (title LIKE %s OR content LIKE %s)"
            kw = f"%{keyword}%"
            params.extend([kw, kw])
        
        sql += f" ORDER BY created_at DESC LIMIT {limit}"
        
        return self.query(sql, tuple(params) if params else None)
    
    def get_statistics(self, start_date: str, end_date: str) -> Dict:
        """获取统计数据"""
        # 总会话数
        sql = "SELECT COUNT(*) as total FROM conversations WHERE created_at BETWEEN %s AND %s"
        total = self.query_one(sql, (start_date, end_date))
        
        # 成功会话数
        sql = "SELECT COUNT(*) as completed FROM conversations WHERE status = 'completed' AND created_at BETWEEN %s AND %s"
        completed = self.query_one(sql, (start_date, end_date))
        
        # 转人工会话数
        sql = "SELECT COUNT(*) as transferred FROM conversations WHERE transferred_to_human = 1 AND created_at BETWEEN %s AND %s"
        transferred = self.query_one(sql, (start_date, end_date))
        
        return {
            "total": total.get('total', 0) if total else 0,
            "completed": completed.get('completed', 0) if completed else 0,
            "transferred": transferred.get('transferred', 0) if transferred else 0
        }
    
    # ============================================
    # 测试数据管理
    # ============================================
    
    def create_test_user(self, user_id: str | None = None) -> Dict:
        """创建测试用户"""
        import uuid
        
        user_id = user_id or str(uuid.uuid4())[:12]
        
        data = {
            "user_id": user_id,
            "username": f"test_{user_id}",
            "email": f"test_{user_id}@example.com",
            "phone": f"138{user_id[:8]}",
            "status": "active",
            "created_at": "NOW()"
        }
        
        # 使用原始SQL
        sql = f"""
            INSERT INTO users (user_id, username, email, phone, status, created_at)
            VALUES ('{user_id}', '{data['username']}', '{data['email']}', '{data['phone']}', 'active', NOW())
        """
        
        try:
            self.execute(sql)
            return data
        except:
            # 如果表不存在，使用内存数据库
            return {"user_id": user_id, "note": "表不存在"}
    
    def clean_test_data(self, prefix: str = "test_") -> int:
        """
        清理测试数据
        
        Args:
            prefix: 数据前缀
            
        Returns:
            清理行数
        """
        count = 0
        
        tables = ['users', 'conversations', 'messages', 'knowledge_base']
        
        for table in tables:
            try:
                sql = f"DELETE FROM {table} WHERE username LIKE '{prefix}%' OR user_id LIKE '{prefix}%'"
                count += self.execute(sql)
            except:
                pass
        
        logger.info(f"清理测试数据: {count} 行")
        return count
    
    def backup_table(self, table: str, backup_suffix: str = "_bak") -> bool:
        """
        备份表
        
        Args:
            table: 表名
            backup_suffix: 备份后缀
            
        Returns:
            是否成功
        """
        backup_table = f"{table}{backup_suffix}"
        
        try:
            # 删除备份表
            sql = f"DROP TABLE IF EXISTS {backup_table}"
            self.execute(sql)
            
            # 创建备份
            sql = f"CREATE TABLE {backup_table} AS SELECT * FROM {table}"
            self.execute(sql)
            
            logger.info(f"表 {table} 已备份到 {backup_table}")
            return True
            
        except Exception as e:
            logger.error(f"备份表失败: {e}")
            return False
    
    # ============================================
    # 工具方法
    # ============================================
    
    def table_exists(self, table: str) -> bool:
        """检查表是否存在"""
        try:
            sql = f"SELECT 1 FROM {table} LIMIT 1"
            self.query(sql)
            return True
        except:
            return False
    
    def get_table_columns(self, table: str) -> List[str]:
        """获取表字段"""
        try:
            sql = f"DESCRIBE {table}"
            results = self.query(sql)
            return [r['Field'] for r in results]
        except:
            return []
    
    def execute_sql_file(self, file_path: str) -> bool:
        """
        执行SQL文件
        
        Args:
            file_path: SQL文件路径
            
        Returns:
            是否成功
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                sql = f.read()
                
            # 分割SQL语句
            statements = [s.strip() for s in sql.split(';') if s.strip()]
            
            for stmt in statements:
                self.execute(stmt)
            
            logger.info(f"执行SQL文件成功: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"执行SQL文件失败: {e}")
            return False


# 便捷函数
def create_db_manager(env: str = "test") -> DBManager:
    """创建数据库管理器"""
    return DBManager(env=env)
