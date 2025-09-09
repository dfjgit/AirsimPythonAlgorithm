import time
import logging
from typing import Dict, Any, Optional
import threading
import time

# 配置日志
logger = logging.getLogger("DataStorage")

class DataStorageManager:
    """
    管理数据存储和转发功能
    提供数据的存储、检索、删除和列表功能
    """
    def __init__(self):
        # 数据存储结构，用于数据转发功能
        self.stored_data: Dict[str, Dict[str, Any]] = {}
        # 用于多线程安全访问存储的数据
        self.data_lock = threading.Lock()

    def store_data(self, data_id: str, content: Any) -> Dict[str, Any]:
        """
        存储数据
        
        :param data_id: 数据标识符
        :param content: 要存储的内容
        :return: 包含状态和消息的字典
        """
        if not data_id:
            return {
                "status": "error",
                "message": "缺少data_id参数"
            }
        
        # 使用锁确保线程安全
        with self.data_lock:
            self.stored_data[data_id] = {
                "content": content,
                "timestamp": time.time()  # 记录存储时间戳
            }
        
        logger.info(f"已存储数据，ID: {data_id}")
        return {
            "status": "success", 
            "message": f"数据已存储，ID: {data_id}",
            "data_id": data_id
        }

    def retrieve_data(self, data_id: str) -> Dict[str, Any]:
        """
        获取存储的数据
        
        :param data_id: 数据标识符
        :return: 包含状态、消息和数据内容的字典
        """
        if not data_id:
            return {
                "status": "error",
                "message": "缺少data_id参数"
            }
        
        with self.data_lock:
            if data_id in self.stored_data:
                data = self.stored_data[data_id]
                return {
                    "status": "success",
                    "message": f"成功获取数据，ID: {data_id}",
                    "data_id": data_id,
                    "content": data["content"],
                    "timestamp": data["timestamp"]
                }
            else:
                return {
                    "status": "error", 
                    "message": f"未找到ID为{data_id}的数据"
                }

    def delete_data(self, data_id: str) -> Dict[str, Any]:
        """
        删除存储的数据
        
        :param data_id: 数据标识符
        :return: 包含状态和消息的字典
        """
        if not data_id:
            return {
                "status": "error",
                "message": "缺少data_id参数"
            }
        
        with self.data_lock:
            if data_id in self.stored_data:
                del self.stored_data[data_id]
                return {
                    "status": "success",
                    "message": f"已删除数据，ID: {data_id}",
                    "data_id": data_id
                }
            else:
                return {
                    "status": "error", 
                    "message": f"未找到ID为{data_id}的数据"
                }

    def list_data_ids(self) -> Dict[str, Any]:
        """
        列出所有存储的数据ID
        
        :return: 包含状态、消息和数据ID列表的字典
        """
        with self.data_lock:
            data_ids = list(self.stored_data.keys())
            return {
                "status": "success",
                "message": f"共找到{len(data_ids)}条数据",
                "count": len(data_ids),
                "data_ids": data_ids
            }

    def clear_all_data(self) -> Dict[str, Any]:
        """
        清除所有存储的数据
        
        :return: 包含状态和消息的字典
        """
        with self.data_lock:
            count = len(self.stored_data)
            self.stored_data.clear()
            logger.info(f"已清除所有{count}条数据")
            return {
                "status": "success",
                "message": f"已清除所有{count}条数据",
                "cleared_count": count
            }