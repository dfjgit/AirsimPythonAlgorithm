import time
import logging
from typing import Dict, Any, Optional

# 配置日志
logger = logging.getLogger("UnityEnvironment")

class UnityEnvironmentExtension:
    """
    处理Unity自定义环境数据的扩展模块
    提供环境数据的存储、更新和查询功能
    """
    def __init__(self):
        # 存储Unity传递的环境数据（字符串格式）
        self.unity_environment_data = ""
        # 记录最后一次更新时间（Unix时间戳）
        self.env_data_update_time = 0

    def update_unity_environment_data(self, env_str: str) -> bool:
        """
        更新Unity发送的环境数据
        
        :param env_str: Unity传递的环境数据字符串（任意格式）
        :return: 是否更新成功
        """
        if not isinstance(env_str, str):
            logger.error("Unity环境数据必须是字符串格式")
            return False
            
        self.unity_environment_data = env_str
        self.env_data_update_time = self._get_current_timestamp()
        logger.info(f"Unity环境数据已更新，最后更新时间: {self.env_data_update_time}")
        logger.debug(f"环境数据内容: {env_str}")
        return True

    def get_unity_environment_data(self) -> Dict[str, Any]:
        """
        获取当前存储的Unity环境数据
        
        :return: 包含环境数据和元信息的字典
        """
        return {
            "env_data": self.unity_environment_data,
            "last_update_time": self.env_data_update_time,
            "is_empty": len(self.unity_environment_data) == 0
        }

    def reset_environment_data(self) -> None:
        """重置环境数据（清空存储）"""
        self.unity_environment_data = ""
        self.env_data_update_time = 0
        logger.info("Unity环境数据已重置")

    def _get_current_timestamp(self) -> int:
        """获取当前Unix时间戳（秒级）"""
        return int(time.time())


class DroneControllerWithEnv(DroneController, UnityEnvironmentExtension):
    """
    扩展原有DroneController，添加环境数据处理功能
    注意：需要在原有DroneController所在文件中导入此类
    """
    def __init__(self):
        # 初始化父类
        DroneController.__init__(self)
        UnityEnvironmentExtension.__init__(self)
        
    def reset(self) -> bool:
        """重写重置方法，同时重置环境数据"""
        result = super().reset()
        if result:
            self.reset_environment_data()
        return result
    