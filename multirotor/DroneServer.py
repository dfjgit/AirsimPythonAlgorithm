import setup_path
import logging

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("drone_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DroneServer")

# 导入拆分后的组件
from AirsimServer.drone_controller import DroneController
from AirsimServer.data_storage import DataStorageManager
from AirsimServer.command_processor import CommandProcessor
from AirsimServer.socket_server import DroneSocketServer
from AirsimServer.unity_environment import DroneControllerProxy


class DroneServer:
    """
    无人机服务器主类
    组合了所有组件，提供完整的无人机控制服务
    """
    def __init__(self, host: str = '0.0.0.0', port: int = 65432):
        # 初始化各个组件
        self.drone_controller = DroneController()
        self.data_manager = DataStorageManager()
        
        # 使用无人机控制器代理类
        self.drone_proxy = DroneControllerProxy(self.drone_controller)
        
        # 初始化命令处理器
        self.command_processor = CommandProcessor(
            drone_controller=self.drone_proxy,
            data_manager=self.data_manager
        )
        
        # 初始化Socket服务器
        self.socket_server = DroneSocketServer(
            host=host,
            port=port,
            command_processor=self.command_processor
        )
    
    def start(self):
        """启动服务器"""
        self.socket_server.start()
    
    def stop(self):
        """停止服务器"""
        self.socket_server.stop()


if __name__ == "__main__":
    # 创建服务器实例
    server = DroneServer()
    try:
        # 启动服务器
        server.start()
    except KeyboardInterrupt:  # Ctrl+C停止服务器
        logger.info("收到中断信号，正在停止服务器...")
        server.stop()
    except Exception as e:
        logger.critical(f"服务器运行时发生致命错误: {e}", exc_info=True)
        server.stop()
