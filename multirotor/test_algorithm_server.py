import sys
import time
import json
import socket
import threading
import logging
from typing import Dict, Any, List
import traceback

# 添加项目根目录到Python路径
sys.path.append('d:\\Project\\Python\\AirsimAlgorithmPython')

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("TestAlgorithmServer")

# 导入必要的类
from multirotor.AlgorithmServer import MultiDroneAlgorithmServer
from multirotor.AirsimServer.data_pack import PackType
from multirotor.Algorithm.Vector3 import Vector3
from multirotor.Algorithm.scanner_runtime_data import ScannerRuntimeData

class UnityClientSimulator:
    """模拟Unity客户端的类"""
    
    def __init__(self, host='localhost', port=5000, buffer_size=4096):
        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        self.socket = None
        self.connected = False
        self.received_data = []
        self.running = False
        self.receive_thread = None
        
    def connect(self) -> bool:
        """连接到AlgorithmServer"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.connected = True
            self.running = True
            self.receive_thread = threading.Thread(target=self._receive_data, daemon=True)
            self.receive_thread.start()
            logger.info(f"已连接到服务器 {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"连接失败: {str(e)}")
            self.connected = False
            return False
    
    def disconnect(self) -> None:
        """断开连接"""
        self.running = False
        if self.receive_thread and self.receive_thread.is_alive():
            self.receive_thread.join(1.0)
        if self.socket:
            try:
                self.socket.close()
            except Exception as e:
                logger.error(f"关闭socket出错: {str(e)}")
            self.socket = None
        self.connected = False
        logger.info("已断开连接")
    
    def _receive_data(self) -> None:
        """接收来自服务器的数据"""
        buffer = ""
        while self.running and self.connected:
            try:
                data = self.socket.recv(self.buffer_size).decode('utf-8')
                if data:
                    buffer += data
                    # 尝试按行分割数据（因为服务器发送时会添加换行符）
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        if line.strip():
                            try:
                                parsed = json.loads(line)
                                self.received_data.append(parsed)
                                logger.info(f"收到服务器数据: {parsed.get('type')}")
                            except json.JSONDecodeError as e:
                                logger.error(f"解析JSON数据错误: {str(e)}, 数据: {line}")
            except Exception as e:
                logger.error(f"接收数据出错: {str(e)}")
                self.connected = False
                break
    
    def send_data(self, data_type: str, data: Any, uav_name: str = None) -> bool:
        """发送数据到服务器 - 严格按照DataPacks结构"""
        if not self.connected:
            logger.warning("未连接，无法发送数据")
            return False
        
        try:
            # 构建数据包 - 严格按照DataPacks类的结构要求
            # 注意：服务器期望的是type、time_span和pack_data_list三个字段
            callback_data = {
                'type': data_type,
                'time_span': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                'pack_data_list': data  # 直接使用data作为pack_data_list
            }
            
            # 发送数据 - 确保每次只发送一个完整的JSON对象，并添加换行符作为分隔符
            json_data = json.dumps(callback_data, ensure_ascii=False) + '\n'
            self.socket.sendall(json_data.encode('utf-8'))
            logger.info(f"已发送{data_type}数据，内容: {json_data[:100]}...")
            
            # 短暂延迟，确保数据被服务器正确接收
            time.sleep(0.1)
            return True
        except Exception as e:
            logger.error(f"发送数据出错: {str(e)}")
            return False
    
    def get_received_data(self, data_type: str = None) -> List[Dict[str, Any]]:
        """获取接收到的数据"""
        if data_type:
            return [d for d in self.received_data if d.get('type') == data_type]
        return self.received_data

class AlgorithmServerTester:
    """AlgorithmServer测试类"""
    
    def __init__(self):
        self.server = None
        self.client = None
        self.server_thread = None
    
    def setup(self) -> bool:
        """设置测试环境"""
        try:
            # 启动服务器（在单独线程中）
            self.server = MultiDroneAlgorithmServer(drone_names=["UAV1", "UAV2"])
            self.server_thread = threading.Thread(target=self._start_server, daemon=True)
            self.server_thread.start()
            
            # 等待服务器启动
            time.sleep(2)
            
            # 启动模拟客户端
            self.client = UnityClientSimulator()
            return True
        except Exception as e:
            logger.error(f"设置测试环境失败: {str(e)}")
            return False
    
    def _start_server(self) -> None:
        """启动服务器的方法"""
        try:
            # 使用mock来避免实际连接AirSim
            self.server.drone_controller.connect = lambda: True
            self.server.drone_controller.enable_api_control = lambda x, y: True
            self.server.drone_controller.arm_disarm = lambda x, y: True
            self.server.drone_controller.takeoff = lambda x: True
            self.server.drone_controller.move_by_velocity = lambda *args, **kwargs: True
            
            # 启动服务器
            self.server.start()
            logger.info("服务器已启动（模拟模式）")
        except Exception as e:
            logger.error(f"服务器启动失败: {str(e)}", exc_info=True)
    
    def test_data_exchange(self) -> bool:
        """测试数据交换功能"""
        try:
            # 1. 连接客户端
            if not self.client.connect():
                logger.error("客户端连接失败")
                return False
            
            # 等待连接建立
            time.sleep(2)
            
            # 增加详细日志记录
            logger.info("连接已建立，开始测试数据交换...")
            
            # 2. 测试发送grid_data - 严格按照DataStruct/grid.json的格式
            grid_data = {
                'cells': [
                    {
                        'center': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                        'entropy': 80.0
                    },
                    {
                        'center': {'x': 5.0, 'y': 0.0, 'z': 0.0},
                        'entropy': 0.0
                    }
                ]
            }
            if not self.client.send_data(PackType.grid_data.value, grid_data):
                logger.error("发送grid_data失败")
                return False
            
            # 等待数据处理
            time.sleep(1)
            
            # 3. 测试发送runtime_data - 严格按照ScannerRuntimeData的要求格式和unity_socket_server.py的期望
            # 注意：根据unity_socket_server.py第187-190行，runtime_data的pack_data_list应该是列表，每个元素包含uavname
            runtime_data = [
                {
                    'uavname': "UAV1",
                    'position': {'x': -15.72, 'y': -279.35883, 'z': -1.5},
                    'forward': {'x': 0.0, 'y': 0.0, 'z': 1.0},
                    'leaderPosition': {'x': 0.29982, 'y': 0.5, 'z': 0.0},
                    'scoreDir': {'x': 0.0, 'y': 0.0, 'z': 1.0},
                    'collideDir': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                    'pathDir': {'x': 0.0, 'y': 0.0, 'z': 1.0},
                    'leaderRangeDir': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                    'directionRetentionDir': {'x': 0.0, 'y': 0.0, 'z': 1.0},
                    'finalMoveDir': {'x': 0.0, 'y': 0.0, 'z': 1.0}
                }
            ]
            if not self.client.send_data(PackType.runtime_data.value, runtime_data):
                logger.error("发送runtime_data失败")
                return False
            
            # 等待数据处理
            time.sleep(1)
            
            # 4. 测试发送config_data
            config_data = {
                'repulsionCoefficient': 2.5,
                'entropyCoefficient': 3.5,
                'moveSpeed': 3.0,
                'updateInterval': 0.5,
                'distanceCoefficient': 2.0,
                'leaderRangeCoefficient': 2.0,
                'directionRetentionCoefficient': 2.0,
                'rotationSpeed': 120.0,
                'scanRadius': 2.0,
                'altitude': 2.0,
                'maxRepulsionDistance': 5.0,
                'minSafeDistance': 1.0,
                'avoidRevisits': True,
                'targetSearchRange': 20.0,
                'revisitCooldown': 10.0
            }
            if not self.client.send_data(PackType.config_data.value, config_data):
                logger.error("发送config_data失败")
                return False
            
            # 等待数据处理
            time.sleep(3)
            
            # 5. 验证接收到的数据
            received_data = self.client.get_received_data()
            if not received_data:
                logger.error("未收到服务器返回的数据")
                # 打印所有日志来帮助调试
                logger.info("尝试重新发送数据以进行调试...")
                self.client.send_data(PackType.config_data.value, config_data)
                time.sleep(1)
                received_data = self.client.get_received_data()
                logger.info(f"重试后收到{len(received_data)}条数据")
                return False
            
            # 打印接收到的数据详情
            for i, data in enumerate(received_data):
                logger.info(f"收到的数据 #{i+1}: type={data.get('type')}, keys={list(data.keys())}")
            
            # 检查是否收到了config_data类型的数据
            config_data_received = False
            for data in received_data:
                if data.get('type') == 'config_data':
                    config_data_received = True
                    break
            
            if not config_data_received:
                logger.error("未收到预期的config_data数据")
                return False
            
            logger.info(f"测试成功! 收到{len(received_data)}条数据")
            return True
        except Exception as e:
            logger.error(f"测试数据交换失败: {str(e)}", exc_info=True)
            return False
    
    def cleanup(self) -> None:
        """清理测试环境"""
        if self.client:
            self.client.disconnect()
        if self.server:
            self.server.stop()
        logger.info("测试环境已清理")

# 主测试函数
def run_test():
    tester = AlgorithmServerTester()
    try:
        if tester.setup():
            if tester.test_data_exchange():
                logger.info("所有测试通过!")
            else:
                logger.error("测试失败!")
    except Exception as e:
        logger.error(f"测试过程中发生错误: {str(e)}", exc_info=True)
    finally:
        tester.cleanup()

if __name__ == "__main__":
    run_test()