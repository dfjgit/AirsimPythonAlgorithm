import sys
import os

# 设置 Python 路径，添加项目根目录
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import socket
import threading
import json
from Vector3 import Vector3
from scanner_algorithm import ScannerAlgorithm
from HexGridDataModel import HexGridDataModel
from scanner_config_data import ScannerConfigData
from scanner_runtime_data import ScannerRuntimeData

HEX_GRID_KEY = 'hex_grid'  # 通用网格数据键（只负责读取）
CONFIG_KEY_PREFIX = 'scanner_config_'  #参数配置键 （初始化时写入）
RUNTIME_UNITY_KEY_PREFIX = 'unity_scanner_runtime_'  # Unity数据键（只负责读取）
RUNTIME_PYTHON_KEY_PREFIX = 'python_scanner_runtime_' # 计算结果键（只负责写入）

# 数据交互配置常量（仅hex_grid通用，其他按无人机区分）
def get_unity_data_key(vehicle_name):
    return RUNTIME_UNITY_KEY_PREFIX + vehicle_name  # 无人机专属Unity数据键

def get_calculated_movement_key(vehicle_name):
    return RUNTIME_PYTHON_KEY_PREFIX + vehicle_name  # 无人机专属计算结果键

class ScannerClient:
    def __init__(self, host='localhost', port=65432, 
                 config_file='Algorithm/scanner_config.json', 
                 vehicle_names=["UAV1"]):
        # 初始化配置
        self.host = host
        self.port = port
        self.client_socket = None
        self.config_file = config_file
        self.vehicle_names = vehicle_names  # 多无人机名称列表
        
        # 数据存储
        self.config_data_map = {}  # {vehicle_name: ScannerConfigData}
        self.runtime_data_map = {}  # {vehicle_name: ScannerRuntimeData}
        self.scanner_algorithm_map = {}  # {vehicle_name: ScannerAlgorithm}
        self.grid_model = None  # 网格模型，从服务器获取，所有无人机共享
        
        # 线程安全控制
        self.lock = threading.Lock()
        self.running = False
        self.threads = []
    
    def initialize_configuration(self):
        """初始化配置（网格模型仅加载一次）"""
        try:
            # 读取全局配置
            config_data = {}
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
            
            # 为每个无人机初始化独立数据
            for vehicle_name in self.vehicle_names:
                vehicle_config = config_data.get(vehicle_name, config_data)
                
                # 创建配置数据对象
                self.config_data_map[vehicle_name] = ScannerConfigData.from_dict(vehicle_config)
                
                # 创建初始运行时数据对象
                self.runtime_data_map[vehicle_name] = ScannerRuntimeData()
                # 设置初始值
                self.runtime_data_map[vehicle_name].position = Vector3(0, 0, 0)
                self.runtime_data_map[vehicle_name].forward = Vector3(0, 0, 1)  # 默认向前
                self.runtime_data_map[vehicle_name].leaderPosition = Vector3(0, 0, 0)
                
                # 使用配置数据初始化算法
                self.scanner_algorithm_map[vehicle_name] = ScannerAlgorithm(
                    config_data=self.config_data_map[vehicle_name]
                )
            
            print(f"成功初始化 {len(self.vehicle_names)} 个无人机配置")
            return True
        except Exception as e:
            print(f"初始化配置出错: {str(e)}")
            return False
    
    def connect_to_server(self):
        """连接服务器"""
        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.connect((self.host, self.port))
            print(f"成功连接到服务器: {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"连接服务器失败: {str(e)}")
            return False
    
    def send_command(self, command, params=None):
        """线程安全的命令发送"""
        if params is None:
            params = {}
        
        try:
            with self.lock:
                data = json.dumps({'command': command, 'params': params})
                self.client_socket.sendall(data.encode('utf-8'))
                response = self.client_socket.recv(4096).decode('utf-8')
            return json.loads(response) if response else None
        except Exception as e:
            print(f"发送命令出错: {str(e)}")
            return None
    
    def store_data(self, data_id, content):
        """存储数据到服务器"""
        try:
            response = self.send_command('store_data', {'data_id': data_id, 'content': content})
            return response and response.get('status') == 'success'
        except Exception as e:
            print(f"存储数据出错: {str(e)}")
            return False
    
    def retrieve_data(self, data_id):
        """从服务器获取数据"""
        try:
            response = self.send_command('retrieve_data', {'data_id': data_id})
            return response.get('content', {}) if (response and response.get('status') == 'success') else None
        except Exception as e:
            print(f"获取数据出错: {str(e)}")
            return None
    
    def _update_runtime_data(self, vehicle_name, unity_data):
        """更新单个无人机的运行时数据"""
        runtime_data = self.runtime_data_map[vehicle_name]
        
        # 更新位置
        if 'position' in unity_data:
            pos = unity_data['position']
            runtime_data.position = Vector3(pos.get('x', 0), pos.get('y', 0), pos.get('z', 0))
        
        # 更新方向
        if 'forward' in unity_data:
            forward = unity_data['forward']
            runtime_data.forward = Vector3(forward.get('x', 0), forward.get('y', 0), forward.get('z', 1))
        
        # 更新领航者信息
        if 'leaderPosition' in unity_data:
            leader_pos = unity_data['leaderPosition']
            runtime_data.leaderPosition = Vector3(leader_pos.get('x', 0), leader_pos.get('y', 0), leader_pos.get('z', 0))
        
        if 'leaderScanRadius' in unity_data:
            runtime_data.leaderScanRadius = unity_data.get('leaderScanRadius', 0.0)
        
        # 更新配置参数
        if 'scannerConfig' in unity_data:
            cfg = unity_data['scannerConfig']
            self.config_data_map[vehicle_name].update_from_dict(cfg)
    
    def update_common_hex_grid(self):
        """更新通用网格数据（所有无人机共享）"""
        try:
            hex_grid_data = self.retrieve_data(HEX_GRID_KEY)
            if not hex_grid_data:
                return False
            
            # 更新通用网格模型
            self.grid_model = HexGridDataModel.from_dict(hex_grid_data)
            print(f"通用网格数据更新，包含 {len(self.grid_model.cells)} 个单元")
            return True
        except Exception as e:
            print(f"更新通用网格数据出错: {str(e)}")
            return False
    
    def calculate_movement(self, vehicle_name):
        """计算单个无人机的移动方向和新位置"""
        try:
            # 确保有网格数据
            if not self.grid_model:
                print(f"警告: 暂无网格数据，使用默认方向")
                return self.runtime_data_map[vehicle_name].direction
            
            # 使用算法计算更新后的运行时数据
            updated_runtime_data = self.scanner_algorithm_map[vehicle_name].update_runtime_data(
                self.grid_model, 
                self.runtime_data_map[vehicle_name]
            )
            
            # 更新运行时数据
            self.runtime_data_map[vehicle_name] = updated_runtime_data
            
            return updated_runtime_data.direction
        except Exception as e:
            print(f"无人机 {vehicle_name} 计算移动方向出错: {str(e)}")
            return self.runtime_data_map[vehicle_name].direction
    
    def process_vehicle(self, vehicle_name):
        """单个无人机的处理逻辑"""
        try:
            # 无人机初始化
            print(f"初始化无人机 {vehicle_name}...")
            self.send_command('enable_api', {'vehicle_name': vehicle_name})
            time.sleep(1)
            self.send_command('arm', {'vehicle_name': vehicle_name})
            time.sleep(1)
            self.send_command('takeoff', {'vehicle_name': vehicle_name})
            time.sleep(5)
            
            # 无人机主循环
            while self.running:
                # 获取当前无人机的Unity数据
                unity_data = self.retrieve_data(get_unity_data_key(vehicle_name))
                if not unity_data:
                    time.sleep(0.1)  # 短暂等待后重试
                    continue
                
                # 更新当前无人机的运行时数据
                self._update_runtime_data(vehicle_name, unity_data)
                
                # 计算移动方向
                move_dir = self.calculate_movement(vehicle_name)
                
                # 发送移动指令
                self.send_command('move_by_velocity', {
                    'vx': move_dir.x,  # 使用方向向量的分量作为速度
                    'vy': move_dir.y,
                    'vz': move_dir.z,
                    'vehicle_name': vehicle_name
                })
                
                # 发送计算结果（配置数据和运行时数据）
                # self.store_data(
                #     CONFIG_KEY_PREFIX + vehicle_name,
                #     self.config_data_map[vehicle_name].to_dict()
                # )
                
                # 发送计算后数据
                self.store_data(
                    RUNTIME_PYTHON_KEY_PREFIX + vehicle_name,
                    self.runtime_data_map[vehicle_name].to_dict()
                )
                
                # 发送计算后的移动方向
                self.store_data(
                    get_calculated_movement_key(vehicle_name),
                    move_dir.to_dict()
                )
                
                # 等待下一周期
                time.sleep(0.1)  # 100ms更新频率
                
        except Exception as e:
            print(f"无人机 {vehicle_name} 处理出错: {str(e)}")
    
    def run(self):
        """主运行逻辑"""
        if not self.initialize_configuration() or not self.connect_to_server():
            return False
        
        try:
            # 连接模拟器
            self.send_command('connect')
            time.sleep(1)
            
            # 启动通用网格数据更新线程（单独线程定期更新）
            def update_grid_loop():
                while self.running:
                    self.update_common_hex_grid()
                    time.sleep(0.5)  # 每0.5秒更新一次网格数据
            
            # 启动网格更新线程
            grid_thread = threading.Thread(target=update_grid_loop)
            grid_thread.daemon = True
            self.threads.append(grid_thread)
            
            # 为每个无人机启动处理线程
            for vehicle_name in self.vehicle_names:
                thread = threading.Thread(target=self.process_vehicle, args=(vehicle_name,))
                thread.daemon = True
                self.threads.append(thread)
            
            # 启动所有线程
            self.running = True
            for thread in self.threads:
                thread.start()
            
            # 主线程等待（可以添加退出条件）
            while self.running:
                time.sleep(1)
        
        except Exception as e:
            print(f"主运行逻辑出错: {str(e)}")
            self.running = False
        finally:
            # 清理资源
            for thread in self.threads:
                thread.join()
            if self.client_socket:
                self.client_socket.close()
            print("客户端已停止")
            return True

if __name__ == "__main__":
    client = ScannerClient(
        host='localhost',
        port=65432,
        config_file='Algorithm/scanner_config.json',
        vehicle_names=["UAV1"]
    )
    client.run()