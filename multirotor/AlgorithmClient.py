import json
import time
import os
import socket
from Algorithm.scannerData import ScannerData
from Algorithm.scanner_algorithm import ScannerAlgorithm
from Algorithm.HexGridDataModelData import HexGridDataModel
from Algorithm.Vector3 import Vector3

class ScannerClient:
    def __init__(self, host='localhost', port=65432, config_file='Algorithm/scanner_config.json', grid_data_file=None):
        # 初始化配置
        self.host = host
        self.port = port
        self.client_socket = None
        self.config_file = config_file
        self.grid_data_file = grid_data_file
        
        # 初始化算法和数据
        self.scanner_data = None
        self.scanner_algorithm = None
        self.grid_model = None
        
        # 初始化标志
        self.running = False
        
    def initialize_configuration(self):
        """初始化配置数据"""
        try:
            # 读取配置文件
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                self.scanner_data = ScannerData(config_data)
                print(f"成功加载配置文件: {self.config_file}")
            else:
                # 如果没有配置文件，创建默认配置
                self.scanner_data = ScannerData()
                print("使用默认配置")
                
            # 初始化算法
            self.scanner_algorithm = ScannerAlgorithm(scanner_data=self.scanner_data)
            
            # 加载网格数据模型（如果有）
            if self.grid_data_file and os.path.exists(self.grid_data_file):
                self.grid_model = HexGridDataModel.deserialize_from_json_file(self.grid_data_file)
                self.scanner_algorithm.grid_model = self.grid_model
                print(f"成功加载网格数据模型: {self.grid_data_file}")
                
            return True
        except Exception as e:
            print(f"初始化配置数据时出错: {str(e)}")
            return False
    
    def connect_to_server(self):
        """连接无人机服务端"""
        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.connect((self.host, self.port))
            print(f"成功连接到服务器: {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"连接服务器失败: {str(e)}")
            return False
    
    def send_command(self, command, params=None):
        """发送命令到服务器并获取响应"""
        if params is None:
            params = {}
        
        try:
            data = json.dumps({'command': command, 'params': params})
            self.client_socket.sendall(data.encode('utf-8'))
            response = self.client_socket.recv(4096).decode('utf-8')
            return json.loads(response)
        except Exception as e:
            print(f"发送命令时出错: {str(e)}")
            return None
    
    def get_unity_data(self):
        """获取Unity数据"""
        try:
            response = self.send_command('get_unity_data')
            if response and response.get('success', False):
                unity_data = response.get('data', {})
                # 更新scanner_data中的位置和其他Unity提供的数据
                if 'position' in unity_data:
                    pos_data = unity_data['position']
                    self.scanner_data.position = Vector3(pos_data.get('x', 0), 
                                                                 pos_data.get('y', 0), 
                                                                 pos_data.get('z', 0))
                if 'forward' in unity_data:
                    forward_data = unity_data['forward']
                    self.scanner_data.forward = Vector3(forward_data.get('x', 0), 
                                                                forward_data.get('y', 0), 
                                                                forward_data.get('z', 1))
                # 可以添加更多需要更新的数据字段
                return unity_data
            return None
        except Exception as e:
            print(f"获取Unity数据时出错: {str(e)}")
            return None
    
    def calculate_finalMoveDir(self):
        """计算最终移动方向"""
        try:
            # 使用算法处理扫描器数据并获取最终移动方向
            final_move_dir = self.scanner_algorithm.process(self.scanner_data)
            return final_move_dir
        except Exception as e:
            print(f"计算最终移动方向时出错: {str(e)}")
            # 出错时返回当前forward方向
            return self.scanner_data.forward
    
    def send_drone_command(self, command, params=None):
        """发送无人机指令"""
        try:
            response = self.send_command(command, params)
            if response and response.get('success', False):
                print(f"成功发送无人机指令: {command}")
                return True
            print(f"发送无人机指令失败: {command}, 响应: {response}")
            return False
        except Exception as e:
            print(f"发送无人机指令时出错: {str(e)}")
            return False
    
    def send_calculated_data(self):
        """发送计算后的数据给服务器"""
        try:
            # 准备要发送的数据
            calculated_data = {
                'position': self.scanner_data.position.to_dict(),
                'forward': self.scanner_data.forward.to_dict(),
                'finalMoveDir': self.scanner_data.finalMoveDir.to_dict(),
                'scoreDir': self.scanner_data.scoreDir.to_dict()
            }
            
            response = self.send_command('update_calculated_data', {'data': calculated_data})
            if response and response.get('success', False):
                print("成功发送计算后的数据")
                return True
            print(f"发送计算后的数据失败，响应: {response}")
            return False
        except Exception as e:
            print(f"发送计算后的数据时出错: {str(e)}")
            return False
    
    def run(self):
        """运行主循环"""
        # 初始化配置
        if not self.initialize_configuration():
            print("配置初始化失败，退出程序")
            return False
        
        # 连接服务器
        if not self.connect_to_server():
            print("服务器连接失败，退出程序")
            return False
        
        try:
            # 初始化无人机（可以根据需要调整初始化步骤）
            print("初始化无人机...")
            
            # 连接模拟器
            self.send_command('connect')
            time.sleep(1)
            
            # 启用API控制
            self.send_drone_command('enable_api', {'vehicle_name': 'UAV1'})
            time.sleep(1)
            
            # 解锁无人机
            self.send_drone_command('arm', {'vehicle_name': 'UAV1'})
            time.sleep(1)
            
            # 起飞
            self.send_drone_command('takeoff', {'vehicle_name': 'UAV1'})
            time.sleep(5)
            
            # 主循环
            self.running = True
            while self.running:
                # 获取Unity数据
                unity_data = self.get_unity_data()
                if not unity_data:
                    time.sleep(self.scanner_data.updateInterval)
                    continue
                
                # 计算最终移动方向
                final_move_dir = self.calculate_finalMoveDir()
                
                # 发送无人机指令（根据最终移动方向设置速度）
                move_params = {
                    'vx': final_move_dir.x,
                    'vy': final_move_dir.y,
                    'vz': final_move_dir.z,
                    'vehicle_name': 'UAV1'
                }
                self.send_drone_command('move_by_velocity', move_params)
                
                # 发送计算后的数据给服务器
                self.send_calculated_data()
                
                # 等待下一更新周期
                time.sleep(self.scanner_data.updateInterval)
                
        except KeyboardInterrupt:
            print("用户中断程序")
        except Exception as e:
            print(f"程序运行出错: {str(e)}")
        finally:
            # 清理工作
            self.running = False
            
            try:
                # 降落无人机
                if self.client_socket:
                    print("正在降落无人机...")
                    self.send_drone_command('land', {'vehicle_name': 'UAV1'})
                    time.sleep(5)
                    
                    # 上锁
                    self.send_drone_command('arm', {'arm': False, 'vehicle_name': 'UAV1'})
                    time.sleep(1)
                    
                    # 禁用API控制
                    self.send_drone_command('enable_api', {'enable': False, 'vehicle_name': 'UAV1'})
            except Exception as e:
                print(f"清理工作时出错: {str(e)}")
            
            # 关闭连接
            if self.client_socket:
                self.client_socket.close()
                print("已关闭与服务器的连接")
        
        return True

# 主函数
def main():
    # 创建并运行客户端
    client = ScannerClient()
    client.run()

if __name__ == '__main__':
    main()