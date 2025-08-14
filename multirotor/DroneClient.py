import base64
import io
import socket
import json
import time
from PIL import Image  # 仅保留PIL的Image导入，避免与tkinter冲突

class DroneClient:
    def __init__(self, host='127.0.0.1', port=65432):
        self.host = host
        self.port = port
        self.socket = None

    def connect(self):
        """连接到服务器"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            print(f"已连接到服务器 {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"连接失败: {e}")
            return False

    def send_command(self, command, params=None, timeout=10.0):
        """发送命令到服务器并返回响应"""
        if not self.socket:
            print("未建立连接，请先调用connect()")
            return None

        if params is None:
            params = {}

        # 构建命令JSON
        cmd_data = {
            "command": command,
            "params": params
        }

        try:
            # 设置超时
            self.socket.settimeout(timeout)
            
            # 发送命令
            self.socket.sendall(json.dumps(cmd_data).encode('utf-8'))
            print(f"发送命令: {command} 参数: {params}")

            # 接收响应（循环接收完整数据）
            response_data = b''
            while True:
                chunk = self.socket.recv(4096)
                if not chunk:
                    break
                response_data += chunk
                # 简单判断JSON是否完整（实际应用可能需要更复杂的处理）
                if b'}' in chunk:
                    break

            if response_data:
                try:
                    response = json.loads(response_data.decode('utf-8'))
                    print(f"收到响应: {'成功' if response.get('status') == 'success' else '失败'} - {response.get('message')}")
                    return response
                except json.JSONDecodeError:
                    print(f"响应格式错误，无法解析JSON: {response_data}")
                    return None
            else:
                print("未收到响应")
                return None
        except socket.timeout:
            print(f"命令'{command}'超时")
            return None
        except json.JSONDecodeError:
            print("响应格式错误，无法解析JSON")
            return None
        except Exception as e:
            print(f"发送命令出错: {e}")
            return None

    def disconnect(self):
        """断开与服务器的连接"""
        if self.socket:
            try:
                # 发送断开连接命令
                self.send_command("disconnect")
            except Exception:
                pass
            finally:
                self.socket.close()
                self.socket = None
                print("已断开与服务器的连接")

    def save_image(self, response, vehicle_name, image_type):
        """保存图像数据到文件"""
        if not response or response.get("status") != "success":
            print("无法保存图像，响应无效")
            return False

        try:
            # 解码图像数据
            image_data = base64.b64decode(response["image_data"])
            
            # 验证图像格式
            with Image.open(io.BytesIO(image_data)) as img:
                print(f"图像信息 - 尺寸: {img.size}, 格式: {img.format}, 模式: {img.mode}")
                
                # 生成文件名
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"{vehicle_name}_{image_type}_{timestamp}.jpg"
                
                # 保存图像
                with open(filename, "wb") as f:
                    f.write(image_data)
                print(f"图像已保存至: {filename}")
                return True
        except Exception as e:
            print(f"处理图像时出错: {str(e)}")
            return False


if __name__ == "__main__":
    # 创建客户端实例
    client = DroneTestClient()
    
    # 连接服务器
    if client.connect():
        # 测试命令序列
        try:
            # 首先连接到模拟器（全局操作，不需要指定无人机）
            print("===== 连接到模拟器 ====")
            response = client.send_command("connect")
            if response.get("status") != "success":
                print("连接模拟器失败，终止测试")
                exit(1)
            time.sleep(1)

            # 重置模拟器状态
            print("\n===== 重置模拟器 ====")
            client.send_command("reset")
            time.sleep(2)

            # 测试UAV1控制
            vehicle_name = "UAV1"
            print(f"\n===== 测试无人机 {vehicle_name} ====")
            
            # 启用API控制
            client.send_command("enable_api", {"enable": True, "vehicle_name": vehicle_name})
            time.sleep(1)
            
            # 获取并打印无人机状态
            state = client.send_command("get_state", {"vehicle_name": vehicle_name})
            print(f"初始状态: {state.get('state')}")
            time.sleep(1)
            
            # 解锁无人机
            client.send_command("arm", {"arm": True, "vehicle_name": vehicle_name})
            time.sleep(1)
            
            # 起飞（设置超时时间）
            client.send_command("takeoff", {
                "vehicle_name": vehicle_name,
                "timeout": 20
            })
            time.sleep(6)  # 等待起飞完成
            
            # 查看起飞后的状态
            client.send_command("get_state", {"vehicle_name": vehicle_name})
            time.sleep(1)
            
            # 移动到指定位置
            print("\n===== 移动到目标位置 ====")
            client.send_command("move_to_position", {
                "x": 10, "y": 0, "z": -5, 
                "speed": 2, 
                "timeout": 15,
                "vehicle_name": vehicle_name
            })
            time.sleep(6)  # 等待移动完成
            
            # 获取当前位置
            client.send_command("get_state", {"vehicle_name": vehicle_name})
            time.sleep(1)
            
            # 测试图像获取
            print("\n===== 测试图像获取 ====")
            image_types = ["Scene", "Segmentation", "DepthVis"]
            for img_type in image_types:
                print(f"\n获取{img_type}类型图像...")
                response = client.send_command("get_image", {
                    "vehicle_name": vehicle_name, 
                    "camera_name": "1", 
                    "image_type": img_type
                })
                client.save_image(response, vehicle_name, img_type)
            time.sleep(2)
            

            
            # 测试UAV2
            # vehicle_name = "UAV2"
            # print(f"\n===== 测试无人机 {vehicle_name} ====")
            # client.send_command("enable_api", {"enable": True, "vehicle_name": vehicle_name})
            # time.sleep(1)
            # client.send_command("arm", {"arm": True, "vehicle_name": vehicle_name})
            # time.sleep(1)
            # client.send_command("takeoff", {"vehicle_name": vehicle_name, "timeout": 20})
            # time.sleep(6)
            # client.send_command("move_to_position", {
            #     "x": 0, "y": 10, "z": -5, 
            #     "speed": 2, 
            #     "timeout": 15,
            #     "vehicle_name": vehicle_name
            # })
            # time.sleep(6)

            # 降落并清理
            print("\n===== 任务完成，准备降落 ====")
            client.send_command("land", {"vehicle_name": "UAV1", "timeout": 20})
            # client.send_command("land", {"vehicle_name": "UAV2", "timeout": 20})
            time.sleep(6)
            
            # 上锁
            client.send_command("arm", {"arm": False, "vehicle_name": "UAV1"})
            # client.send_command("arm", {"arm": False, "vehicle_name": "UAV2"})
            time.sleep(1)
            
            # 禁用API控制
            client.send_command("enable_api", {"enable": False, "vehicle_name": "UAV1"})
            # client.send_command("enable_api", {"enable": False, "vehicle_name": "UAV2"})
            
        except Exception as e:
            print(f"测试过程中发生错误: {e}")
        finally:
            client.disconnect()
            print("\n测试完成")
    