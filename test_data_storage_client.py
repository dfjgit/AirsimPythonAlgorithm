#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试通过Socket接口使用数据存储功能的客户端脚本"""

import socket
import json
import time

print("=== AirSim DroneServer 数据存储Socket客户端测试 ===")

# 服务器配置
SERVER_HOST = 'localhost'
SERVER_PORT = 65432

class DataStorageClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.socket = None
    
    def connect(self):
        """连接到服务器"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            print(f"✓ 已连接到服务器 {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"✗ 连接服务器失败: {e}")
            print("  请确保DroneServer.py正在运行")
            return False
    
    def send_command(self, command):
        """发送命令到服务器"""
        if not self.socket:
            print("✗ 未连接到服务器")
            return None
        
        try:
            # 发送命令
            self.socket.sendall(json.dumps(command).encode('utf-8'))
            
            # 接收响应
            response_data = self.socket.recv(4096).decode('utf-8')
            response = json.loads(response_data)
            return response
        except Exception as e:
            print(f"✗ 发送命令时出错: {e}")
            return None
    
    def disconnect(self):
        """断开与服务器的连接"""
        if self.socket:
            try:
                self.socket.close()
                print("✓ 已断开与服务器的连接")
            except Exception as e:
                print(f"✗ 断开连接时出错: {e}")
            self.socket = None

# 创建测试客户端并连接到服务器
client = DataStorageClient(SERVER_HOST, SERVER_PORT)

if client.connect():
    # 测试存储数据
    print("\n测试存储数据...")
    store_cmd = {
        "command": "store_data",
        "params": {
            "data_id": "socket_test_id_1",
            "content": {"name": "socket_test_data", "value": 789, "timestamp": time.time()}
        }
    }
    response = client.send_command(store_cmd)
    if response:
        print(f"  存储结果: {response.get('status')} - {response.get('message')}")
    
    # 测试列出所有数据ID
    print("\n测试列出所有数据ID...")
    list_cmd = {
        "command": "list_data_ids",
        "params": {}
    }
    response = client.send_command(list_cmd)
    if response:
        print(f"  列表结果: {response.get('status')} - {response.get('message')}")
        if response.get('status') == 'success':
            print(f"    数据ID数量: {response.get('count')}")
    
    # 测试检索数据
    print("\n测试检索数据...")
    retrieve_cmd = {
        "command": "retrieve_data",
        "params": {
            "data_id": "socket_test_id_1"
        }
    }
    response = client.send_command(retrieve_cmd)
    if response:
        print(f"  检索结果: {response.get('status')} - {response.get('message')}")
        # 不打印数据内容
    
    # 测试更新数据
    print("\n测试更新数据...")
    update_cmd = {
        "command": "store_data",
        "params": {
            "data_id": "socket_test_id_1",
            "content": {"name": "updated_socket_test_data", "value": 999, "updated_at": time.time()}
        }
    }
    response = client.send_command(update_cmd)
    if response:
        print(f"  更新结果: {response.get('status')} - {response.get('message')}")
    
    # 再次检索数据，验证更新
    print("\n验证数据更新...")
    response = client.send_command(retrieve_cmd)
    if response:
        print(f"  更新后数据验证: {response.get('status')} - {response.get('message')}")
        # 不打印数据内容
    
    # 测试删除数据
    print("\n测试删除数据...")
    delete_cmd = {
        "command": "delete_data",
        "params": {
            "data_id": "socket_test_id_1"
        }
    }
    response = client.send_command(delete_cmd)
    if response:
        print(f"  删除结果: {response.get('status')} - {response.get('message')}")
    
    # 验证数据是否已删除
    print("\n验证数据是否已删除...")
    response = client.send_command(retrieve_cmd)
    if response:
        print(f"  验证删除结果: {response.get('status')} - {response.get('message')}")
    
    # 断开连接
    client.disconnect()
else:
    print("\n无法连接到服务器，无法进行Socket接口测试")
    print("请先启动DroneServer.py，然后再运行此测试脚本")

print("\n=== Socket客户端数据存储功能测试完成 ===")