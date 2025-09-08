import socket
import json
import time
import Algorithm 

# 连接服务器
def connect_server(host='localhost', port=65432):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    return client_socket

# 发送命令
def send_command(client_socket, command, params=None):
    if params is None:
        params = {}
    data = json.dumps({'command': command, 'params': params})
    client_socket.sendall(data.encode('utf-8'))
    response = client_socket.recv(1024).decode('utf-8')
    return json.loads(response)
# 这个代码需要改成调用算法一直计算 Algorithm
# 示例使用
def main():
    client = connect_server()
    try:
        # 连接模拟器
        print(send_command(client, 'connect'))
        time.sleep(1)
        
        # 启用API控制
        print(send_command(client, 'enable_api', {'vehicle_name': 'UAV1'}))
        time.sleep(1)
        
        # 解锁无人机
        print(send_command(client, 'arm', {'vehicle_name': 'UAV1'}))
        time.sleep(1)
        
        # 起飞
        print(send_command(client, 'takeoff', {'vehicle_name': 'UAV1'}))
        time.sleep(5)
        
        # 移动到指定位置
        print(send_command(client, 'move_to_position', {
            'x': 10, 'y': 10, 'z': -5, 
            'vehicle_name': 'UAV1'
        }))
        time.sleep(5)
        
        # 获取状态
        print(send_command(client, 'get_state', {'vehicle_name': 'UAV1'}))
        time.sleep(1)
        
        # 降落
        print(send_command(client, 'land', {'vehicle_name': 'UAV1'}))
        time.sleep(5)
        
        # 上锁
        print(send_command(client, 'arm', {'arm': False, 'vehicle_name': 'UAV1'}))
        time.sleep(1)
        
        # 禁用API控制
        print(send_command(client, 'enable_api', {'enable': False, 'vehicle_name': 'UAV1'}))
        
    finally:
        # 断开连接
        client.close()

if __name__ == '__main__':
    main()