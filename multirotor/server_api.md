# DroneServer API 文档

本文档详细说明了DroneServer的API接口，帮助开发者快速了解如何使用该服务器进行无人机控制。

## 1. 服务器概述

DroneServer是一个基于Socket通信的无人机控制服务器，提供了与AirSim模拟器交互的各种功能，包括无人机的连接、起飞、降落、移动和图像获取等。

## 2. 服务器配置

- **默认监听地址**: `0.0.0.0`
- **默认端口**: `65432`
- **通信协议**: TCP
- **数据格式**: JSON

## 3. 命令格式

所有命令都采用JSON格式，基本结构如下：

```json
{
  "command": "命令名称",
  "params": {
    "参数名1": "参数值1",
    "参数名2": "参数值2",
    ...
  }
}
```

## 4. 响应格式

服务器响应也采用JSON格式，基本结构如下：

```json
{
  "status": "success|error",
  "message": "响应消息",
  "data": {}
  // 其他可能的返回字段
}
```

## 5. 核心命令列表

### 5.1 连接与基础操作

#### 5.1.1 connect - 连接到AirSim模拟器

**请求参数**: 无

**响应**: 
```json
{"status": "success", "message": "已连接到AirSim模拟器"}
```

#### 5.1.2 reset - 重置模拟器状态

**请求参数**: 无

**响应**: 
```json
{"status": "success", "message": "模拟器已重置"}
```

#### 5.1.3 disconnect - 断开连接

**请求参数**: 无

**响应**: 
```json
{"status": "success", "message": "连接已关闭"}
```

### 5.2 无人机控制命令

#### 5.2.1 enable_api - 启用/禁用API控制

**请求参数**: 
```json
{
  "enable": true|false,  // 可选，默认为true
  "vehicle_name": "UAV1"  // 可选，指定无人机名称
}
```

**响应**: 
```json
{"status": "success", "message": "无人机UAV1API控制已启用"}
```

#### 5.2.2 arm - 解锁/上锁无人机

**请求参数**: 
```json
{
  "arm": true|false,  // 可选，默认为true
  "vehicle_name": "UAV1"  // 可选，指定无人机名称
}
```

**响应**: 
```json
{"status": "success", "message": "无人机UAV1已解锁"}
```

#### 5.2.3 takeoff - 无人机起飞

**请求参数**: 
```json
{
  "vehicle_name": "UAV1"  // 可选，指定无人机名称
}
```

**响应**: 
```json
{"status": "success", "message": "无人机UAV1起飞完成"}
```

#### 5.2.4 land - 无人机降落

**请求参数**: 
```json
{
  "vehicle_name": "UAV1"  // 可选，指定无人机名称
}
```

**响应**: 
```json
{"status": "success", "message": "无人机UAV1降落完成"}
```

#### 5.2.5 move_to_position - 移动到指定位置

**请求参数**: 
```json
{
  "x": 10.0,  // X坐标
  "y": 20.0,  // Y坐标
  "z": -5.0,  // Z坐标（负值表示高度）
  "speed": 3.0,  // 可选，移动速度，默认为3
  "vehicle_name": "UAV1"  // 可选，指定无人机名称
}
```

**响应**: 
```json
{"status": "success", "message": "无人机UAV1已移动到(10.0,20.0,-5.0)"}
```

### 5.3 数据获取命令

#### 5.3.1 get_image - 获取图像

**请求参数**: 
```json
{
  "camera_name": "0",  // 可选，相机名称，默认为"0"
  "image_type": "Scene",  // 可选，图像类型，默认为"Scene"
  "vehicle_name": "UAV1"  // 可选，指定无人机名称
}
```

**可用图像类型**: Scene, DepthPlanar, DepthPerspective, DepthVis, DisparityNormalized, Segmentation, SurfaceNormals, Infrared, OpticalFlow, OpticalFlowVis

**响应**: 
```json
{
  "status": "success", 
  "message": "图像获取成功",
  "image_data": "Base64编码的图像数据"
}
```

#### 5.3.2 get_state - 获取无人机状态

**请求参数**: 
```json
{
  "vehicle_name": "UAV1"  // 可选，指定无人机名称
}
```

**响应**: 
```json
{
  "status": "success", 
  "message": "已获取无人机UAV1状态",
  "state": {
    "armed": true,  // 是否解锁
    "flying": true,  // 是否在飞行
    "api_enabled": true,  // API是否启用
    "position": [10.0, 20.0, -5.0]  // 当前位置
  }
}
```

### 5.4 数据存储与转发命令

DroneServer提供了通用的数据存储与转发功能，允许客户端存储、检索和管理任意类型的数据。这些功能可以用于在不同客户端之间共享数据，或临时存储需要在多个操作之间保持的数据。

#### 5.4.1 store_data - 存储数据

**功能说明**: 将任意JSON格式的数据存储到服务器中，返回唯一标识符。

**请求参数**: 
```json
{
  "data_id": "unique_id",  // 可选，数据唯一标识，如未提供则自动生成
  "content": {}  // 要存储的数据内容（必须是有效的JSON格式）
}
```

**响应**: 
```json
{
  "status": "success", 
  "message": "数据已存储，ID: unique_id",
  "data_id": "unique_id"  // 实际使用的数据ID（如未提供则为自动生成的ID）
}
```

#### 5.4.2 retrieve_data - 获取存储的数据

**功能说明**: 根据数据标识检索之前存储的数据内容。

**请求参数**: 
```json
{
  "data_id": "unique_id"  // 要获取的数据标识
}
```

**响应**: 
```json
{
  "status": "success", 
  "message": "成功获取数据，ID: unique_id",
  "data_id": "unique_id",
  "content": {},  // 存储的数据内容
  "timestamp": 123456789  // 存储时间戳（Unix时间，秒级）
}
```

#### 5.4.3 delete_data - 删除存储的数据

**功能说明**: 删除指定标识的数据。

**请求参数**: 
```json
{
  "data_id": "unique_id"  // 要删除的数据标识
}
```

**响应**: 
```json
{
  "status": "success", 
  "message": "已删除数据，ID: unique_id",
  "data_id": "unique_id"
}
```

#### 5.4.4 list_data_ids - 列出所有存储的数据ID

**功能说明**: 获取当前服务器中存储的所有数据的标识列表。

**请求参数**: 无

**响应**: 
```json
{
  "status": "success", 
  "message": "共找到3条数据",
  "count": 3,  // 数据条数
  "data_ids": ["id1", "id2", "id3"]  // 数据ID列表
}
```

#### 5.4.5 clear_all_data - 清空所有存储的数据

**功能说明**: 删除服务器中存储的所有数据（谨慎使用）。

**请求参数**: 无

**响应**: 
```json
{
  "status": "success", 
  "message": "已清空所有存储的数据",
  "cleared_count": 5  // 清空的数据条数
}
```

## 6. 错误处理

当命令执行失败时，服务器会返回错误信息，格式如下：

```json
{"status": "error", "message": "错误描述信息"}
```

常见的错误包括：
- 未连接到AirSim模拟器
- API控制未启用
- 缺少必要的参数
- 参数格式不正确
- 无人机操作失败

## 7. 多无人机支持

DroneServer支持控制多个无人机，通过`vehicle_name`参数来指定要控制的无人机。默认情况下，控制的是名为"UAV1"的无人机。

## 8. 客户端示例代码

以下是一个简单的Python客户端示例，展示如何连接服务器并发送命令：

```python
import socket
import json
import time

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
```

## 9. 注意事项

1. 所有操作都需要先连接到AirSim模拟器
2. 起飞前需要先启用API控制并解锁无人机
3. 移动操作只能在无人机处于飞行状态时执行
4. 图像数据以Base64编码格式返回，需要解码后使用
5. 服务器支持多客户端同时连接，但同一时间对同一无人机的操作可能会产生冲突
6. 对于长时间运行的操作，建议在客户端实现超时处理