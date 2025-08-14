# 无人机控制服务器接口文档

## 服务器基本信息

- **通信方式**：TCP
- **监听地址**：0.0.0.0
- **监听端口**：65432
- **数据格式**：JSON（请求与响应均为JSON格式，编码为UTF-8）
- **启动方式**：

 ```bash
 python .\multirotor\MyDroneServer.py
  ```

## 通用请求格式

客户端通过TCP发送JSON格式的命令，结构如下：

```json
{
  "command": "命令名称",   // 字符串类型，必填，命令名称（大小写不敏感）
  "params": {}            // 对象类型，可选，命令所需参数
}
```

## 通用响应格式

服务器处理命令后返回JSON格式响应，结构如下：

```json
{
  "status": "success|error",  // 字符串类型，标识命令执行结果
  "message": "描述信息"        // 字符串类型，结果描述或错误信息
}
```

## 接口详情

### 1. 连接到AirSim模拟器

- **命令名称**：`connect`
- **功能**：建立与AirSim模拟器的连接
- **参数**：无
- **请求示例**：

  ```json
  {
    "command": "connect",
    "params": {}
  }
  ```

- **响应示例**：
  - 成功：

    ```json
    {
      "status": "success",
      "message": "已连接到AirSim模拟器"
    }
    ```

  - 失败：
  
    ```json
    {
      "status": "error",
      "message": "连接失败"
    }
    ```

### 2. 重置模拟器状态

- **命令名称**：`reset`
- **功能**：重置AirSim模拟器的状态
- **参数**：无
- **请求示例**：
  
  ```json
  {
    "command": "reset",
    "params": {}
  }
  ```

- **响应示例**：
  
  ```json
  {
    "status": "success",
    "message": "模拟器已重置"
  }
  ```

### 3. 启用/禁用API控制

- **命令名称**：`enable_api`
- **功能**：启用或禁用对无人机的API控制
- **参数**：
  - `enable`：布尔类型，可选，默认值为`true`。`true`表示启用API控制，`false`表示禁用
- **请求示例**：
  
  ```json
  {
    "command": "enable_api",
    "params": {
      "enable": true,
      "vehicle_name": "UAV1"
    }
  }
  ```

- **响应示例**：
  
  ```json
  {
    "status": "success",
    "message": "API控制已启用"
  }
  ```

### 4. 无人机解锁/上锁

- **命令名称**：`arm`
- **功能**：控制无人机解锁（可起飞）或上锁（不可起飞）
- **参数**：
  - `arm`：布尔类型，可选，默认值为`true`。`true`表示解锁，`false`表示上锁
- **请求示例**：
  
  ```json
  {
    "command": "arm",
    "params": {
      "arm": true,
      "vehicle_name": "UAV1"
    }
  }
  ```

- **响应示例**：
  
  ```json
  {
    "status": "success",
    "message": "无人机已解锁"
  }
  ```


### 5. 无人机起飞

- **命令名称**：`takeoff`
- **功能**：控制无人机完成起飞动作
- **参数**：无
- **请求示例**：
  
  ```json
  {
    "command": "takeoff",
    "params": {
       "vehicle_name": "UAV1"
      }
  }
  ```

- **响应示例**：

  ```json
  {
    "status": "success",
    "message": "起飞完成"
  }
  ```

### 6. 无人机降落

- **命令名称**：`land`
- **功能**：控制无人机完成降落动作
- **参数**：无
- **请求示例**：
  
  ```json
  {
    "command": "land",
    "params": { 
      "vehicle_name": "UAV1"
    }
  }
  ```

- **响应示例**：
  
  ```json
  {
    "status": "success",
    "message": "降落完成"
  }
  ```

### 7. 移动到指定位置

- **命令名称**：`move_to_position`
- **功能**：控制无人机移动到指定的三维坐标位置
- **参数**：
  - `x`：数值类型，必填，目标位置X坐标
  - `y`：数值类型，必填，目标位置Y坐标
  - `z`：数值类型，必填，目标位置Z坐标
  - `speed`：数值类型，可选，移动速度，默认值为3
- **请求示例**：
  
  ```json
  {
    "command": "move_to_position",
    "params": {
      "x": 10,
      "y": 20,
      "z": -5,
      "speed": 2,
      "vehicle_name": "UAV1"
    }
  }
  ```

- **响应示例**：

  ```json
  {
    "status": "success",
    "message": "已移动到(10,20,-5)"
  }
  ```

### 8. 断开连接

- **命令名称**：`disconnect`
- **功能**：关闭客户端与服务器的连接
- **参数**：无
- **请求示例**：
  
  ```json
  {
    "command": "disconnect",
    "params": {
      "vehicle_name": "UAV1"
      }
  }
  ```

- **响应示例**：
  
  ```json
  {
    "status": "success",
    "message": "连接已关闭"
  }
  ```

## 错误处理

- 当客户端发送的JSON格式无效时，服务器返回：
  
  ```json
  {
    "status": "error",
    "message": "无效的JSON格式"
  }
  ```

- 当客户端发送未知命令时，服务器返回：
  
  ```json
  {
    "status": "error",
    "message": "未知命令: [命令名称]"
  }
  ```

- 其他操作错误（如参数缺失、文件读取失败等），错误信息将在`message`字段中详细说明
