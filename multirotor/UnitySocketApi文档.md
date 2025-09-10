# Unity Socket API 文档

## 概述

本文档描述了Unity与Python算法服务器之间的Socket通信协议，明确使用现有的三种数据结构：ScannerConfigData、ScannerRuntimeData和HexGridDataModel，供Unity开发者参考。

## 连接参数

- **服务器地址**: localhost (本地连接)
- **端口号**: 5000
- **通信协议**: TCP/IP
- **数据格式**: JSON

## 通信流程

1. Unity作为客户端连接到Python Socket服务器
2. Unity向服务器发送`grid_data`(HexGridDataModel类型)和`runtime_data`(ScannerRuntimeData类型)
3. Python服务器向Unity发送`config_data`(ScannerConfigData类型)和经过算法计算的`runtime_data`(ScannerRuntimeData类型)

## 数据结构说明

### 1. 三种数据结构

我们将使用三种现有数据结构进行Socket通信：

**1. ScannerConfigData**
- 描述：扫描器的配置数据
- 主要字段：
  - repulsionCoefficient: float - 排斥系数
  - entropyCoefficient: float - 熵系数
  - distanceCoefficient: float - 距离系数
  - leaderRangeCoefficient: float - 领导者范围系数
  - directionRetentionCoefficient: float - 方向保持系数
  - updateInterval: float - 更新间隔
  - moveSpeed: float - 移动速度
  - rotationSpeed: float - 旋转速度
  - scanRadius: float - 扫描半径
  - altitude: float - 高度
  - maxRepulsionDistance: float - 最大排斥距离
  - minSafeDistance: float - 最小安全距离
  - avoidRevisits: bool - 是否避免重访
  - targetSearchRange: float - 目标搜索范围
  - revisitCooldown: float - 重访冷却时间
  
**2. HexGridDataModel**
- 描述：六边形网格数据模型
- 主要字段：
  - cells: List[Dict] - 网格单元列表
    - center: Dict[str, float] - 中心点坐标
    - entropy: float - 熵值
  
**3. ScannerRuntimeData**
- 描述：扫描器运行时数据
- 主要字段：
  - position: Dict[str, float] - 当前位置
  - forward: Dict[str, float] - 前方向量
  - velocity: Dict[str, float] - 速度向量
  - direction: Dict[str, float] - 方向向量
  - leaderPosition: Dict[str, float] - 领导者位置
  - leader_velocity: Dict[str, float] - 领导者速度
  - visitedCells: List[Dict] - 已访问的蜂窝
  - otherScannerPositions: List[Dict] - 其他扫描者位置
  - scoreDir: Dict[str, float] - 分数方向
  - collideDir: Dict[str, float] - 碰撞方向
  - pathDir: Dict[str, float] - 路径方向
  - leaderRangeDir: Dict[str, float] - 领导者范围方向
  - directionRetentionDir: Dict[str, float] - 方向保持方向
  - finalMoveDir: Dict[str, float] - 最终移动方向
  - leaderScanRadius: float - 领导者扫描半径

### 2. 通信协议格式

我们现在使用独立的消息格式发送每种数据类型，每个消息都包含`type`字段来标识数据类型：

**1. 从Unity发送到Python的数据格式**

```json
{
  "type": "unity_to_python",
  "grid_data": {"cells": [...]},    // HexGridDataModel类型数据
  "runtime_data": {...}  // ScannerRuntimeData类型数据
}
```

**2. 从Python发送到Unity的数据格式**

```json
{
  "type": "python_to_unity",
  "timestamp": 1234567890.123,        // 时间戳
  "config_data": {...},               // ScannerConfigData类型数据
  "runtime_data": {...}               // 算法计算后的ScannerRuntimeData类型数据
}
```

### 3. 数据结构详细说明

#### ScannerConfigData 数据结构

```json
{
  "repulsionCoefficient": 2.0,
  "entropyCoefficient": 3.0,
  "distanceCoefficient": 2.0,
  "leaderRangeCoefficient": 3.0,
  "directionRetentionCoefficient": 2.0,
  "updateInterval": 0.2,
  "moveSpeed": 2.0,
  "rotationSpeed": 120.0,
  "scanRadius": 5.0,
  "altitude": 10.0,
  "maxRepulsionDistance": 5.0,
  "minSafeDistance": 2.0,
  "avoidRevisits": true,
  "targetSearchRange": 20.0,
  "revisitCooldown": 60.0
}
```

#### HexGridDataModel 数据结构

```json
{
  "cells": [
    {
      "center": {"x": 0.0, "y": 0.0, "z": 0.0},
      "entropy": 0.5
    }
  ]
}
```

#### ScannerRuntimeData 数据结构

```json
{
  "position": {"x": 0.0, "y": 0.0, "z": 0.0},
  "forward": {"x": 0.0, "y": 0.0, "z": 1.0},
  "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
  "direction": {"x": 0.0, "y": 0.0, "z": 1.0},
  "leaderPosition": {"x": 0.0, "y": 0.0, "z": 0.0},
  "leader_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
  "visitedCells": [],
  "otherScannerPositions": [],
  "scoreDir": {"x": 0.0, "y": 0.0, "z": 0.0},
  "collideDir": {"x": 0.0, "y": 0.0, "z": 0.0},
  "pathDir": {"x": 0.0, "y": 0.0, "z": 0.0},
  "leaderRangeDir": {"x": 0.0, "y": 0.0, "z": 0.0},
  "directionRetentionDir": {"x": 0.0, "y": 0.0, "z": 0.0},
  "finalMoveDir": {"x": 0.0, "y": 0.0, "z": 1.0},
  "leaderScanRadius": 5.0
}

## Unity客户端实现示例

以下是Unity中实现Socket通信的C#示例代码，使用与Python端对应的三个数据结构：

```csharp
using System;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using Newtonsoft.Json;
using System.Threading;

public class PythonSocketClient : MonoBehaviour
{
    private TcpClient socketClient;
    private NetworkStream stream;
    private Thread receiveThread;
    private bool isConnected = false;
    private string serverAddress = "localhost";
    private int serverPort = 5000;

    // 连接到Python服务器
    public bool Connect()
    {
        try
        {
            socketClient = new TcpClient(serverAddress, serverPort);
            stream = socketClient.GetStream();
            isConnected = true;

            // 启动接收线程
            receiveThread = new Thread(ReceiveData);
            receiveThread.IsBackground = true;
            receiveThread.Start();

            Debug.Log("已连接到Python服务器");
            return true;
        }
        catch (Exception e)
        {
            Debug.LogError("连接Python服务器失败: " + e.Message);
            return false;
        }
    }

    // 断开连接
    public void Disconnect()
    {
        isConnected = false;

        if (stream != null)
        {
            stream.Close();
            stream = null;
        }

        if (socketClient != null)
        {
            socketClient.Close();
            socketClient = null;
        }

        if (receiveThread != null && receiveThread.IsAlive)
        {
            receiveThread.Abort();
            receiveThread = null;
        }

        Debug.Log("已断开与Python服务器的连接");
    }

    // 发送数据到Python服务器
    public void SendData(Dictionary<string, object> data)
    {
        if (!isConnected || stream == null)
        {
            Debug.LogWarning("未连接到服务器，无法发送数据");
            return;
        }

        try
        {
            string jsonData = JsonConvert.SerializeObject(data);
            byte[] buffer = Encoding.UTF8.GetBytes(jsonData);
            stream.Write(buffer, 0, buffer.Length);
            Debug.Log("已发送数据到Python服务器: " + jsonData);
        }
        catch (Exception e)
        {
            Debug.LogError("发送数据时出错: " + e.Message);
            isConnected = false;
        }
    }

    // 接收来自Python服务器的数据
    private void ReceiveData()
    {
        byte[] buffer = new byte[4096];

        while (isConnected)
        {
            try
            {
                if (stream.DataAvailable)
                {
                    int bytesRead = stream.Read(buffer, 0, buffer.Length);
                    string jsonData = Encoding.UTF8.GetString(buffer, 0, bytesRead);
                    Debug.Log("接收到Python服务器数据: " + jsonData);

                    // 解析JSON数据
                    var receivedData = JsonConvert.DeserializeObject<Dictionary<string, object>>(jsonData);
                    ProcessReceivedData(receivedData);
                }
                else
                {
                    Thread.Sleep(10);  // 减少CPU占用
                }
            }
            catch (Exception e)
            {
                Debug.LogError("接收数据时出错: " + e.Message);
                isConnected = false;
            }
        }
    }

    // 处理接收到的数据
    private void ProcessReceivedData(Dictionary<string, object> data)
    {
        // 检查数据类型
        if (data.ContainsKey("type") && data["type"].ToString() == "config_and_runtime_data")
        {
            // 获取配置数据
            if (data.ContainsKey("config_data"))
            {
                var configData = JsonConvert.DeserializeObject<ScannerConfigData>(data["config_data"].ToString());
                // 处理配置数据...
            }

            // 获取运行时数据
            if (data.ContainsKey("runtime_data"))
            {
                var runtimeData = JsonConvert.DeserializeObject<ScannerRuntimeData>(data["runtime_data"].ToString());
                // 处理运行时数据...
            }
        }
    }

    // 发送网格数据和运行时数据到Python服务器的示例
    public void SendGridAndRuntimeData(HexGridDataModel gridData, ScannerRuntimeData runtimeData)
    {
        var data = new Dictionary<string, object>
        {
            { "grid_data", gridData },
            { "runtime_data", runtimeData }
        };
        SendData(data);
    }
}

// 对应Python端的ScannerConfigData类
[Serializable]
public class ScannerConfigData
{
    // 系数参数
    public float repulsionCoefficient = 2.0f;
    public float entropyCoefficient = 3.0f;
    public float distanceCoefficient = 2.0f;
    public float leaderRangeCoefficient = 3.0f;
    public float directionRetentionCoefficient = 2.0f;
    public float updateInterval = 0.2f;

    // 运动参数
    public float moveSpeed = 2.0f;
    public float rotationSpeed = 120.0f;
    public float scanRadius = 5.0f;
    public float altitude = 10.0f;

    // 距离参数
    public float maxRepulsionDistance = 5.0f;
    public float minSafeDistance = 2.0f;

    // 目标选择策略
    public bool avoidRevisits = true;
    public float targetSearchRange = 20.0f;
    public float revisitCooldown = 60.0f;
}

// 对应Python端的ScannerRuntimeData类
[Serializable]
public class ScannerRuntimeData
{
    // 方向向量
    public Vector3 scoreDir = Vector3.zero;
    public Vector3 collideDir = Vector3.zero;
    public Vector3 pathDir = Vector3.zero;
    public Vector3 leaderRangeDir = Vector3.zero;
    public Vector3 directionRetentionDir = Vector3.zero;
    public Vector3 finalMoveDir = Vector3.zero;
    public Vector3 direction = Vector3.forward;
    public Vector3 velocity = Vector3.zero;

    // 当前位置和方向
    public Vector3 position = Vector3.zero;
    public Vector3 forward = Vector3.forward;

    // Leader信息
    public Vector3 leaderPosition = Vector3.zero;
    public float leaderScanRadius = 0.0f;
    public Vector3 leader_velocity = Vector3.zero;

    // 已访问记录和其它扫描者坐标
    public List<Vector3> visitedCells = new List<Vector3>();
    public List<Vector3> otherScannerPositions = new List<Vector3>();
}

// 对应Python端的HexGridDataModel类
[Serializable]
public class HexGridDataModel
{
    public List<HexCell> cells = new List<HexCell>();
}

// 对应Python端的HexCell类
[Serializable]
public class HexCell
{
    public Vector3 center = Vector3.zero;
    public float entropy = 0.0f;
}
```

## 注意事项

1. 请确保在使用Socket通信前，Python算法服务器已经启动
2. 使用JSON序列化/反序列化时，需要确保Unity项目中包含Newtonsoft.Json库
3. 为了避免Unity主线程阻塞，推荐在单独的线程中处理Socket通信
4. 请处理好连接断开、重连等异常情况
5. 数据格式需要严格遵守本文档中定义的结构，使用与Python端对应的三个数据结构

## 常见问题排查

1. **无法连接到服务器**: 检查Python服务器是否已启动，端口号是否正确
2. **数据解析错误**: 确保JSON数据格式与文档中定义的一致，特别是要使用正确的三个数据结构
3. **连接不稳定**: 检查网络环境，增加重连机制

## 版本信息

- 版本: 1.1
- 发布日期: 2023-10-10
- 更新内容: 使用现有的ScannerConfigData、ScannerRuntimeData和HexGridDataModel数据结构重写文档