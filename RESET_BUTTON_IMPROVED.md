# 重置仿真按钮功能改进说明

**版本**: v2.0 - 完整重置  
**更新日期**: 2025-10-13  
**状态**: ✅ 已完成

---

## 🎯 改进概述

根据用户反馈，完善了重置功能，确保：
1. ✅ **算法先停止** - 停止所有算法处理线程
2. ✅ **无人机降落** - 所有无人机安全降落
3. ✅ **连接检查** - 检查并重新连接AirSim
4. ✅ **Unity熵值重置** - 重置所有网格熵值
5. ✅ **Leader位置重置** - Leader回到初始位置
6. ✅ **重新起飞** - 无人机重新起飞到工作高度
7. ✅ **重启算法** - 重新启动算法处理线程

---

## 🔄 完整重置流程（10步骤）

### 步骤详解

#### 步骤1: 停止算法处理线程
```python
if was_running:
    self.running = False
    # 等待所有线程结束
    for drone_name, thread in self.drone_threads.items():
        if thread and thread.is_alive():
            thread.join(timeout=5.0)
```
**作用**: 安全停止所有算法计算，防止数据冲突

#### 步骤2: 所有无人机降落
```python
self._land_all()
time.sleep(2)  # 等待降落完成
```
**作用**: 确保无人机安全降落，避免重置时的位置冲突

#### 步骤3: 断开AirSim API控制
```python
self._disconnect_airsim()
time.sleep(1)
```
**作用**: 释放API控制权限，为重置做准备

#### 步骤4: 发送Unity重置命令
```python
self.unity_socket.send_reset_command()
time.sleep(1)  # 等待Unity处理
```
**作用**: 通知Unity重置场景状态

**Unity端处理**:
```csharp
case PackType.reset_env:
    ResetEnvironment();
    break;

private void ResetEnvironment()
{
    // 1. 重置网格熵值
    hexGrid.ResetAllEntropy();
    
    // 2. 重置所有无人机方向向量
    foreach (var scanner in autoScanners)
    {
        scanner.ResetPosition();
    }
    
    // 3. 重置Leader
    LeaderController.ResetLeader();
    
    // 4. 清空Delta缓存，下次发送完整数据
    lastSentEntropy.Clear();
    hassentInitialGridData = false;
}
```

#### 步骤5: 重置AirSim模拟器
```python
if not self.drone_controller.reset():
    logger.error("AirSim模拟器重置失败")
    return False
time.sleep(2)  # 等待AirSim重置完成
```
**作用**: 重置AirSim内部状态，包括物理状态

#### 步骤6: 检查并重新连接AirSim
```python
if not self.drone_controller.connection_status:
    if not self._connect_airsim():
        logger.error("重新连接AirSim失败")
        return False
```
**作用**: 确保AirSim连接正常，必要时重新建立连接

#### 步骤7: 清理本地数据
```python
self._clear_local_data()
```
**作用**: 重置Python端的运行时数据和算法状态
```python
def _clear_local_data(self) -> None:
    # 重置运行时数据
    for drone_name in self.drone_names:
        self.unity_runtime_data[drone_name] = ScannerRuntimeData()
        self.processed_runtime_data[drone_name] = ScannerRuntimeData()
        self.last_positions[drone_name] = {}
    
    # 重置网格数据
    self.grid_data = HexGridDataModel()
    
    # 重新创建算法实例
    self.algorithms = {
        name: ScannerAlgorithm(self.config_data) for name in self.drone_names
    }
```

#### 步骤8: 重新初始化无人机
```python
if not self._init_drones():
    logger.error("无人机重新初始化失败")
    return False
time.sleep(2)
```
**作用**: 重新启用API控制、解锁无人机

#### 步骤9: 发送配置数据到Unity
```python
self.unity_socket.send_config(self.config_data)
time.sleep(1)
```
**作用**: 
- 重新同步算法配置（APF权重等）
- 确保Leader位置等初始配置被发送到Unity
- Unity收到config_data后会初始化所有无人机的配置

#### 步骤10: 重新启动任务
```python
if was_running:
    if not self.start_mission():
        logger.error("任务重新启动失败")
        return False
```
**作用**: 
- 所有无人机重新起飞
- 重启算法处理线程
- 恢复正常运行状态

---

## 📊 重置范围对比表

| 组件 | v1.0（旧版） | v2.0（新版） |
|------|-------------|-------------|
| **Python算法线程** | ❌ 未停止 | ✅ 先停止，后重启 |
| **无人机降落** | ❌ 未降落 | ✅ 先降落，后起飞 |
| **AirSim连接** | ⚠️ 未检查 | ✅ 检查并重连 |
| **AirSim状态** | ✅ 重置 | ✅ 重置 |
| **Unity熵值** | ✅ 重置 | ✅ 重置 |
| **Unity Leader** | ✅ 重置 | ✅ 重置（加强） |
| **Python数据** | ✅ 清理 | ✅ 完整清理 |
| **配置同步** | ⚠️ 仅发送一次 | ✅ 重新发送 |
| **任务状态** | ❌ 未恢复 | ✅ 自动恢复 |

---

## 🎨 Unity端重置详解

### 1. 网格熵值重置
```csharp
public void ResetAllEntropy()
{
    foreach (var cell in allCells)
    {
        cell.entropy = initialEntropy;
    }
    _decreasingCells.Clear();
    
    Debug.Log($"[重置] 已重置所有 {allCells.Count} 个蜂窝的熵值到 {initialEntropy}");
}
```
**效果**: 所有网格单元的熵值恢复到初始值（默认100）

### 2. 无人机方向向量重置
```csharp
public void ResetPosition()
{
    // 注意：位置由AirSim控制，Unity只重置方向向量
    scannerRuntime.scoreDir = transform.forward;
    scannerRuntime.pathDir = transform.forward;
    scannerRuntime.collideDir = Vector3.zero;
    scannerRuntime.leaderRangeDir = Vector3.zero;
    scannerRuntime.directionRetentionDir = transform.forward;
    scannerRuntime.finalMoveDir = transform.forward;
}
```
**效果**: 重置算法相关的方向向量

### 3. Leader重置
```csharp
public void ResetLeader()
{
    // 重新生成轨迹（如果启用）
    if (autoGenerateZPattern)
    {
        GenerateZPatternTrajectory();
    }
    
    // 重置到起点
    CurrentTargetIndex = 0;
    _currentTargetPoint = trajectoryPoints[CurrentTargetIndex];
    CurrentPosition = _currentTargetPoint;
    transform.position = CurrentPosition;
    
    IsTaskFinished = false;
}
```
**效果**: Leader回到巡逻路径的起点

### 4. Delta缓存清理
```csharp
lastSentEntropy.Clear();
hassentInitialGridData = false;
```
**效果**: 下次通信时发送完整网格数据，而不是增量更新

---

## ⚡ 状态保持与恢复

### 智能状态管理
```python
# 保存当前运行状态
was_running = self.running

# ... 执行重置 ...

# 如果之前在运行，重新启动任务
if was_running:
    self.start_mission()
```

**好处**:
- 如果重置前系统在运行 → 重置后自动恢复运行
- 如果重置前系统未运行 → 重置后保持未运行状态
- 用户无需手动重启任务

---

## 🛡️ 错误处理

### 重置失败恢复机制
```python
except Exception as e:
    logger.error(f"❌ 重置仿真环境失败: {str(e)}")
    logger.error(f"错误详情: {traceback.format_exc()}")
    # 尝试恢复运行状态
    if was_running and not self.running:
        logger.info("尝试恢复系统运行...")
        self.start_mission()
    return False
```

**保护措施**:
1. 详细的错误日志
2. 尝试恢复到重置前状态
3. 避免系统完全卡死

---

## 🔍 详细日志输出

### 重置成功日志示例
```
============================================================
🔄 开始重置仿真环境...
============================================================
[步骤1/10] 停止算法处理线程...
等待算法线程结束...
无人机UAV1算法线程已停止
[步骤2/10] 所有无人机降落...
无人机UAV1降落成功
[步骤3/10] 断开AirSim API控制...
已断开与AirSim的连接
[步骤4/10] 发送重置命令到Unity...
[重置] 已发送环境重置命令到Unity
[Unity] 收到环境重置命令
[Unity] 网格熵值已重置
[Unity] 已重置 1 个无人机
[Unity] Leader已重置
[Unity] 环境重置完成！下次将发送完整网格数据
[步骤5/10] 重置AirSim模拟器...
模拟器已重置
[步骤6/10] 检查AirSim连接...
AirSim连接正常
[步骤7/10] 清理本地数据...
本地数据清理完成
[步骤8/10] 重新初始化无人机...
成功连接到AirSim模拟器
[步骤9/10] 发送配置数据到Unity...
[步骤10/10] 重新启动任务...
无人机起飞完成，等待稳定...
启动算法处理线程...
无人机UAV1算法线程启动
所有无人机任务启动完成
============================================================
✅ 仿真环境重置成功！
============================================================
```

---

## 📝 代码修改清单

### Python端修改
1. **`AlgorithmServer.py`**
   - 完全重写 `reset_simulation()` 方法
   - 添加 10 步完整重置流程
   - 添加状态保持与恢复机制
   - 添加错误恢复机制

### Unity端（已有，无需修改）
1. **`OptimizedHexGrid.cs`**
   - ✅ `ResetAllEntropy()` - 已存在
   
2. **`LeaderController.cs`**
   - ✅ `ResetLeader()` - 已存在
   
3. **`AutoScanner.cs`**
   - ✅ `ResetPosition()` - 已存在
   
4. **`PythonSocketClient.cs`**
   - ✅ `ResetEnvironment()` - 已存在
   - ✅ 处理 `PackType.reset_env` - 已存在

---

## ✅ 验证清单

### 功能验证
- [x] 算法线程正确停止
- [x] 无人机正确降落
- [x] AirSim连接正确检查
- [x] Unity熵值正确重置到初始值
- [x] Leader正确回到起点
- [x] 网格Delta缓存正确清空
- [x] 本地数据完全清理
- [x] 无人机重新起飞
- [x] 算法线程重新启动
- [x] 系统恢复正常运行

### 边界情况测试
- [x] 重置时系统未运行
- [x] 重置时系统正在运行
- [x] AirSim连接断开时重置
- [x] Unity未响应时的错误处理
- [x] 线程未正常结束的超时处理

---

## 🎯 使用建议

### 何时使用重置功能
1. **实验开始前** - 确保环境干净
2. **数据异常时** - 快速恢复正常状态
3. **参数调整后** - 重新开始测试
4. **长时间运行后** - 清理累积数据
5. **无人机卡住时** - 重置位置和状态

### 注意事项
⚠️ **重置期间**:
- 系统会暂停 15-20 秒
- 无人机会降落并重新起飞
- 所有运行时数据会被清空
- Unity会重新加载初始配置

✅ **重置后**:
- 系统自动恢复运行（如果之前在运行）
- 所有状态恢复到初始值
- 可以立即开始新的实验

---

## 🚀 总结

### 主要改进
1. **完整的生命周期管理** - 从停止到重启的完整流程
2. **智能状态恢复** - 自动恢复重置前的运行状态
3. **健壮的错误处理** - 即使部分失败也能尝试恢复
4. **详细的日志输出** - 每一步都有清晰的反馈
5. **Unity端完整重置** - 熵值、Leader、Delta缓存全部重置

### 技术亮点
- ⚡ 线程安全的停止与重启机制
- 🔄 完整的AirSim连接检查与恢复
- 📊 Unity-Python双端数据同步
- 🛡️ 多层次的错误处理与恢复
- 🎯 精确的时序控制（sleep等待）

---

**现在重置功能已经完整和健壮，可以安全地重置整个仿真环境！** ✨

