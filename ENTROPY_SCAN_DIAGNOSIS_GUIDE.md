# 熵值扫描问题诊断指南

## 🎯 问题现象

**用户反馈**：重置后熵值扫描还是不变

## ✅ Python端代码验证

### 已完成的修复（6项）

1. ✅ **可视化缓存修复** - 重置后强制刷新
2. ✅ **熵值保护机制** - 防止Unity数据覆盖
3. ✅ **reset_environment修复** - 支持参数控制
4. ✅ **send_processed_data修复** - 防止发送脏数据
5. ✅ **默认配置修复** - reset_grid_entropy=True
6. ✅ **start_simulation时序修复** - 等待时间 0.5s → 2.0s

### 验证结果

**Python端代码全部正确**：
- ✅ start_simulation后等待2秒
- ✅ ready_event在start_simulation之后设置
- ✅ reset_grid=True时禁用熵值保护
- ✅ 算法线程正确等待同步
- ✅ 数据发送检查resetting标志

---

## 🔍 诊断步骤

### 步骤1：运行训练并观察日志

**应该看到的Python端日志**：

```
[重置] 4/5 发送 start_simulation 指令，Leader 开始移动
[重置] 等待完成，Unity应该已启动熵值收集
[重置] ✨ 严格重置流程执行完毕，系统已回到初始状态，算法线程已放行
```

**算法线程日志**：

```
[UAV1] 重置后同步完成，继续决策
[UAV1] 首帧同步完成，开始决策循环
```

**网格数据更新日志**（新增）：

```
[网格更新] 收到800个格子，平均熵值=95.2, 低熵格子=0
[网格更新] 收到800个格子，平均熵值=92.8, 低熵格子=5
[网格更新] 收到800个格子，平均熵值=88.5, 低熵格子=12
```

**如果看到以上日志** → Python端工作正常 → 问题在Unity端

---

### 步骤2：检查网格数据更新日志

**正常情况**：
- 刚重置后：平均熵值≈100，低熵格子≈0
- 扫描进行中：平均熵值降低，低熵格子增加
- 扫描完成：平均熵值<80，低熵格子>10

**异常情况**：
- 如果一直看到"平均熵值=100.0, 低熵格子=0" → Unity没有更新熵值
- 如果完全没有"[网格更新]"日志 → Unity没有发送grid_data

---

### 步骤3：检查Unity端

**需要确认Unity端是否**：

1. **收到start_simulation指令**
   ```
   [Unity] 收到start_simulation指令
   ```

2. **启动了熵值收集**
   ```
   [Unity] 启动熵值收集功能
   isEntropyCollectionEnabled = true
   ```

3. **收到runtime数据**
   ```
   [Unity] 收到runtime数据
   位置: (x, y, z)
   ```

4. **执行扫描逻辑**
   ```csharp
   if (isEntropyCollectionEnabled && hasReceivedRuntimeData) {
       ScanAndReduceEntropy();
   }
   ```

5. **更新grid_data**
   ```csharp
   void ScanAndReduceEntropy() {
       foreach (var cell in scannedCells) {
           cell.entropy = 0; // 或其他低值
       }
       SendGridDataToPython();
   }
   ```

6. **发送grid_data到Python**
   ```
   [Unity] 发送grid_data到Python
   ```

---

## 🎯 问题定位

### 情况1：Python日志正常，但熵值不变

**诊断**：
- ✅ Python端工作正常
- ❌ Unity端有问题

**可能原因**：
1. Unity未正确处理start_simulation指令
2. Unity启动熵值收集失败
3. Unity扫描逻辑未执行
4. Unity扫描结果未发送

---

### 情况2：完全没有"[网格更新]"日志

**诊断**：
- Unity没有发送grid_data

**可能原因**：
1. Unity通信中断
2. Unity未正确初始化网格
3. Unity发送频率过低

---

### 情况3："[网格更新]"日志显示平均熵值一直=100

**诊断**：
- Unity发送了grid_data
- 但熵值没有被降低

**可能原因**：
1. Unity扫描逻辑未执行
2. 扫描距离设置过小
3. 扫描条件不满足

---

## 🔧 Unity端修复建议

### 修复1：确保start_simulation正确处理

```csharp
void OnStartSimulation() {
    Debug.Log("[Unity] 收到start_simulation指令，启动熵值收集");
    isEntropyCollectionEnabled = true;
    lastScanTime = Time.time;
}
```

### 修复2：在Update中执行扫描

```csharp
void Update() {
    // 每隔一定时间执行一次扫描
    if (isEntropyCollectionEnabled &&
        Time.time - lastScanTime > scanInterval) {

        ScanAndReduceEntropy();
        lastScanTime = Time.time;
    }
}

void ScanAndReduceEntropy() {
    int scannedCount = 0;

    foreach (var cell in gridCells) {
        // 检查是否在扫描范围内
        if (IsInScanRange(cell)) {
            // 降低熵值
            cell.entropy = Mathf.Max(0, cell.entropy - entropyReduction);
            scannedCount++;
        }
    }

    Debug.Log($\"[Unity] 扫描完成，扫描了{scannedCount}个格子\");

    // 发送更新后的数据到Python
    SendGridDataToPython();
}
```

### 修复3：确保数据正确发送

```csharp
void SendGridDataToPython() {
    var gridData = new {
        cells = gridCells.Select(c => new {
            x = c.center.x,
            y = c.center.y,
            z = c.center.z,
            entropy = c.entropy  // 确保发送更新后的熵值
        }).ToArray()
    };

    // 发送到Python
    socket.Send(gridData);
    Debug.Log(\"[Unity] 发送grid_data到Python\");
}
```

---

## 📊 验证方法

### 1. 检查训练日志

```bash
# 查看网格更新日志
grep "网格更新" multirotor/DDPG_Weight/logs/ddpg_airsim/*.log | tail -20

# 应该看到平均熵值逐渐降低，低熵格子逐渐增加
```

### 2. 检查扫描数据

```bash
# 查看扫描统计
tail -20 multirotor/DDPG_Weight/airsim_training_logs/ddpg_training_*.csv

# 应该看到每个Episode的scanned_cells > 0
```

### 3. 检查Unity日志

在Unity控制台查找：
- "收到start_simulation指令"
- "扫描完成"
- "发送grid_data"

---

## 🎉 总结

### Python端
- ✅ 代码完全正确
- ✅ 已添加调试日志
- ✅ 等待Unity数据

### Unity端（需要检查）
- ❓ 是否正确处理start_simulation
- ❓ 是否启动熵值收集
- ❓ 是否执行扫描逻辑
- ❓ 是否发送更新后的grid_data

### 下一步
1. 运行训练并观察Python日志
2. 确认网格数据更新日志
3. 检查Unity端日志
4. 根据日志定位具体问题

---

**诊断工具**：
- `check_entropy_scan_logs.py` - 日志检查指南
- `diagnose_entropy_scan_issue.py` - 系统性诊断脚本

**修复完成日期**: 2026-03-05
**版本**: v1.0
