# DQN CPU优化指南

## 🎯 问题描述

在没有独立显卡的环境中（如AMD 6800H集成显卡），DQN训练可能会遇到以下问题：
- 程序卡住或响应缓慢
- CPU占用率过高
- PyTorch初始化时间过长

## ✅ 已实施的优化

### 1. 强制使用CPU设备
```python
# 设备管理：强制使用CPU，避免在没有GPU的环境中卡住
self.device = torch.device('cpu')
```

### 2. 限制线程数
```python
# 设置PyTorch线程数，避免CPU占用过高
torch.set_num_threads(2)  # 限制为2个线程，避免占用所有CPU核心
```

### 3. 梯度裁剪
```python
# 梯度裁剪，防止梯度爆炸（对CPU训练特别重要）
torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
```

### 4. 设备一致性
- 所有张量创建时直接指定设备：`.to(self.device)`
- 模型加载时强制映射到CPU：`map_location=self.device`

## 🔧 配置选项

### 禁用DQN学习（推荐）

如果您的设备性能有限，建议完全禁用DQN学习，使用传统的人工势场算法：

**文件**: `multirotor/scanner_config.json`

```json
{
    "dqn": {
        "enabled": false,  // ← 设置为 false 禁用DQN
        ...
    }
}
```

### 启用DQN学习（轻量级模式）

如果您想启用DQN，建议使用以下轻量级配置：

```json
{
    "dqn": {
        "enabled": true,
        "learning_rate": 0.001,
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_min": 0.1,       // ← 提高最小探索率，减少训练
        "epsilon_decay": 0.99,    // ← 加快探索率衰减
        "batch_size": 32,         // ← 减小批次大小（从64降到32）
        "target_update": 20,      // ← 降低更新频率（从10增加到20）
        "memory_capacity": 5000,  // ← 减小内存容量（从10000降到5000）
        "model_save_interval": 2000  // ← 降低保存频率
    }
}
```

## 📊 性能对比

| 配置 | CPU占用 | 内存占用 | 训练速度 | 推荐场景 |
|------|---------|----------|----------|----------|
| **禁用DQN** | 5-10% | ~100MB | N/A | 生产环境、低性能设备 |
| **轻量级DQN** | 20-30% | ~200MB | 慢 | 开发测试、学习实验 |
| **标准DQN** | 40-60% | ~500MB | 中等 | 高性能CPU |
| **GPU加速** | 10-15% | ~1GB | 快 | 有独立显卡 |

## 🚀 性能调优技巧

### 1. 调整线程数
```python
# 根据您的CPU核心数调整
torch.set_num_threads(2)  # 2线程：低性能设备
torch.set_num_threads(4)  # 4线程：中等性能设备
torch.set_num_threads(8)  # 8线程：高性能设备
```

### 2. 降低网络复杂度
修改 `DQN` 类的隐藏层维度：
```python
# 默认
self.fc1 = nn.Linear(state_dim, 128)  # 128维隐藏层

# 轻量级
self.fc1 = nn.Linear(state_dim, 64)   # 64维隐藏层
```

### 3. 增加更新间隔
在 `AlgorithmServer.py` 中调整：
```python
self.config_data.updateInterval = 1.0  # 增加到1秒
```

## 🔍 故障排查

### 问题：程序启动时卡住

**原因**：PyTorch尝试检测CUDA设备

**解决方案**：
1. 确认配置文件中 `dqn.enabled` 为 `false`
2. 设置环境变量：
   ```bash
   # Windows PowerShell
   $env:CUDA_VISIBLE_DEVICES = ""
   
   # Linux/Mac
   export CUDA_VISIBLE_DEVICES=""
   ```

### 问题：CPU占用率过高

**原因**：PyTorch默认使用所有可用核心

**解决方案**：
1. 降低线程数（修改 `DqnLearning.py` 中的 `torch.set_num_threads(2)`）
2. 增加更新间隔
3. 减小批次大小

### 问题：内存占用过高

**原因**：经验回放缓冲区过大

**解决方案**：
1. 减小 `memory_capacity`（从10000降到5000或更少）
2. 减小 `batch_size`
3. 降低状态维度（如果可能）

## 💡 最佳实践

### 开发阶段
- **禁用DQN**：专注于调试APF算法逻辑
- **使用可视化**：观察无人机行为

### 测试阶段
- **启用轻量级DQN**：验证DQN集成是否正常
- **短时间运行**：收集少量训练数据

### 生产阶段
- **禁用DQN**：使用经过调优的APF参数
- **或使用预训练模型**：加载已训练好的DQN模型

## 📝 推荐配置

### AMD 6800H（无独显）推荐配置

```json
{
    "dqn": {
        "enabled": false  // 推荐禁用
    },
    "updateInterval": 1.0,  // 降低更新频率
    "moveSpeed": 2.0
}
```

### 如果必须使用DQN

```python
# 修改 DqnLearning.py
torch.set_num_threads(2)  # 限制线程数

# 配置文件
{
    "dqn": {
        "enabled": true,
        "batch_size": 16,      // 最小批次
        "memory_capacity": 2000,  // 最小容量
        "target_update": 50    // 最低更新频率
    }
}
```

## ⚠️ 注意事项

1. **DQN学习需要大量计算资源**，在无独显的笔记本上运行可能会：
   - 显著降低整体系统性能
   - 增加CPU温度和风扇噪音
   - 延长训练时间

2. **推荐使用传统APF算法**，它已经过优化且效率高

3. **如需机器学习优化**，建议：
   - 在有GPU的机器上训练DQN模型
   - 导出训练好的模型
   - 在目标设备上仅使用推理（inference）模式

## 🎓 相关文档

- [DQN集成文档](./README.md)
- [配置指南](../Configuration_Guide.md)
- [主README](../../README.MD)

