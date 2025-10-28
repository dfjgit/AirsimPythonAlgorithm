# 🎨 可视化功能完整指南

## 📋 快速索引

| 文档 | 内容 | 适用场景 |
|------|------|---------|
| **本文件** | 快速入门和文件说明 | 新用户必读 |
| [训练可视化说明.md](训练可视化说明.md) | TrainingVisualizer详细文档 | 训练时使用 |
| [模型使用指南.md](模型使用指南.md) | 模型训练和使用方法 | 训练和部署 |
| [可视化更新总结.md](可视化更新总结.md) | 本次更新的详细说明 | 了解更新内容 |
| [更新说明.md](更新说明.md) | 整体功能更新 | 了解所有更新 |

---

## 🚀 快速开始

### 1️⃣ 训练新模型（带可视化）

```bash
cd AirsimAlgorithmPython/multirotor/DQN_Weight
python train_with_airsim_improved.py
```

**看到什么**：
- ✅ 训练统计窗口（Episode、奖励、步数）
- ✅ 实时奖励曲线图
- ✅ 当前权重可视化
- ✅ 环境状态监控

### 2️⃣ 使用训练好的模型（带可视化）

```bash
cd AirsimAlgorithmPython/multirotor
python AlgorithmServer.py --use-learned-weights
```

**看到什么**：
- ✅ 环境和无人机状态
- ✅ DQN预测的权重
- ✅ 系统运行信息
- ✅ 重置按钮

---

## 📁 文件说明

### 核心代码文件

| 文件 | 功能 | 用途 |
|------|------|------|
| `training_visualizer.py` | 训练专用可视化 | **训练时**显示统计信息 |
| `../Algorithm/simple_visualizer.py` | 常规可视化 | **运行时**显示环境状态 |
| `train_with_airsim_improved.py` | 训练脚本 | DQN模型训练 |
| `../AlgorithmServer.py` | 算法服务器 | 连接Unity和AirSim |

### 文档文件

| 文件 | 主要内容 |
|------|---------|
| `训练可视化说明.md` | TrainingVisualizer使用指南、窗口布局、故障排除 |
| `模型使用指南.md` | 训练模型、使用模型、选择模型、命令示例 |
| `可视化更新总结.md` | 更新内容、两种可视化对比、使用流程 |
| `更新说明.md` | 修复的问题、新增功能、使用方法更新 |
| `问题修复总结.md` | 问题诊断、修复效果、完整流程 |

### 测试工具

| 文件 | 功能 |
|------|------|
| `test_model_loading.py` | 测试模型加载功能 |
| `测试模型加载.bat` | 一键测试脚本（Windows） |

---

## 🎯 两种可视化的区别

### TrainingVisualizer（训练可视化）🆕

**何时使用**：训练DQN模型时

**启动方式**：
```bash
python train_with_airsim_improved.py
# 自动启动（如果ENABLE_VISUALIZATION = True）
```

**显示内容**：
```
┌─────────────────────────────────────────┐
│  [环境状态]  [环境视图]  [训练统计]     │
│                                         │
│  网格统计     网格+无人机   Episode统计 │
│  扫描进度     Leader位置    奖励统计    │
│                                         │
│  [当前权重]              [奖励曲线]    │
│  5个权重值              趋势图        │
└─────────────────────────────────────────┘
```

**特点**：
- ✅ 实时训练统计
- ✅ 奖励曲线图
- ✅ 权重可视化
- ✅ Episode进度

### SimpleVisualizer（常规可视化）

**何时使用**：正常运行系统时

**启动方式**：
```bash
python AlgorithmServer.py
# 默认启用，或使用--no-visualization禁用
```

**显示内容**：
```
┌─────────────────────────────────────────┐
│  [系统状态]  [环境视图]  [熵值图例]     │
│                                         │
│  无人机数     网格+无人机   颜色说明    │
│  DQN模式     Leader位置    0-100       │
│  权重显示                              │
│                                         │
│  [重置按钮]              [权重面板]    │
│  重置仿真                5个权重详情   │
└─────────────────────────────────────────┘
```

**特点**：
- ✅ 环境监控
- ✅ DQN权重预测
- ✅ 重置按钮
- ✅ 系统统计

---

## 🔄 完整工作流程

### 阶段1：训练模型 🎓

```bash
# 1. 启动训练（使用TrainingVisualizer）
cd AirsimAlgorithmPython/multirotor/DQN_Weight
python train_with_airsim_improved.py
```

**预期**：
- 窗口标题：`🎯 DQN训练实时可视化`
- 窗口大小：1400x900
- 显示内容：训练统计 + 环境状态

**生成文件**：
- `models/best_model.zip` ⭐ 最佳模型
- `models/weight_predictor_airsim.zip` 最终模型
- `models/checkpoint_*.zip` 检查点

### 阶段2：测试模型 🧪

```bash
# 2. 测试模型加载
python test_model_loading.py
```

**预期**：
- 验证所有模型文件
- 显示可用模型列表
- 测试加载成功/失败

### 阶段3：使用模型 🚀

```bash
# 3. 使用最佳模型运行（使用SimpleVisualizer）
cd ..
python AlgorithmServer.py --use-learned-weights --model-path DQN_Weight/models/best_model
```

**预期**：
- 窗口标题：`无人机环境实时可视化`
- 窗口大小：1200x800
- 显示内容：环境状态 + DQN权重

---

## 💡 常见问题速查

### Q1: 训练时可视化窗口没弹出？

**检查**：
```bash
pip install pygame
```

**日志标志**：
```
✅ 训练可视化已启动  ← 成功
⚠️  训练可视化初始化失败  ← 失败
```

### Q2: 如何选择使用哪个模型？

**方法1**（自动选择）：
```bash
python AlgorithmServer.py --use-learned-weights
# 优先级：best_model > weight_predictor_airsim > weight_predictor_simple
```

**方法2**（指定模型）：
```bash
python AlgorithmServer.py --use-learned-weights --model-path DQN_Weight/models/best_model
```

### Q3: 两种可视化有什么区别？

| 场景 | 可视化类 | 窗口标题 | 关键内容 |
|------|---------|---------|---------|
| 训练 | TrainingVisualizer | DQN训练实时可视化 | Episode、奖励曲线 |
| 运行 | SimpleVisualizer | 无人机环境实时可视化 | 系统状态、重置按钮 |

### Q4: 如何禁用可视化？

**训练时**：
```python
# train_with_airsim_improved.py
ENABLE_VISUALIZATION = False
```

**运行时**：
```bash
python AlgorithmServer.py --no-visualization
```

---

## 📚 学习路径

### 新手入门
1. ✅ 阅读本文件（5分钟）
2. ✅ 运行 `python train_with_airsim_improved.py`（观察训练可视化）
3. ✅ 阅读 [训练可视化说明.md](训练可视化说明.md)（了解详细功能）

### 进阶使用
1. ✅ 阅读 [模型使用指南.md](模型使用指南.md)（学习模型选择）
2. ✅ 运行 `python test_model_loading.py`（验证模型）
3. ✅ 阅读 [可视化更新总结.md](可视化更新总结.md)（理解架构）

### 问题排查
1. ✅ 阅读 [训练可视化说明.md § 故障排除](训练可视化说明.md)
2. ✅ 查看日志中的 ✅/❌ 标志
3. ✅ 阅读 [问题修复总结.md](../问题修复总结.md)

---

## 🎯 最佳实践

### 训练阶段
```bash
# 1. 首次训练 - 启用可视化
ENABLE_VISUALIZATION = True
TOTAL_TIMESTEPS = 5000

# 2. 观察前几个Episode
# 验证训练正常、奖励合理、环境重置正常

# 3. 长时间训练 - 可考虑禁用可视化
ENABLE_VISUALIZATION = False
TOTAL_TIMESTEPS = 50000
```

### 使用阶段
```bash
# 1. 测试最佳模型
python AlgorithmServer.py --use-learned-weights

# 2. 对比不同模型
python AlgorithmServer.py --use-learned-weights --model-path DQN_Weight/models/checkpoint_5000

# 3. 多无人机场景
python AlgorithmServer.py --use-learned-weights --drones 3
```

---

## 🆘 获取帮助

### 查看命令帮助
```bash
python AlgorithmServer.py --help
```

### 查看详细文档
- [训练可视化说明.md](训练可视化说明.md) - TrainingVisualizer完整指南
- [模型使用指南.md](模型使用指南.md) - 训练和使用方法
- [可视化更新总结.md](可视化更新总结.md) - 技术细节和架构

### 测试工具
```bash
python test_model_loading.py  # 测试模型加载
测试模型加载.bat              # Windows一键测试
```

---

## 🎉 总结

现在你有**两个强大的可视化工具**：

| 工具 | 用途 | 优势 |
|------|------|------|
| **TrainingVisualizer** | 训练监控 | 实时统计、趋势分析 |
| **SimpleVisualizer** | 运行监控 | 环境状态、系统控制 |

**开始使用**：
1. 训练模型：`python train_with_airsim_improved.py`
2. 使用模型：`python AlgorithmServer.py --use-learned-weights`
3. 享受可视化带来的便利！

🚀 **Happy Training!** 🎨

