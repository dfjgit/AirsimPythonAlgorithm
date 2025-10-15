# 📚 DQN模块总览

## 🗂️ 文件夹结构

DQN相关代码现已重新组织为两个独立模块：

```
multirotor/
├── DQN_Movement/          # 🚁 DQN无人机移动控制
│   ├── movement_env.py
│   ├── train_movement_dqn.py
│   ├── test_movement_dqn.py
│   ├── movement_dqn_config.json
│   └── 文档...
│
├── DQN_Weight/            # ⚖️ DQN权重APF学习
│   ├── simple_weight_env.py
│   ├── train_simple.py
│   ├── test_trained_model.py
│   ├── dqn_reward_config.json
│   └── 文档...
│
└── DQN_README.md          # 📘 本文件
```

## 📁 模块说明

### 1. DQN_Movement - 无人机移动控制 🚁

**功能**: 使用DQN学习6方向移动策略

**特点**:
- **动作空间**: 6个离散动作（上/下/左/右/前/后）
- **算法**: DQN
- **控制**: 直接控制无人机移动
- **适用**: 简单直接的移动任务

**快速开始**:
```bash
# 返回到AirsimAlgorithmPython目录
cd ..\..
train_movement_dqn.bat
```

**详细文档**: `DQN_Movement/README_MOVEMENT.md`

---

### 2. DQN_Weight - APF权重学习 ⚖️

**功能**: 使用DDPG学习APF算法的5个权重系数

**特点**:
- **动作空间**: 5个连续权重（α1-α5）
- **算法**: DDPG
- **控制**: 调整APF权重系数
- **适用**: 复杂场景的行为优化

**快速开始**:
```bash
cd multirotor/DQN_Weight
python train_simple.py
```

**详细文档**: `DQN_Weight/README.md`

---

## 🆚 两个模块的对比

| 特性 | DQN_Movement | DQN_Weight |
|-----|-------------|-----------|
| **算法** | DQN | DDPG |
| **动作类型** | 离散 | 连续 |
| **动作空间** | 6个方向 | 5个权重 |
| **观察空间** | 21维 | 18维 |
| **控制对象** | 无人机位置 | APF权重 |
| **训练难度** | 较简单 | 较复杂 |
| **效果** | 直接移动 | 间接行为调整 |
| **适用场景** | 简单任务 | 复杂环境 |

## 🎯 如何选择？

### 选择 DQN_Movement（移动控制）如果：
- ✅ 你想直接学习移动策略
- ✅ 场景相对简单
- ✅ 需要快速原型验证
- ✅ 第一次使用强化学习

### 选择 DQN_Weight（权重APF）如果：
- ✅ 已有APF算法框架
- ✅ 需要调整复杂行为
- ✅ 场景需要精细控制
- ✅ 追求更优的性能

## 📚 快速导航

### 🚁 DQN_Movement（移动控制）

| 文档 | 路径 | 说明 |
|-----|------|------|
| 快速开始 | `DQN_Movement/README_MOVEMENT.md` | 5分钟入门 |
| 完整文档 | `DQN_Movement/MOVEMENT_DQN.md` | 详细说明 |
| 安装指南 | `DQN_Movement/INSTALL_GUIDE.md` | 依赖安装 |
| 文件索引 | `DQN_Movement/INDEX.md` | 快速查找 |

### ⚖️ DQN_Weight（权重APF）

| 文档 | 路径 | 说明 |
|-----|------|------|
| 模块说明 | `DQN_Weight/README.md` | 功能概述 |
| 权重平衡 | `DQN_Weight/WEIGHT_BALANCING.md` | 权重调整策略 |
| 配置指南 | `DQN_Weight/DQN_CONFIG_GUIDE.md` | 参数配置 |
| 更新日志 | `DQN_Weight/CHANGELOG_REWARD_CONFIG.md` | 版本记录 |

## 🚀 启动脚本

所有启动脚本位于 `AirsimAlgorithmPython/` 目录：

### DQN_Movement 启动脚本
```bash
train_movement_dqn.bat              # 纯模拟训练
test_movement_dqn.bat               # 模型测试
train_movement_with_airsim.bat     # AirSim集成
```

### DQN_Weight 启动脚本
需要进入对应目录运行：
```bash
cd multirotor\DQN_Weight
python train_simple.py              # 纯模拟训练
python train_with_airsim.py        # AirSim集成
python test_trained_model.py       # 模型测试
```

## 💡 使用建议

### 学习路径

**入门阶段**:
1. 从 `DQN_Movement` 开始
2. 阅读 `README_MOVEMENT.md`
3. 运行纯模拟训练
4. 理解基本概念

**进阶阶段**:
1. 学习 `DQN_Weight`
2. 理解APF权重系统
3. 对比两种方法
4. 选择合适的方案

### 实际应用

**简单场景**:
- 使用 DQN_Movement 直接控制
- 快速部署和验证

**复杂场景**:
- 使用 DQN_Weight 调整权重
- 结合现有APF算法
- 获得更好的性能

## 🔧 依赖安装

### DQN_Movement
```bash
cd DQN_Movement
pip install -r requirements_movement.txt
```

### DQN_Weight
使用现有项目依赖：
```bash
cd ../..
pip install -r requirements.txt
```

## 📊 性能对比

### 训练时间

| 模块 | 纯模拟 | AirSim集成 |
|-----|--------|-----------|
| DQN_Movement | 10-30分钟 | 30-60分钟 |
| DQN_Weight | 15-40分钟 | 40-90分钟 |

*基于10万训练步数，实际时间取决于硬件*

### 效果表现

| 指标 | DQN_Movement | DQN_Weight |
|-----|-------------|-----------|
| 收敛速度 | ⚡⚡⚡ 快 | ⚡⚡ 中等 |
| 最终性能 | ⭐⭐⭐ 良好 | ⭐⭐⭐⭐ 优秀 |
| 稳定性 | 👍 稳定 | 👍👍 很稳定 |
| 泛化能力 | 🎯 一般 | 🎯🎯 较好 |

## ❓ 常见问题

### Q: 两个模块可以同时使用吗？
A: 可以，但需要选择其中一个作为主要控制方式。

### Q: 哪个模块效果更好？
A: 取决于场景。简单任务用Movement，复杂任务用Weight。

### Q: 可以结合使用吗？
A: 理论上可以，Movement学习粗粒度移动，Weight进行细粒度调整。

### Q: 如何迁移旧代码？
A: 原DQN文件夹的文件已分别移动到两个新文件夹，路径需要更新。

## 🔗 相关资源

- **算法服务器**: `../AlgorithmServer.py`
- **APF算法**: `../Algorithm/scanner_algorithm.py`
- **配置文件**: `../scanner_config.json`
- **项目文档**: `../../README.MD`

## 📝 迁移说明

### 从旧DQN文件夹迁移

**旧路径** → **新路径**

移动控制相关:
```
DQN/movement_env.py → DQN_Movement/movement_env.py
DQN/train_movement_dqn.py → DQN_Movement/train_movement_dqn.py
```

权重APF相关:
```
DQN/simple_weight_env.py → DQN_Weight/simple_weight_env.py
DQN/train_simple.py → DQN_Weight/train_simple.py
```

### 更新导入路径

如果你的代码中有导入DQN模块，需要更新：

```python
# 旧的导入
from multirotor.DQN.movement_env import MovementEnv

# 新的导入
from multirotor.DQN_Movement.movement_env import MovementEnv
```

## 📅 版本历史

### v1.1.0 (2024-10-14)
- ✨ 重组文件夹结构
- ✨ 分离移动控制和权重APF模块
- 📝 更新所有文档
- 🔧 更新批处理脚本路径

### v1.0.0
- 🎉 初始版本

---

**维护者**: AirsimProject Team  
**最后更新**: 2024-10-14  
**状态**: ✅ 活跃维护

