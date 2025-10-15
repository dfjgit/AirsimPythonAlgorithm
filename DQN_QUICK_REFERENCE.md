# 🎯 DQN模块快速参考

## 📂 新文件夹结构

```
multirotor/
├── 🚁 DQN_Movement/    # 移动控制（6方向，DQN）
├── ⚖️ DQN_Weight/      # 权重APF（5权重，DDPG）
└── 📘 DQN_README.md    # 总览文档
```

---

## ⚡ 快速启动

### DQN_Movement（移动控制）

```bash
# 训练
train_movement_dqn.bat

# 测试
test_movement_dqn.bat

# AirSim
train_movement_with_airsim.bat
```

### DQN_Weight（权重APF）

```bash
cd multirotor\DQN_Weight

# 训练
python train_simple.py

# 测试  
python test_trained_model.py

# AirSim
python train_with_airsim.py
```

---

## 📚 文档速查

| 需求 | 文档路径 |
|-----|---------|
| **快速开始** | `DQN_Movement/README_MOVEMENT.md` |
| **Movement详细** | `DQN_Movement/MOVEMENT_DQN.md` |
| **Weight说明** | `DQN_Weight/README.md` |
| **权重平衡** | `DQN_Weight/WEIGHT_BALANCING.md` |
| **总览对比** | `multirotor/DQN_README.md` |
| **安装指南** | `DQN_Movement/INSTALL_GUIDE.md` |

---

## 🆚 选择指南

| 场景 | 选择 |
|-----|------|
| 简单移动任务 | 🚁 DQN_Movement |
| 复杂行为优化 | ⚖️ DQN_Weight |
| 快速原型 | 🚁 DQN_Movement |
| 精细控制 | ⚖️ DQN_Weight |
| 第一次使用 | 🚁 DQN_Movement |
| 已有APF算法 | ⚖️ DQN_Weight |

---

## 🔑 关键差异

|  | DQN_Movement | DQN_Weight |
|--|-------------|-----------|
| **算法** | DQN | DDPG |
| **动作** | 6个离散方向 | 5个连续权重 |
| **观察** | 21维 | 18维 |
| **控制** | 直接位置 | 调整权重 |

---

## 📦 依赖安装

```bash
# DQN_Movement
cd multirotor\DQN_Movement
pip install -r requirements_movement.txt

# DQN_Weight（使用项目依赖）
cd ..\..
pip install -r requirements.txt
```

---

## 💡 推荐学习路径

```
1. 阅读 DQN_README.md（了解两个模块）
   ↓
2. 选择 DQN_Movement 入门
   ↓
3. 运行 train_movement_dqn.bat
   ↓
4. 理解基本概念后学习 DQN_Weight
   ↓
5. 根据需求选择使用
```

---

**更新**: 2024-10-14  
**状态**: ✅ 可用

