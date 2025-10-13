# DQN权重学习模块

使用DDPG学习APF算法的最优权重系数

---

## 🚀 快速开始

### 1. 训练模型
```bash
cd multirotor/DQN
python train_simple.py
```

### 2. 测试模型
```bash
python test_trained_model.py
python test_integration.py
```

### 3. 使用学习的权重
```bash
cd ..
python AlgorithmServer.py --use-learned-weights
```

---

## 📁 文件说明

### 核心代码
- `simple_weight_env.py` - Gym环境（18维状态 → 5维权重）
- `train_simple.py` - DDPG训练脚本
- `test_trained_model.py` - 模型测试
- `test_integration.py` - 集成测试

### 模型文件
- `models/weight_predictor_simple.zip` - 训练好的模型

---

## 📚 详细文档

所有详细文档已移至 `docs/DQN/` 目录：

- [快速开始](../../docs/DQN/QUICK_START.md)
- [V2设计](../../docs/DQN/V2_DESIGN.md)
- [实现指南](../../docs/DQN/IMPLEMENTATION_GUIDE.md)
- [V2需求](../../docs/DQN/V2_REQUIREMENTS.md)
- [V1归档](../../docs/DQN/README_V1_ARCHIVED.md)
- [CPU优化](../../docs/DQN/CPU_OPTIMIZATION.md)

---

## 🎯 核心思想

**学习权重而非动作**：
```
状态(18维) → DDPG → 权重(5维) → APF → 动作
```

**5个权重**：
- α1: repulsionCoefficient (排斥力)
- α2: entropyCoefficient (熵)
- α3: distanceCoefficient (距离)
- α4: leaderRangeCoefficient (Leader范围)
- α5: directionRetentionCoefficient (方向保持)

---

**更新日期**: 2025-10-13

