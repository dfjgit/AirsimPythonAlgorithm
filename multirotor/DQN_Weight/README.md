# ⚖️ DQN权重APF学习模块

## 📋 模块说明

这是**DQN权重APF学习**模块，使用深度确定性策略梯度(DDPG)训练APF(人工势场)算法的权重系数。

### 核心特点

- **动作空间**: 5个连续权重系数（α1-α5）
- **观察空间**: 18维状态（位置、速度、熵值等）
- **算法**: DDPG（连续动作空间）
- **目标**: 学习最优的APF权重组合

## 📁 文件列表

### 核心代码
- `simple_weight_env.py` - 权重学习环境类 ⭐核心
- `train_simple.py` - 纯模拟训练脚本
- `train_with_airsim.py` - AirSim集成训练脚本
- `test_trained_model.py` - 模型测试脚本
- `test_integration.py` - 集成测试脚本

### 配置文件
- `dqn_reward_config.json` - 奖励配置
- `dqn_reward_config_data.py` - 配置加载类

### 文档
- `README.md` - 本文件
- `WEIGHT_BALANCING.md` - 权重平衡详细说明
- `DQN_CONFIG_GUIDE.md` - 配置指南
- `CHANGELOG_REWARD_CONFIG.md` - 更新日志

### 模型目录
- `models/` - 训练好的模型文件
  - `weight_predictor_simple.zip`
  - `weight_predictor_airsim.zip`
  - 等

## 🚀 快速开始

### 1. 开始训练（纯模拟）
```bash
python train_simple.py
```

### 2. AirSim集成训练
```bash
# 先启动Unity客户端
python train_with_airsim.py
```

### 3. 测试模型
```bash
python test_trained_model.py
```

## 🎯 权重系数说明

学习5个APF权重系数：

| 权重 | 名称 | 作用 |
|-----|------|------|
| α1 | repulsionCoefficient | 斥力系数（避障） |
| α2 | entropyCoefficient | 熵值系数（探索） |
| α3 | distanceCoefficient | 距离系数（目标吸引） |
| α4 | leaderRangeCoefficient | Leader范围系数 |
| α5 | directionRetentionCoefficient | 方向保持系数 |

## ⚙️ 配置说明

主要配置在 `dqn_reward_config.json`:

```json
{
  "weight_min": 0.1,        // 权重最小值
  "weight_max": 5.0,        // 权重最大值
  "std_threshold": 2.0,     // 标准差阈值
  "exploration_reward": 10.0,  // 探索奖励
  "collision_penalty": -50.0   // 碰撞惩罚
}
```

详细配置说明请查看 `DQN_CONFIG_GUIDE.md`

## 📊 观察空间 (18维)

- 位置(3) + 速度(3) + 方向(3)
- 熵值(3) + Leader(3) + 扫描(3)

## 💡 与移动控制模块的区别

| 特性 | 权重APF(本模块) | 移动控制 |
|-----|----------------|---------|
| 动作空间 | 5个连续权重系数 | 6个离散移动方向 |
| 算法 | DDPG | DQN |
| 控制方式 | 调整APF权重 | 直接控制移动 |
| 适用场景 | 复杂场景的行为调整 | 简单直接的移动策略 |

## 🔧 高级功能

### 权重归一化
自动进行权重归一化和平衡，避免某个权重过高。

### 软归一化
使用标准差阈值进行平滑处理。

### 最大最小比例限制
确保权重在合理范围内。

详见 `WEIGHT_BALANCING.md`

## 📚 详细文档

- **权重平衡**: `WEIGHT_BALANCING.md` - 详细的权重调整策略
- **配置指南**: `DQN_CONFIG_GUIDE.md` - 完整配置说明
- **更新日志**: `CHANGELOG_REWARD_CONFIG.md` - 版本更新记录

## 🎓 使用DDPG的原因

由于APF权重是**连续值**（如0.5, 2.3等），使用DDPG算法更合适：
- DQN: 适合离散动作（如上/下/左/右）
- DDPG: 适合连续动作（如权重0.1-5.0）

## 📈 预期效果

训练成功后，模型应该能：
- ✅ 根据环境动态调整权重
- ✅ 在避障和探索之间平衡
- ✅ 适应不同的扫描场景

## 🔗 相关链接

- **移动控制模块**: `../DQN_Movement/`
- **算法服务器**: `../AlgorithmServer.py`
- **APF算法**: `../Algorithm/scanner_algorithm.py`

## 📝 注意事项

1. 权重范围默认 0.1-5.0，可在配置文件中调整
2. 训练时会自动进行权重归一化
3. 建议先用纯模拟模式训练，再用AirSim验证

---

**版本**: v1.0.0  
**最后更新**: 2024-10-14  
**算法**: DDPG (Deep Deterministic Policy Gradient)
