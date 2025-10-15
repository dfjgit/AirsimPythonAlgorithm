# DQN无人机移动控制模块

## 📋 概述

这是一个使用深度Q网络(DQN)训练无人机进行自主移动和区域扫描的模块。与传统的权重学习不同，这个模块直接学习无人机的移动策略，通过6个方向的离散动作控制无人机。

## 🎯 功能特点

- **离散动作空间**: 6个方向移动（上/下/左/右/前/后）
- **丰富的观察空间**: 包含位置、速度、熵值、Leader信息等21维状态
- **灵活的奖励机制**: 探索奖励、碰撞惩罚、越界惩罚等多种奖励设计
- **支持纯模拟训练**: 可在无AirSim环境下快速测试和验证
- **支持AirSim集成**: 可连接真实Unity环境进行训练

## 📁 文件结构

```
DQN/
├── movement_env.py                    # 无人机移动环境类
├── movement_dqn_config.json          # 配置文件（奖励、阈值、训练参数）
├── train_movement_dqn.py             # 纯模拟训练脚本
├── train_movement_with_airsim.py    # AirSim集成训练脚本
├── test_movement_dqn.py              # 模型测试脚本
├── MOVEMENT_DQN.md                   # 本文档
└── models/                           # 训练模型保存目录
    ├── movement_dqn_final.zip
    └── movement_dqn_checkpoint_*.zip
```

## 🚀 快速开始

### 1. 环境准备

确保已安装必要的依赖：

```bash
pip install torch stable-baselines3 numpy gym
```

### 2. 纯模拟训练（推荐入门）

直接在本地训练，无需启动Unity：

```bash
# Windows
train_movement_dqn.bat

# 或者直接运行Python
python multirotor/DQN/train_movement_dqn.py
```

**优点**:
- ⚡ 快速启动，无需Unity环境
- 🔄 适合快速迭代和参数调试
- 💻 资源占用小

**缺点**:
- 🤖 使用模拟数据，与真实环境有差异

### 3. AirSim集成训练（真实环境）

连接Unity AirSim环境进行训练：

```bash
# 1. 先启动Unity客户端
# 2. 运行训练脚本
train_movement_with_airsim.bat

# 或者
python multirotor/DQN/train_movement_with_airsim.py
```

**优点**:
- 🎮 真实环境反馈
- 🎯 更准确的模型性能
- 🔗 直接应用到实际任务

**缺点**:
- ⏱️ 训练速度较慢
- 🖥️ 需要Unity环境支持

### 4. 测试模型

训练完成后测试模型性能：

```bash
# Windows
test_movement_dqn.bat

# 或者
python multirotor/DQN/test_movement_dqn.py
```

## ⚙️ 配置说明

配置文件 `movement_dqn_config.json` 包含以下主要部分：

### 移动参数 (movement)

```json
{
  "step_size": 1.0,        // 每步移动距离（米）
  "max_steps": 500         // 每个episode最大步数
}
```

### 奖励设计 (rewards)

```json
{
  "exploration": 10.0,          // 发现新区域奖励
  "entropy_reduction": 5.0,     // 降低熵值奖励
  "collision": -50.0,           // 碰撞惩罚
  "out_of_range": -30.0,        // 越界惩罚
  "smooth_movement": 1.0,       // 平滑移动奖励
  "step_penalty": -0.1,         // 每步小惩罚
  "success": 100.0              // 完成任务大奖励
}
```

### 阈值设置 (thresholds)

```json
{
  "collision_distance": 2.0,          // 碰撞判定距离（米）
  "scanned_entropy": 30.0,            // 已扫描判定阈值
  "nearby_entropy_distance": 10.0,    // 局部熵值统计范围（米）
  "success_scan_ratio": 0.95          // 任务成功比例
}
```

### 训练参数 (training)

```json
{
  "total_timesteps": 100000,           // 训练总步数
  "learning_rate": 0.0001,            // 学习率
  "buffer_size": 50000,               // 经验回放缓冲区
  "batch_size": 32,                   // 批次大小
  "gamma": 0.99,                      // 折扣因子
  "exploration_fraction": 0.3,        // 探索衰减比例
  "exploration_initial_eps": 1.0,     // 初始探索率
  "exploration_final_eps": 0.05       // 最终探索率
}
```

## 📊 状态空间详解

环境观察空间为21维向量：

| 维度范围 | 名称 | 说明 |
|---------|------|------|
| 0-2 | 位置 | x, y, z坐标 |
| 3-5 | 速度 | vx, vy, vz速度分量 |
| 6-8 | 朝向 | forward向量 |
| 9-11 | 局部熵值 | 平均熵、最大熵、熵标准差 |
| 12-14 | Leader相对位置 | dx, dy, dz |
| 15-16 | Leader范围信息 | 距离、是否越界 |
| 17-19 | 扫描进度 | 已扫描比例、数量、剩余 |
| 20 | 最近无人机距离 | 避障用 |

## 🎮 动作空间详解

6个离散动作：

| 动作ID | 方向 | 位移向量 |
|--------|------|---------|
| 0 | 上 | (0, 0, +step_size) |
| 1 | 下 | (0, 0, -step_size) |
| 2 | 左 | (-step_size, 0, 0) |
| 3 | 右 | (+step_size, 0, 0) |
| 4 | 前 | (0, +step_size, 0) |
| 5 | 后 | (0, -step_size, 0) |

## 📈 训练监控

### 使用Tensorboard查看训练曲线

```bash
# 纯模拟训练日志
tensorboard --logdir=multirotor/DQN/logs/movement_dqn/

# AirSim集成训练日志
tensorboard --logdir=multirotor/DQN/logs/movement_dqn_airsim/
```

### 关键指标

- **ep_rew_mean**: 平均episode奖励（越高越好）
- **ep_len_mean**: 平均episode长度（太长说明效率低）
- **exploration_rate**: 探索率（逐渐衰减到0.05）
- **loss**: 训练损失（应该逐渐下降）

## 🔧 常见问题

### Q1: 训练很慢怎么办？

**A**: 
- 使用GPU: 确保安装了PyTorch GPU版本
- 减少训练步数: 修改 `total_timesteps`
- 增大批次: 修改 `batch_size` (需要更多内存)
- 使用纯模拟模式: 不连接Unity

### Q2: 模型性能不好？

**A**:
- 调整奖励权重: 增加探索奖励，减少惩罚
- 增加训练时间: 提高 `total_timesteps`
- 调整探索率: 延长探索时间 `exploration_fraction`
- 检查数据: 确保环境状态正确

### Q3: 如何继续训练已有模型？

**A**:
```python
# 在 train_movement_with_airsim.py 中会自动检测
# 也可以手动加载
model = DQN.load("models/movement_dqn_final.zip", env=env)
model.learn(total_timesteps=50000)  # 继续训练
```

### Q4: 如何调整移动步长？

**A**:
修改 `movement_dqn_config.json` 中的 `step_size`:
```json
{
  "movement": {
    "step_size": 2.0  // 改为2米/步
  }
}
```

## 🎓 高级用法

### 自定义奖励函数

编辑 `movement_env.py` 中的 `_calculate_reward` 方法：

```python
def _calculate_reward(self, action, prev_state, next_state):
    reward = 0.0
    
    # 添加自定义奖励
    # 例如：鼓励向高熵区域移动
    entropy_diff = next_state[9] - prev_state[9]  # 第9维是平均熵
    if entropy_diff > 0:
        reward += 2.0  # 移向高熵区域
    
    # ... 其他奖励计算
    return reward
```

### 使用不同的DQN变体

Stable-Baselines3支持多种DQN变体：

```python
from stable_baselines3 import DQN  # 标准DQN
# from stable_baselines3 import DoubleDQN  # 双Q网络（需要更新版本）

# 或使用其他算法
from stable_baselines3 import A2C, PPO

model = PPO("MlpPolicy", env, ...)  # 使用PPO代替DQN
```

### 并行训练

```python
from stable_baselines3.common.vec_env import SubprocVecEnv

# 创建多个并行环境
def make_env():
    return MovementEnv(server=None, drone_name="UAV1")

env = SubprocVecEnv([make_env for _ in range(4)])  # 4个并行环境
model = DQN("MlpPolicy", env, ...)
```

## 📝 与现有系统集成

### 在AlgorithmServer中使用

在 `AlgorithmServer.py` 中集成训练好的模型：

```python
from stable_baselines3 import DQN

class MultiDroneAlgorithmServer:
    def __init__(self, use_dqn_movement=False):
        # ...
        if use_dqn_movement:
            self.dqn_model = DQN.load("path/to/movement_dqn_final.zip")
    
    def _control_loop(self, drone_name):
        # 使用DQN预测动作
        obs = self._get_observation(drone_name)
        action = self.dqn_model.predict(obs, deterministic=True)[0]
        self._apply_action(drone_name, action)
```

## 📚 参考资料

- [Stable-Baselines3 文档](https://stable-baselines3.readthedocs.io/)
- [DQN原论文](https://arxiv.org/abs/1312.5602)
- [OpenAI Gym 文档](https://www.gymlibrary.dev/)

## 🤝 贡献

如果您有改进建议或发现问题，欢迎提出Issue或Pull Request。

## 📄 更新日志

### v1.0.0 (2024-10-14)
- ✨ 初始版本发布
- 🎮 支持6方向离散动作空间
- 📊 21维状态观察空间
- 🔧 完整的配置系统
- 📈 训练和测试脚本
- 📝 完整文档

---

**作者**: AirsimProject Team  
**最后更新**: 2024-10-14

