# 🚁 DQN无人机移动控制模块

## 📋 模块说明

这是**DQN无人机移动控制**模块，使用深度Q网络(DQN)训练无人机进行自主移动和区域扫描。

### 核心特点

- **动作空间**: 6个离散动作（上/下/左/右/前/后）
- **观察空间**: 21维连续状态（位置、速度、熵值等）
- **训练模式**: 纯模拟训练 & AirSim集成训练
- **灵活配置**: JSON配置文件

## 📁 文件列表

### 核心代码
- `movement_env.py` - 环境类（动作/观察空间、奖励函数）⭐核心
  - 21维观察空间：位置、速度、朝向、熵值、Leader信息等
  - 6个离散动作：上/下/左/右/前/后
  - 使用 gymnasium（不是旧版 gym）
- `train_movement_dqn.py` - 纯模拟训练脚本
- `train_movement_with_airsim.py` - AirSim集成训练脚本
- `test_movement_dqn.py` - 模型测试脚本

### 配置文件
- `movement_dqn_config.json` - 训练配置（奖励、阈值、参数）
  - 详细的奖励函数配置
  - DQN训练超参数
  - 可自定义阈值设置
- `requirements_movement.txt` - Python依赖列表

### 文档
- `README_MOVEMENT.md` - 快速开始指南 ⭐推荐先看
- `MOVEMENT_DQN.md` - 完整详细文档
- `INSTALL_GUIDE.md` - 安装和问题排查
- `INDEX.md` - 文件索引

### 模型目录
- `models/` - 训练好的模型保存目录
  - `movement_dqn_final.zip` - 最终模型
  - `movement_dqn_checkpoint_*.zip` - 训练检查点

## 🚀 快速开始

### 1. 安装依赖

**方式一：使用requirements文件**
```bash
pip install -r requirements_movement.txt
```

**方式二：手动安装**
```bash
pip install torch stable-baselines3 numpy gymnasium
```

> **注意**: 使用 `gymnasium` 而不是旧版的 `gym`

### 2. 开始训练（纯模拟模式）

**Windows**:
```bash
# 返回项目根目录运行
cd ..\..
train_movement_dqn.bat
```

**直接运行Python**:
```bash
python multirotor/DQN_Movement/train_movement_dqn.py
```

训练时间：约10-30分钟（10万步，取决于硬件）

### 3. 测试模型

**Windows**:
```bash
cd ..\..
test_movement_dqn.bat
```

**Python**:
```bash
python multirotor/DQN_Movement/test_movement_dqn.py
```

### 4. 监控训练（可选）

```bash
tensorboard --logdir=multirotor/DQN_Movement/logs/movement_dqn/
# 然后访问 http://localhost:6006
```

## 📚 详细文档

- **快速上手**: 查看 `README_MOVEMENT.md`
- **完整说明**: 查看 `MOVEMENT_DQN.md`
- **安装问题**: 查看 `INSTALL_GUIDE.md`

## 🎯 动作和观察空间

### 动作空间 (6个离散动作)

使用 `gymnasium.spaces.Discrete(6)`

| 动作ID | 方向 | 位移向量 (米) |
|--------|------|--------------|
| 0 | 向上 | (0, 0, +step_size) |
| 1 | 向下 | (0, 0, -step_size) |
| 2 | 向左 | (-step_size, 0, 0) |
| 3 | 向右 | (+step_size, 0, 0) |
| 4 | 向前 | (0, +step_size, 0) |
| 5 | 向后 | (0, -step_size, 0) |

默认 `step_size = 1.0` 米，可在配置文件中调整。

### 观察空间 (21维)

使用 `gymnasium.spaces.Box(shape=(21,))`

| 维度 | 名称 | 说明 |
|------|------|------|
| 0-2 | 位置 | x, y, z 坐标 |
| 3-5 | 速度 | vx, vy, vz |
| 6-8 | 朝向 | forward 向量 |
| 9-11 | 局部熵值 | 平均熵、最大熵、熵标准差 |
| 12-14 | Leader相对位置 | dx, dy, dz |
| 15-16 | Leader范围信息 | 距离、是否越界标志 |
| 17-19 | 扫描进度 | 已扫描比例、已扫描数、未扫描数 |
| 20 | 最近无人机距离 | 与其他无人机的最小距离 |

## 💡 与权重APF模块的区别

| 特性 | 移动控制(本模块) | 权重APF |
|-----|----------------|---------|
| 动作空间 | 6个离散移动方向 | 5个连续权重系数 |
| 算法 | DQN | DDPG |
| 控制方式 | 直接控制移动 | 调整APF权重 |
| 适用场景 | 简单直接的移动策略 | 复杂场景的行为调整 |

## 📊 预期性能

训练成功后的性能指标：

| 指标 | 目标值 | 说明 |
|------|--------|------|
| **平均奖励** | > 100 | episode平均累积奖励 |
| **扫描完成率** | > 90% | 区域扫描覆盖率 |
| **碰撞次数** | < 5次/episode | 与其他无人机的碰撞 |
| **训练时间** | 10-30分钟 | 10万步，取决于CPU/GPU |

**训练参数**（默认配置）：
- 总训练步数：100,000
- 学习率：0.0001
- 批次大小：32
- 缓冲区大小：50,000
- 探索率：1.0 → 0.05（30%时间内衰减）

## 🔗 相关链接

- **权重APF模块**: `../DQN_Weight/` - 使用DDPG学习APF权重系数
- **算法服务器**: `../AlgorithmServer.py` - 主控制服务器
- **项目主页**: `../../README.MD` - 项目总体说明
- **Stable-Baselines3文档**: https://stable-baselines3.readthedocs.io/
- **Gymnasium文档**: https://gymnasium.farama.org/

## 🛠️ 技术栈

- **Python**: 3.7+
- **强化学习框架**: Stable-Baselines3
- **环境接口**: Gymnasium
- **深度学习框架**: PyTorch
- **算法**: DQN (Deep Q-Network)

## 🔍 代码结构说明

```python
# movement_env.py 核心类
class MovementEnv(gym.Env):
    action_space = spaces.Discrete(6)        # 6个离散动作
    observation_space = spaces.Box(          # 21维连续状态
        shape=(21,), dtype=np.float32
    )
    
    def step(action):    # 执行动作，返回奖励
    def reset():         # 重置环境
    def _calculate_reward():  # 计算奖励函数
```

---

**版本**: v1.0.1  
**最后更新**: 2025-10-16  
**状态**: ✅ 稳定可用

