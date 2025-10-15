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
- `train_movement_dqn.py` - 纯模拟训练脚本
- `train_movement_with_airsim.py` - AirSim集成训练脚本
- `test_movement_dqn.py` - 模型测试脚本

### 配置文件
- `movement_dqn_config.json` - 训练配置（奖励、阈值、参数）
- `requirements_movement.txt` - Python依赖列表

### 文档
- `README_MOVEMENT.md` - 快速开始指南 ⭐推荐先看
- `MOVEMENT_DQN.md` - 完整详细文档
- `INSTALL_GUIDE.md` - 安装和问题排查
- `INDEX.md` - 文件索引

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements_movement.txt
```

### 2. 开始训练
```bash
# Windows - 返回上层目录运行
cd ..\..
train_movement_dqn.bat

# 或直接运行Python
python multirotor\DQN_Movement\train_movement_dqn.py
```

### 3. 测试模型
```bash
# Windows
cd ..\..
test_movement_dqn.bat

# Python
python multirotor\DQN_Movement\test_movement_dqn.py
```

## 📚 详细文档

- **快速上手**: 查看 `README_MOVEMENT.md`
- **完整说明**: 查看 `MOVEMENT_DQN.md`
- **安装问题**: 查看 `INSTALL_GUIDE.md`

## 🎯 动作和观察空间

### 动作空间 (6个离散动作)
```
0: 向上 (+Z)
1: 向下 (-Z)
2: 向左 (-X)
3: 向右 (+X)
4: 向前 (+Y)
5: 向后 (-Y)
```

### 观察空间 (21维)
- 位置(3) + 速度(3) + 朝向(3)
- 局部熵值(3) + Leader信息(5)
- 扫描进度(3) + 最近距离(1)

## 💡 与权重APF模块的区别

| 特性 | 移动控制(本模块) | 权重APF |
|-----|----------------|---------|
| 动作空间 | 6个离散移动方向 | 5个连续权重系数 |
| 算法 | DQN | DDPG |
| 控制方式 | 直接控制移动 | 调整APF权重 |
| 适用场景 | 简单直接的移动策略 | 复杂场景的行为调整 |

## 📊 预期性能

- **平均奖励**: > 100
- **扫描完成率**: > 90%
- **训练时间**: 10-30分钟（10万步）

## 🔗 相关链接

- **权重APF模块**: `../DQN_Weight/`
- **算法服务器**: `../AlgorithmServer.py`
- **项目主页**: `../../README.MD`

---

**版本**: v1.0.0  
**最后更新**: 2024-10-14

