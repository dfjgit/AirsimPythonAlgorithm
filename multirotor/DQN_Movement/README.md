# DQN Movement 目录结构说明

本目录包含DQN移动控制和分层强化学习（HRL）的所有相关文件，已经过重构优化，按功能分类组织。

## 📁 目录结构

```
DQN_Movement/
├── configs/              # 配置文件
│   ├── movement_dqn_config.json          # DQN移动控制配置
│   └── hierarchical_dqn_config.json      # 分层DQN配置
│
├── envs/                 # 环境文件
│   ├── movement_env.py                   # DQN移动环境
│   └── hierarchical_movement_env.py      # 分层移动环境
│
├── scripts/              # 训练脚本
│   ├── train_movement_dqn.py             # DQN移动训练（基础）
│   ├── train_movement_with_airsim.py     # DQN移动训练（AirSim集成）
│   ├── train_hierarchical_dqn.py         # 分层DQN训练（基础）
│   └── train_hierarchical_with_airsim.py # 分层DQN训练（AirSim集成）
│
├── tests/                # 测试文件
│   ├── test_movement_dqn.py              # 测试DQN移动模型
│   ├── test_hierarchical_dqn.py          # 测试分层DQN
│   ├── test_hierarchical_visualization.py # 测试分层可视化
│   ├── test_multi_drone_env.py           # 测试多无人机环境
│   ├── test_callback.py                  # 测试回调功能
│   └── diagnose_dqn_env.py               # 环境诊断工具
│
├── visualizers/          # 可视化工具
│   └── hierarchical_visualizer.py        # 分层训练可视化器
│
├── docs/                 # 文档
│   └── HIERARCHICAL_VISUALIZATION.md     # 分层可视化使用说明
│
├── logs/                 # 训练日志（自动生成）
│   ├── movement_dqn_airsim/              # DQN训练日志
│   ├── hrl_dqn_airsim/                   # 分层DQN日志
│   ├── hrl_tensorboard/                  # TensorBoard日志
│   └── dqn_scan_data/                    # 扫描数据
│
├── models/               # 训练模型（自动生成）
│   ├── hrl_planner/                      # 分层规划器模型
│   └── movement_dqn_airsim_final.zip     # DQN最终模型
│
└── requirements_movement.txt              # 依赖包列表
```

## 🚀 快速开始

### 1. 基础DQN训练（模拟环境）
```bash
# Windows
scripts\Train_DQN_Movement.bat

# 或直接运行Python脚本
python multirotor\DQN_Movement\scripts\train_movement_dqn.py
```

### 2. DQN训练（AirSim集成）
```bash
# Windows
scripts\Train_DQN_Movement_Real_Environment.bat

# 或直接运行
python multirotor\DQN_Movement\scripts\train_movement_with_airsim.py
```

### 3. 分层DQN训练
```bash
# Windows
scripts\Train_Hierarchical_DQN.bat

# 或直接运行
python multirotor\DQN_Movement\scripts\train_hierarchical_dqn.py
```

### 4. 测试训练好的模型
```bash
# Windows
scripts\Test_DQN_Movement.bat

# 或直接运行
python multirotor\DQN_Movement\tests\test_movement_dqn.py
```

## 📝 配置文件说明

### movement_dqn_config.json
- 移动步长、最大步数
- 奖励函数参数（探索、碰撞、熵优化等）
- DQN模型参数（学习率、缓冲区大小等）

### hierarchical_dqn_config.json
- 高层规划器配置（区域选择、决策频率）
- 底层控制器配置（移动控制、执行频率）
- 奖励函数和模型参数

## 🔧 主要组件

### 环境（envs/）
- **MovementEnv**: 基础DQN移动环境，支持6方向移动控制
- **HierarchicalMovementEnv**: 分层环境，包含高层规划器和底层控制器

### 训练脚本（scripts/）
- **train_movement_dqn.py**: 独立训练DQN移动策略
- **train_movement_with_airsim.py**: 与AirSim集成训练
- **train_hierarchical_dqn.py**: 训练分层规划器
- **train_hierarchical_with_airsim.py**: 分层架构与AirSim集成训练

### 可视化（visualizers/）
- **HierarchicalVisualizer**: 实时显示训练过程、奖励曲线、动作分布等

## 📊 日志和模型

训练过程中会自动生成：
- **logs/**: TensorBoard日志、CSV数据、训练统计
- **models/**: 训练好的模型文件（.zip格式）

查看TensorBoard：
```bash
tensorboard --logdir=multirotor/DQN_Movement/logs/hrl_tensorboard
```

## 🐛 故障排查

### 导入错误
如果遇到模块导入问题，确保：
1. 从项目根目录运行脚本
2. Python路径配置正确
3. 已安装所有依赖：`pip install -r requirements_movement.txt`

### 配置文件未找到
配置文件路径已更新到`configs/`目录，如果出现配置文件未找到的错误：
1. 检查`configs/`目录是否存在
2. 确认配置文件名称正确

### 模型加载失败
确保模型文件在`models/`目录下，且文件路径正确。

## 📚 更多信息

- 分层可视化使用说明：[docs/HIERARCHICAL_VISUALIZATION.md](docs/HIERARCHICAL_VISUALIZATION.md)
- DQN与DDPG介绍：[../DDPG与DQN介绍.md](../DDPG与DQN介绍.md)
- 配置文件架构：[../配置文件架构说明.md](../配置文件架构说明.md)

## 🎯 重构说明

本次重构（2026-02-02）将原本杂乱的文件按功能分类到不同目录：
- ✅ 配置文件 → `configs/`
- ✅ 环境文件 → `envs/`
- ✅ 训练脚本 → `scripts/`
- ✅ 测试文件 → `tests/`
- ✅ 可视化工具 → `visualizers/`
- ✅ 文档 → `docs/`

所有导入路径已更新，批处理脚本已同步修改。
