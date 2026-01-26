# AirsimAlgorithmPython - 无人机算法服务器

基于 Python 的多无人机协同控制算法服务器，支持人工势场算法（APF）、DDPG 强化学习权重预测和实时数据采集。

---

## 📋 项目简介

AirsimAlgorithmPython 是无人机仿真系统的算法核心，提供智能控制、路径规划和强化学习训练功能。系统通过 TCP Socket 与 Unity 仿真环境实时通信，实现多无人机协同控制、区域扫描和探索任务。

### 核心特性

- ✅ **人工势场算法（APF）**：多因素权重合成，智能路径规划
- ✅ **强化学习支持**：DDPG 权重预测（DQN 移动控制开发中）
- ✅ **多无人机协同**：支持 1-10 台无人机同时控制
- ✅ **实时通信**：与 Unity 双向数据交互（TCP Socket）
- ✅ **数据采集系统**：自动采集扫描数据、权重值和电量信息
- ✅ **可视化工具**：2D 实时可视化 + 训练可视化（奖励曲线、收敛分析）
- ✅ **统一配置管理**：unified_train_config.json 统一管理所有训练模式
- ✅ **模型覆盖控制**：支持固定名称覆盖或时间戳版本控制

---

## 🎯 系统要求

### 硬件要求
- **操作系统**：Windows 10/11、Linux、macOS
- **内存**：8GB RAM（推荐 16GB，用于 DDPG 训练）
- **显卡**：可选，用于加速 DDPG 训练（CUDA 支持）

### 软件要求
- **Python**：3.7 或更高版本（推荐 3.8+）
- **AirSim**：已安装并运行（用于物理仿真）
- **Unity**：Airsim2022 项目已启动（用于可视化）

---

## 🚀 快速开始

### 1. 安装依赖

```bash
# 克隆或下载项目
cd AirsimAlgorithmPython

# 安装 Python 依赖
pip install -r requirements.txt

# 或使用 setup.py 安装
pip install -e .
```

### 2. 配置参数

编辑 `multirotor/scanner_config.json`：

```json
{
    "repulsionCoefficient": 4.0,      // 排斥力权重
    "entropyCoefficient": 2.0,        // 熵值权重
    "distanceCoefficient": 2.0,       // 距离权重
    "leaderRangeCoefficient": 2.0,    // Leader范围权重
    "directionRetentionCoefficient": 2.0,  // 方向保持权重
    "updateInterval": 0.5,            // 更新间隔（秒）
    "moveSpeed": 1.0,                  // 移动速度（米/秒）
    "scanRadius": 2.0,                 // 扫描半径（米）
    "altitude": 2.0                    // 飞行高度（米）
}
```

### 3. 启动系统

#### 方式一：使用批处理文件（Windows）

```bash
# 运行主菜单
start.bat

# 或直接运行固定权重模式
scripts\运行系统-固定权重.bat
```

#### 方式二：命令行启动

```bash
# 进入 multirotor 目录
cd multirotor

# 使用固定权重（默认）
python AlgorithmServer.py

# 使用 DDPG 权重预测
python AlgorithmServer.py --use-learned-weights

# 多无人机（3台）
python AlgorithmServer.py --drones 3

# 指定模型路径
python AlgorithmServer.py --use-learned-weights --model-path DDPG_Weight/models/best_model

# 禁用可视化
python AlgorithmServer.py --no-visualization
```

### 4. 验证运行

启动后，系统将：
1. 连接到 AirSim 模拟器
2. 等待 Unity 客户端连接（端口 41451）
3. 初始化无人机并起飞
4. 开始执行扫描算法

---

## 🏗️ 项目结构

```
AirsimAlgorithmPython/
├── airsim/                          # AirSim Python 客户端库
│   ├── client.py                    # 客户端核心
│   ├── types.py                     # 数据类型定义
│   └── utils.py                     # 工具函数
│
├── multirotor/                      # 多旋翼控制核心
│   ├── AlgorithmServer.py           # 主服务器入口 ⭐
│   ├── scanner_config.json          # 算法配置文件
│   │
│   ├── Algorithm/                   # 算法实现模块
│   │   ├── scanner_algorithm.py     # APF 算法核心
│   │   ├── scanner_config_data.py   # 配置数据类
│   │   ├── scanner_runtime_data.py  # 运行时数据类
│   │   ├── HexGridDataModel.py      # 网格数据模型
│   │   ├── Vector3.py               # 3D 向量类
│   │   ├── simple_visualizer.py     # 可视化组件
│   │   ├── data_collector.py        # 数据采集模块
│   │   ├── battery_data.py          # 电池数据类
│   │   ├── visualize_scan_csv.py    # CSV数据可视化
│   │   └── visualize_training_data.py # 训练数据可视化 ✨
│   │
│   ├── AirsimServer/                # 服务器组件
│   │   ├── drone_controller.py      # 无人机控制器
│   │   ├── unity_socket_server.py   # Unity 通信服务
│   │   └── data_pack.py             # 数据包定义
│   │
│   ├── Crazyswarm/                  # Crazyflie 实体机支持
│   │   ├── crazyflie_operate.py     # 实体机控制
│   │   ├── crazyflie_wayPoint.py    # 航点控制
│   │   └── crazyswarm.py            # Crazyswarm 集成
│   │
│   ├── DDPG_Weight/                 # DDPG 权重预测模块
│   │   ├── simple_weight_env.py     # 权重环境定义
│   │   ├── crazyflie_weight_env.py  # Crazyflie 权重环境
│   │   ├── train_with_airsim_improved.py  # 训练脚本（仿真）
│   │   ├── train_with_crazyflie_logs.py   # 日志训练
│   │   ├── train_with_crazyflie_online.py # 在线训练
│   │   ├── train_with_hybrid.py     # 虚实融合训练
│   │   ├── training_visualizer.py   # 训练可视化器 ✨
│   │   ├── crazyflie_data_logger.py # Crazyflie 数据记录器 ✨
│   │   ├── test_trained_model.py    # 模型测试
│   │   ├── unified_train_config.json # 统一训练配置 ⭐
│   │   ├── models/                  # 训练好的模型
│   │   ├── crazyflie_logs/          # Crazyflie 训练日志 ✨
│   │   ├── dqn_reward_config.json   # 奖励配置（仿真）
│   │   └── crazyflie_reward_config.json # 奖励配置（实体机）
│   │
│   ├── DQN_Movement/                 # DQN 移动控制模块
│   │   ├── movement_env.py          # 移动环境定义
│   │   ├── train_movement_dqn.py    # 训练脚本
│   │   ├── models/                  # 训练好的模型
│   │   └── movement_dqn_config.json # 配置文件
│   │
│   ├── DDPG与DQN介绍.md              # 强化学习模块说明
│   ├── DDPG_Weight/                 # DDPG 权重预测模块
│   │   ├── airsim_training_logs/    # 虚拟训练数据日志 ✨
│   │   ├── crazyflie_logs/          # Crazyflie 训练日志 ✨
│   │   └── ...
│   └── DQN_Movement/                 # DQN 移动控制模块
│
├── scripts/                          # 批处理脚本
│   ├── 运行系统-固定权重.bat
│   ├── 运行系统-DDPG权重.bat
│   ├── 训练权重DDPG-真实环境.bat
│   ├── 训练权重DDPG-实体机日志.bat
│   ├── 训练权重DDPG-实体机在线.bat
│   ├── 训练权重DDPG-虚实融合.bat
│   ├── 训练移动DQN-真实环境.bat
│   └── 数据可视化分析.bat          # ⭐新增！
│
├── requirements.txt                  # Python 依赖
├── setup.py                          # 安装脚本
└── start.bat                         # 主菜单（Windows）
```

---

## 🔧 核心模块说明

### 1. AlgorithmServer.py（主服务器）

**功能**：系统核心入口，协调所有模块

**主要功能**：
- 连接 AirSim 和 Unity
- 管理多无人机控制
- 执行扫描算法
- 数据采集和可视化

**命令行参数**：
```bash
--use-learned-weights    # 使用 DDPG 权重预测
--model-path PATH        # 指定 DDPG 模型路径
--drones N               # 无人机数量（默认1）
--no-visualization       # 禁用可视化
```

### 2. ScannerAlgorithm（APF 算法）

**功能**：人工势场算法实现

**力向量类型**：
1. **排斥力**（Repulsion）：避免无人机碰撞
2. **熵引力**（Entropy）：探索高熵值区域
3. **距离引力**（Distance）：引导向目标移动
4. **Leader 范围力**（Leader Range）：保持在 Leader 扫描范围内
5. **方向保持力**（Direction Retention）：维持飞行稳定性

**算法流程**：
```
1. 获取当前状态（位置、网格熵值、Leader位置等）
2. 计算各力向量
3. 加权合成最终方向
4. 控制无人机移动
```

### 3. DataCollector（数据采集）

**功能**：自动采集扫描数据、权重值和电量信息

**采集内容**：
- 时间戳和运行时间
- AOI 区域内栅格状态（已侦察/未侦察）
- 扫描比例
- 5 个权重系数值
- 无人机位置信息（x, y, z）
- 电池电压信息（每架无人机）

**输出格式**：CSV 文件
- 位置：`multirotor/DDPG_Weight/airsim_training_logs/scan_data_YYYYMMDD_HHMMSS.csv`
- 频率：每秒一次

**数据可视化**：
- 使用 `visualize_scan_csv.py` 分析扫描数据
- 使用 `visualize_training_data.py` 分析训练数据
- 通过 `start.bat` 选项 [A] 快速启动

### 4. SimpleVisualizer（可视化）

**功能**：2D 实时可视化

**显示内容**：
- 网格熵值（颜色编码）
- 无人机位置和方向
- Leader 位置和扫描范围
- 实时统计信息
- 权重值变化曲线

**启动**：系统自动启动（可通过 `--no-visualization` 禁用）

### 5. UnitySocketServer（通信服务）

**功能**：与 Unity 客户端通信

**通信协议**：
- **协议**：TCP Socket
- **端口**：41451（默认）
- **数据格式**：JSON

**数据包类型**：
- `config_data`：配置数据
- `runtime_data`：运行时数据
- `grid_data`：网格数据
- `reset_command`：重置命令

---

## ⚙️ 配置说明

### 算法配置（scanner_config.json）

```json
{
    "repulsionCoefficient": 4.0,           // 排斥力权重（避障）
    "entropyCoefficient": 2.0,             // 熵值权重（探索）
    "distanceCoefficient": 2.0,           // 距离权重（导航）
    "leaderRangeCoefficient": 2.0,        // Leader范围权重（跟随）
    "directionRetentionCoefficient": 2.0, // 方向保持权重（稳定）
    "updateInterval": 0.5,                 // 更新间隔（秒）
    "moveSpeed": 1.0,                      // 移动速度（米/秒）
    "scanRadius": 2.0,                    // 扫描半径（米）
    "altitude": 2.0,                       // 飞行高度（米）
    "maxRepulsionDistance": 5.0,          // 最大排斥距离
    "minSafeDistance": 1.0,               // 最小安全距离
    "targetSearchRange": 20.0,            // 目标搜索范围
    "avoidRevisits": true,                // 避免重复访问
    "revisitCooldown": 10.0               // 重复访问冷却时间
}
```

### DDPG 奖励配置（DDPG_Weight/dqn_reward_config.json）

```json
{
    "rewards": {
        "exploration_reward": 10.0,        // 探索奖励
        "collision_penalty": -50.0,       // 碰撞惩罚
        "out_of_range_penalty": -20.0,    // 超出范围惩罚
        "time_penalty": -0.1              // 时间惩罚
    },
    "thresholds": {
        "scanned_entropy_threshold": 30   // 已扫描熵值阈值
    }
}
```

---

## 🎮 使用指南

### 基本运行

1. **启动 Unity 项目**（Airsim2022）
2. **启动 Python 服务器**：
   ```bash
   python multirotor/AlgorithmServer.py
   ```
3. **等待连接**：系统自动连接 AirSim 和 Unity
4. **开始任务**：无人机自动起飞并开始扫描

### 多无人机运行

```bash
# 运行 3 台无人机
python multirotor/AlgorithmServer.py --drones 3

# 运行 5 台无人机 + DDPG 权重
python multirotor/AlgorithmServer.py --drones 5 --use-learned-weights
```

### 使用 DDPG 权重预测

```bash
# 使用默认模型（自动选择最佳）
python multirotor/AlgorithmServer.py --use-learned-weights

# 使用指定模型
python multirotor/AlgorithmServer.py --use-learned-weights \
    --model-path DDPG_Weight/models/best_model
```

### 数据采集

数据采集系统自动运行，输出到：
- `multirotor/DDPG_Weight/airsim_training_logs/scan_data_YYYYMMDD_HHMMSS.csv`

**CSV 格式**：
```csv
timestamp,elapsed_time,scanned_count,unscanned_count,total_count,scan_ratio,repulsion_coefficient,entropy_coefficient,distance_coefficient,leader_range_coefficient,direction_retention_coefficient,UAV1_pos_x,UAV1_pos_y,UAV1_pos_z,UAV1_battery_voltage,UAV2_pos_x,UAV2_pos_y,UAV2_pos_z,UAV2_battery_voltage
2026-01-26 15:10:33,0.00,0,25,25,0.00%,4.0,2.0,2.0,2.0,2.0,0.000,0.000,2.000,3.850,5.000,0.000,2.000,3.820
2026-01-26 15:10:34,1.00,3,22,25,12.00%,4.0,2.0,2.0,2.0,2.0,1.234,0.567,2.000,3.845,5.678,0.234,2.000,3.815
```

---

## 📊 数据可视化分析 ✨

### 功能简介

系统提供强大的数据可视化工具，支持对训练数据和扫描数据进行深度分析。

### 支持的数据类型

#### 1. Crazyflie 实体无人机训练数据

**数据格式**：
- JSON 完整训练日志：`crazyflie_training_log_*.json`
- CSV 飞行数据：`crazyflie_flight_*.csv`
- CSV 权重历史：`crazyflie_weights_*.csv`

**可视化内容**：
- 飞行轨迹（
- 速度和加速度曲线
- 权重变化历史
- Episode 奖励曲线
- 电池性能分析

**示例图表**：
- `UAV1_trajectory_2d.png` - 2D 飞行轨迹
- `UAV1_trajectory_3d.png` - 3D 飞行轨迹
- `UAV1_flight_stats.png` - 飞行状态（速度、高度、电池）
- `weight_history.png` - APF 权重系数变化
- `episode_stats.png` - Episode 统计信息

#### 2. DataCollector 扫描数据

**数据格式**：
- CSV 扫描数据：`scan_data_*.csv`

**可视化内容**：
- 扫描进度曲线
- 熵值变化分析
- 多无人机轨迹
- 算法权重变化

### 使用方法

#### 方式一：使用主菜单（推荐）

```bash
# 启动主菜单
start.bat

# 选择选项 [A] 数据可视化分析
```

**子菜单选项**：
- `[1]` 自动分析所有数据（推荐）- 自动扫描所有数据目录
- `[2]` 分析 Crazyflie 训练日志 - 只分析实体机训练数据
- `[3]` 分析扫描数据 - 只分析 DataCollector 数据
- `[4]` 分析指定文件 - 拖拽文件到窗口

#### 方式二：直接运行批处理

```bash
# Windows 中文版
scripts\数据可视化分析.bat

# Windows 英文版
scripts\Data_Visualization_Analysis.bat
```

#### 方式三：命令行使用

```bash
# 自动扫描所有数据
python multirotor/Algorithm/visualize_training_data.py --auto

# 分析单个 JSON 文件
python multirotor/Algorithm/visualize_training_data.py --json path/to/file.json

# 分析单个 CSV 文件
python multirotor/Algorithm/visualize_training_data.py --csv path/to/file.csv

# 分析指定目录
python multirotor/Algorithm/visualize_training_data.py --dir path/to/logs

# 指定输出目录
python multirotor/Algorithm/visualize_training_data.py --auto --out my_analysis
```

### 输出结果

**默认输出目录**：`analysis_results/`

**文件结构**：
```
analysis_results/
├── crazyflie_20260126_153022/     # Crazyflie 训练会话
│   ├── UAV1_trajectory_2d.png      # 2D 轨迹
│   ├── UAV1_trajectory_3d.png      # 3D 轨迹
│   ├── UAV1_flight_stats.png       # 飞行状态
│   ├── weight_history.png          # 权重历史
│   └── episode_stats.png           # Episode 统计
│
└── scan_data_20260126_150000/  # 扫描数据会话
    ├── scan_progress.png           # 扫描进度
    ├── trajectories_xy.png         # 水平轨迹
    ├── trajectories_3d.png         # 3D 轨迹
    └── algorithm_weights.png       # 算法权重
```

### 数据分析建议

#### Crazyflie 训练数据分析

1. **飞行轨迹分析**：
   - 检查飞行路径是否平滑
   - 分析速度变化是否合理
   - 检查是否有异常的加速度

2. **权重变化分析**：
   - 观察权重是否收敛
   - 分析不同权重系数的变化趋势
   - 检查权重变化与奖励的关系

3. **Episode 表现分析**：
   - 绘制奖励曲线
   - 分析 Episode 长度的变化
   - 评估训练效果和收敛速度

4. **电池性能分析**：
   - 监控电池电压变化
   - 评估飞行时间与电量的关系
   - 检查是否有电量过低的情况

#### 扫描数据分析

1. **扫描效率分析**：
   - 分析扫描完成度曲线
   - 评估扫描速度

2. **多机协同分析**：
   - 观察无人机轨迹分布
   - 检查是否有重叠区域

3. **熵值变化分析**：
   - 观察全局平均熵值变化
   - 分析熵值分布的演变

### 示例

**分析所有训练数据**：
```bash
start.bat
# 选择 [A] -> [1]
# 系统自动扫描并分析所有数据
# 结果保存在 analysis_results/ 目录
```

**分析特定训练会话**：
```bash
python multirotor/Algorithm/visualize_training_data.py \
    --json multirotor/DDPG_Weight/crazyflie_logs/crazyflie_training_log_20260126_153022.json
```

---

## ⚡ DDPG 强化学习

### 统一配置文件系统 ✨

**新特性**：从 v1.2.0 开始，所有训练模式统一使用 `unified_train_config.json` 配置文件。

**配置文件位置**：
- `multirotor/DDPG_Weight/unified_train_config.json`

**配置结构**：
```json
{
  "_comment": "统一训练配置文件 - 支持虚拟训练、实体训练、虚实融合训练",
  
  "common": {
    "total_timesteps": 100,
    "enable_visualization": true,
    "checkpoint_freq": 1000,
    "overwrite_model": false,
    "model_name": "weight_predictor"
  },
  
  "airsim_virtual": {
    "drone_names": ["UAV1", "UAV2", "UAV3"],
    "step_duration": 5.0,
    "model_name": "weight_predictor_airsim"
  },
  
  "crazyflie_online": {
    "drone_name": "UAV1",
    "step_duration": 5.0
  },
  
  "crazyflie_logs": {
    "log_path": "crazyflie_flight_log.json",
    "step_stride": 1
  },
  
  "hybrid": {
    "drone_names": ["UAV1", "UAV2", "UAV3"],
    "mirror_drones": ["UAV1"],
    "step_duration": 5.0
  }
}
```

**配置合并逻辑**：
- 每个训练模式会自动合并 `common` 和对应模式的配置
- 模式专用配置优先级高于 `common` 配置
- 例如：AirSim 训练使用 `common` + `airsim_virtual` 的合并结果

**向后兼容**：
- 所有训练脚本仍然支持旧配置文件：
  - `airsim_train_config_template.json`
  - `crazyflie_online_train_config.json`
  - `crazyflie_logs_train_config.json`
  - `hybrid_train_config_template.json`

### 模型覆盖控制 ✨

**新特性**：控制模型保存策略，避免频繁生成新模型。

**使用场景**：
1. **调试阶段**：使用覆盖模式，避免生成大量测试模型
2. **正式训练**：使用时间戳模式，保留每次训练的历史版本

**配置方式**：

1. **配置文件**：
```json
{
  "common": {
    "overwrite_model": false,  // false=生成新模型, true=覆盖现有模型
    "model_name": "weight_predictor_airsim"  // 模型基础名称
  }
}
```

2. **命令行参数**：
```bash
# 覆盖模式（固定名称）
python train_with_airsim_improved.py --overwrite-model --model-name my_model

# 新建模式（带时间戳）
python train_with_airsim_improved.py --model-name my_model
```

**模型命名规则**：
- **覆盖模式** (`overwrite_model=true`)：
  - 最佳模型：`best_{model_name}.zip`
  - 检查点：`ckpt_{checkpoint}_{model_name}.zip`
  - 最终模型：`{model_name}.zip`
  
- **新建模式** (`overwrite_model=false`)：
  - 最佳模型：`best_model_{timestamp}.zip`
  - 检查点：`checkpoint_{checkpoint}_{timestamp}.zip`
  - 最终模型：`{model_name}_{timestamp}.zip`

### 训练可视化器 ✨

**新特性**：实时显示训练进度和奖励曲线，分析模型收敛情况。

**显示内容**：
1. **Episode 奖励曲线**：每个 Episode 的总奖励
2. **平滑奖励曲线**：移动平均，观察趨势
3. **收敛分析**：
   - 训练状态：未收敛 / 收敛中 / 已收敛
   - 目标奖励：显示90%最大奖励基准线
   - 收敛进度：百分比显示
4. **实时统计**：当前 Episode、平均奖励、最大奖励、最小奖励

**启用方式**：
```json
{
  "common": {
    "enable_visualization": true
  }
}
```

**关闭可视化**：
```bash
python train_with_airsim_improved.py --no-visualization
```

### 训练模式

系统支持 4 种训练模式，均使用统一配置文件：

#### 1️⃣ 虚拟训练（AirSim 环境）

**适用场景**：快速迭代、安全测试、多无人机协同

**运行方式**：
```bash
# 方式 1：使用批处理脚本（推荐）
scripts\训练权重DDPG-真实环境.bat

# 方式 2：命令行（使用统一配置）
cd multirotor/DDPG_Weight
python train_with_airsim_improved.py

# 方式 3：指定自定义配置
python train_with_airsim_improved.py --config my_config.json

# 方式 4：命令行覆盖参数
python train_with_airsim_improved.py --overwrite-model --total-timesteps 500
```

**前置条件**：
- Unity AirSim 仿真场景已启动
- 配置中的无人机名称与 Unity 场景中一致

#### 2️⃣ 实体在线训练（Crazyflie）

**适用场景**：真实环境验证、在线调优

**运行方式**：
```bash
# 使用批处理脚本
scripts\训练权重DDPG-实体机在线.bat

# 命令行
cd multirotor/DDPG_Weight
python train_with_crazyflie_online.py
```

**前置条件**：
- Crazyflie 实体机已连接
- AlgorithmServer 和 Crazyswarm 已启动

#### 3️⃣ 实体离线训练（日志）

**适用场景**：离线分析、不影响实体机运行

**运行方式**：
```bash
# 使用批处理脚本
scripts\训练权重DDPG-实体机日志.bat

# 命令行
cd multirotor/DDPG_Weight
python train_with_crazyflie_logs.py
```

**前置条件**：
- 在配置中指定 `log_path`（.json 或 .csv 文件）

#### 4️⃣ 虚实融合训练

**适用场景**：结合虚拟和真实环境的优势

**运行方式**：
```bash
# 使用批处理脚本
scripts\训练权重DDPG-虚实融合.bat

# 命令行
cd multirotor/DDPG_Weight
python train_with_hybrid.py

# 指定镜像无人机（使用实体机数据）
python train_with_hybrid.py --mirror-drones UAV1 UAV2
```

**前置条件**：
- Unity AirSim 场景已启动
- Crazyflie 实体机已连接（用于镜像无人机）

**特点**：
- 指定的 `mirror_drones` 使用实体机实时数据
- 其他无人机使用 AirSim 虚拟数据

### 通用命令行参数

所有训练脚本支持以下参数：

```bash
--config PATH              # 指定配置文件路径
--overwrite-model          # 覆盖现有模型（不生成时间戳）
--model-name NAME          # 指定模型名称
--total-timesteps N        # 总训练步数
--no-visualization         # 关闭训练可视化
--continue-model PATH      # 继续训练指定模型
```

**示例**：
```bash
# 调试模式：快速迭代，覆盖模型
python train_with_airsim_improved.py \
  --overwrite-model \
  --model-name debug_model \
  --total-timesteps 100 \
  --no-visualization

# 生产模式：保留历史版本
python train_with_airsim_improved.py \
  --model-name production_v1 \
  --total-timesteps 10000

# 继续训练
python train_with_airsim_improved.py \
  --continue-model models/weight_predictor_airsim_20260126 \
  --total-timesteps 5000
```

### 旧配置文件说明

为了向后兼容，以下旧配置文件仍然可用：

- `airsim_train_config_template.json` - AirSim 虚拟训练配置
- `crazyflie_online_train_config.json` - Crazyflie 在线训练配置
- `crazyflie_logs_train_config.json` - Crazyflie 日志训练配置
- `hybrid_train_config_template.json` - 虚实融合训练配置

**使用方式**：
```bash
python train_with_airsim_improved.py --config airsim_train_config_template.json
python train_with_crazyflie_online.py --config crazyflie_online_train_config.json
```

**推荐使用统一配置文件** `unified_train_config.json`，更易于管理和维护。

### 使用训练好的模型

```bash
# 使用默认最佳模型
python AlgorithmServer.py --use-learned-weights

# 使用指定模型
python AlgorithmServer.py --use-learned-weights \
    --model-path DDPG_Weight/models/weight_predictor_airsim

# 使用时间戳模型
python AlgorithmServer.py --use-learned-weights \
    --model-path DDPG_Weight/models/weight_predictor_airsim_20260126_153022
```

### DQN 移动控制

**功能**：使用 DQN 直接控制无人机移动

**训练模型**：
```bash
cd multirotor/DQN_Movement

# 训练模型
python train_movement_with_airsim.py
```

**状态**：开发中，存在一些问题

---

## 📊 数据采集系统

### 功能说明

数据采集系统（`data_collector.py`）独立运行，自动采集：

1. **栅格状态统计**：
   - AOI 区域内已侦察栅格数
   - AOI 区域内未侦察栅格数
   - 扫描比例

2. **权重值记录**：
   - 5 个 APF 权重系数
   - 实时权重变化

### 配置

采集间隔可在 `AlgorithmServer.py` 中调整：
```python
self.data_collector = DataCollector(collection_interval=1.0)  # 1秒采集一次
```

### 输出文件

- **位置**：`multirotor/DDPG_Weight/airsim_training_logs/`
- **命名**：`scan_data_YYYYMMDD_HHMMSS.csv`
- **格式**：CSV，UTF-8 编码

---

## 🐛 故障排查

### 常见问题

#### 1. 无法连接到 AirSim

**症状**：日志显示连接失败

**解决方案**：
- 确认 AirSim 已启动
- 检查 `settings.json` 配置
- 确认端口未被占用
- 查看 AirSim 日志

#### 2. 无法连接到 Unity

**症状**：等待 Unity 连接超时

**解决方案**：
- 确认 Unity 项目已启动
- 检查防火墙设置
- 确认端口 41451 未被占用
- 查看 Unity Console 错误信息

#### 3. 可视化窗口不显示

**症状**：没有弹出可视化窗口

**解决方案**：
- 检查 pygame 是否安装：`pip install pygame`
- 确认图形环境可用（非无头服务器）
- 查看控制台错误信息
- 尝试禁用可视化：`--no-visualization`

#### 4. DDPG 模型加载失败

**症状**：模型加载错误

**解决方案**：
- 确认模型文件存在（`.zip` 文件）
- 检查 stable-baselines3 版本
- 查看模型路径是否正确（应为 `DDPG_Weight/models/`）
- 确认模型与代码版本兼容

#### 5. 无人机不移动

**症状**：无人机停留在初始位置

**解决方案**：
- 检查算法配置参数
- 确认网格数据已接收
- 查看算法日志输出
- 检查力向量计算是否正常

#### 6. 性能问题

**症状**：系统运行缓慢

**解决方案**：
- 降低更新频率（`updateInterval`）
- 减少无人机数量
- 关闭可视化
- 检查是否有内存泄漏

---

## 🛠️ 开发指南

### 添加新功能

1. **扩展算法**：
   - 修改 `Algorithm/scanner_algorithm.py`
   - 添加新的力向量计算
   - 更新配置数据类

2. **添加数据采集**：
   - 修改 `Algorithm/data_collector.py`
   - 添加新的采集字段
   - 更新 CSV 表头

3. **自定义可视化**：
   - 修改 `Algorithm/simple_visualizer.py`
   - 添加新的显示元素
   - 调整布局和颜色

### 代码规范

- **Python 风格**：遵循 PEP 8
- **命名规范**：使用有意义的变量名
- **注释**：添加必要的文档字符串
- **类型提示**：使用类型注解（Python 3.7+）

### 测试

```bash
# 运行基本测试
python -m pytest tests/

# 测试算法模块
python -c "from Algorithm.scanner_algorithm import ScannerAlgorithm; print('OK')"

# 测试数据采集
python -c "from Algorithm.data_collector import DataCollector; print('OK')"
```

---

## 📖 相关文档

### 项目文档
- **DDPG 与 DQN 介绍**：`multirotor/DDPG与DQN介绍.md`
- **Episode 循环说明**：`multirotor/DDPG_Weight/Episode循环说明.md`

---

## 📦 依赖项

### 核心依赖

```
stable-baselines3    # 强化学习库
gym                   # 强化学习环境
numpy                 # 数值计算
opencv-python         # 图像处理
pygame                # 可视化
tornado               # Web 框架（可选）
```

### AirSim 依赖

```
msgpack-python        # 消息序列化
msgpack-rpc-python    # RPC 通信
backports.ssl_match_hostname  # SSL 支持
```

### 完整列表

查看 `requirements.txt` 获取完整依赖列表。

---

## 🔄 版本信息

- **当前版本**：1.2.0
- **Python 版本**：3.7+
- **最后更新**：2026-01-26

### 更新日志

- **v1.2.0**（2026-01-26）
  - ✨ 新增统一配置文件系统（unified_train_config.json）
  - ✨ 新增模型覆盖控制功能（--overwrite-model）
  - ✨ 新增训练可视化器（Episode 奖励曲线、收敛分析）
  - ✨ 数据采集新增电量信息（电池电压）
  - ✨ 新增虚实融合训练模式
  - ✨ 新增 Crazyflie 实体无人机数据记录器
  - ✨ 新增训练数据可视化分析工具
  - 🔧 所有批处理脚本更新为使用统一配置
  - 🔧 训练脚本支持向后兼容旧配置文件
  - 📝 更新所有配置文件说明和使用指南

- **v1.1.0**（2026-01-21）
  - 增补 Crazyflie 实体机训练说明与配置
  - 更新脚本列表与项目结构说明

- **v1.0.0**（2025-11-28）
  - 添加数据采集系统
  - 支持 DDPG 权重预测
  - 多无人机协同控制
  - 实时可视化
  - 修正命名：DQN_Weight → DDPG_Weight

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

### 贡献指南

1. Fork 项目仓库
2. 创建功能分支
3. 提交代码并测试
4. 发起 Pull Request

### 代码规范

- 遵循 PEP 8 编码规范
- 添加适当的注释和文档
- 提交前进行测试

---

## 📄 许可证

暂未提供许可证信息。

---

## ⚠️ 注意事项

1. **首次运行**：需要安装所有依赖项
2. **AirSim 连接**：确保 AirSim 在 Python 服务器启动前运行
3. **Unity 连接**：Unity 项目必须在 Python 服务器启动后运行
4. **端口占用**：确保端口 41451 未被占用
5. **模型文件**：DDPG 模型文件较大，需要足够的存储空间
6. **目录命名**：确保使用 `DDPG_Weight` 而非 `DQN_Weight`（已重命名）

---

## 📞 联系方式

如有问题或建议，欢迎：
- 提交 Issue
- 发起 Discussion
- 其他联系方式：暂无

---

**开始您的无人机算法开发之旅！** 🚁✨

