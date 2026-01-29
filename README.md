# AirsimAlgorithmPython - 无人机算法服务器

基于 Python 的多无人机协同控制算法服务器，支持人工势场算法（APF）、DDPG 强化学习权重预测和实时数据采集。

---

## 📋 项目简介

AirsimAlgorithmPython 是无人机仿真系统的算法核心，提供智能控制、路径规划和强化学习训练功能。系统通过 TCP Socket 与 Unity 仿真环境实时通信，实现多无人机协同控制、区域扫描和探索任务。

### 核心特性

- ✅ **人工势场算法（APF）**：多因素权重合成，实现避障与探索的动态平衡。
- ✅ **科学论证支持 (PoC)**：内置系统活跃度、多机加速比、续航闭环、学习速率、**策略收敛性**及**姿态稳定性**等量化证明工具。
- ✅ **双算法强化学习体系**：支持 DDPG 权重预测（连续动作空间）与 DQN 移动控制（离散动作空间），实现算法参数的智能进化与对比实验。
- ✅ **全链路数据采集**：同步记录物理、决策、电池及训练元数据，确保分析的闭环完整性。
- ✅ **虚实融合训练**：支持虚拟仿真与实体无人机数据的混合实时训练（Hybrid Training）。
- ✅ **跨格式透明分析**：无缝对比 JSON（实体机）与 CSV（仿真）数据，自动字段对齐，支持 DDPG vs DQN 算法性能对比。
- ✅ **多无人机协同**：支持 1-10 台无人机集群控制，提供协同效能量化评估。

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
│   │   ├── data_collector.py        # 数据采集模块
│   │   ├── HexGridDataModel.py      # 网格数据模型
│   │   ├── battery_data.py          # 电池状态模型
│   │   ├── simple_visualizer.py     # 2D 实时可视化
│   │   ├── visualize_scan_csv.py    # CSV 数据分布可视化
│   │   └── visualize_training_data.py # 科学论证分析工具 (PoC) ✨
│   │
│   ├── AirsimServer/                # 服务器组件
│   │   ├── drone_controller.py      # 无人机控制器
│   │   ├── unity_socket_server.py   # Unity 通信服务
│   │   └── data_pack.py             # 数据包定义
│   │
│   ├── Crazyswarm/                  # Crazyflie 实体机支持
│   │   ├── crazyflie_operate.py     # 实体机控制
│   │   └── crazyswarm.py            # Crazyswarm 集成
│   │
│   ├── DDPG_Weight/                 # DDPG 权重预测模块 ⭐
│   │   ├── configs/                 # 配置文件目录
│   │   ├── envs/                    # 强化学习环境与日志记录器
│   │   ├── models/                  # 训练好的模型
│   │   ├── tests/                   # 集成与单元测试
│   │   ├── train_with_airsim_improved.py  # 训练脚本（仿真）
│   │   ├── train_with_crazyflie_logs.py   # 日志训练
│   │   ├── train_with_crazyflie_online.py # 在线训练
│   │   ├── train_with_hybrid.py     # 虚实融合训练
│   │   ├── train_simple.py          # 基础训练脚本
│   │   └── training_visualizer.py   # 训练实时可视化
│   │
│   ├── DQN_Movement/                 # DQN 移动控制模块 ⭐
│   │   ├── models/                  # 训练好的模型
│   │   ├── logs/                    # 训练日志（自动生成）
│   │   ├── movement_env.py          # 移动环境定义
│   │   ├── train_movement_dqn.py    # DQN 训练脚本
│   │   ├── train_movement_with_airsim.py  # AirSim 环境训练
│   │   ├── test_movement_dqn.py     # DQN 模型测试
│   │   └── movement_dqn_config.json # DQN 配置文件
│   │
│   ├── setup_path.py                # 路径设置工具
│   └── DDPG与DQN介绍.md              # 强化学习模块说明
│
├── scripts/                          # 批处理脚本
│   ├── 运行系统-固定权重.bat
│   ├── 运行系统-DDPG权重.bat
│   ├── 训练权重DDPG-真实环境.bat
│   ├── 训练权重DDPG-实体机日志.bat
│   ├── 训练权重DDPG-实体机在线.bat
│   ├── 训练权重DDPG-虚实融合.bat
│   ├── 训练移动DQN-真实环境.bat
│   ├── 测试移动DQN.bat              # ⭐ DQN 模型测试
│   ├── 数据可视化分析.bat          # ⭐ 科学论证入口
│   └── (对应的英文版批处理文件)
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

### 3. DataCollector（数据采集系统）

**功能**：全自动多维度数据实时采集与落盘，打通“执行-决策-训练”数据全链路。

**采集内容**：
- **物理扫描数据**：AOI 区域内栅格状态（已侦察/未侦察）、实时扫描比例（Scan Ratio）。
- **飞行遥测数据**：多机实时 3D 位置（x, y, z）、欧拉角（Roll, Pitch, Yaw）、电池电压（Battery Voltage）。
- **算法决策数据**：APF 算法的 5 个实时权重系数。
- **强化学习元数据**：Episode 编号、Step 步数、单步奖励（Step Reward）、累计总奖励（Total Reward）。

**数据落盘**：
- **存储路径**：`multirotor/DDPG_Weight/airsim_training_logs/scan_data_YYYYMMDD_HHMMSS.csv`
- **对齐特性**：采用线程安全注入技术，将异步的强化学习奖励数据与同步的扫描详志完美对齐。

**数据可视化**：
- 通过 `start.bat` -> `[A]` 或运行 `scripts/数据可视化分析.bat` 进行深度量化分析。

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

### DDPG 奖励配置

系统使用 JSON 文件定义强化学习的奖励函数：
- **仿真环境**：`multirotor/DDPG_Weight/configs/dqn_reward_config.json`
- **实体环境**：`multirotor/DDPG_Weight/configs/crazyflie_reward_config.json`

**核心配置项**：
```json
{
    "rewards": {
        "exploration_reward": 10.0,        // 探索奖励（发现新栅格）
        "collision_penalty": -50.0,       // 碰撞惩罚
        "out_of_range_penalty": -20.0,    // 超出领机范围惩罚
        "battery_low_penalty": -10.0      // 低电量惩罚
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

## 📊 数据可视化分析 (Scientific Proof of Concept) ✨

### 功能简介

系统提供了一套严谨的科学分析工具，不仅用于显示图表，更用于为算法效能、系统稳定性和工程可行性提供量化证明。

### 核心分析维度 (量化证明)

#### 1. 效能里程碑与覆盖分析 (Efficiency Analysis)
- **任务完成证明**：自动标注 50%、80%、90% 及 95% 覆盖率的达成时间（达成秒数）。
- **覆盖增长率**：分析覆盖率随时间的一阶导数，识别算法搜索效率的“爆发期”与“衰退期”。

#### 2. 系统活跃度与无死锁证明 (Liveness Analysis)
- **活跃度监控**：通过 `liveness_analysis.png` 监控系统实时覆盖速率。
- **死锁判定**：通过速率曲线是否长期归零科学判定系统是否存在死锁或搜索停滞点，证明算法在复杂环境下的持续推进能力。

#### 3. 续航耐力与闭环证明 (Endurance Analysis)
- **电压-进度双轴图**：在 `battery_endurance_analysis.png` 中对齐电压下降曲线（左轴）与扫描进度曲线（右轴）。
- **任务闭环证明**：证明在电池电压下降到安全阈值之前，扫描任务已达到 90% 以上的完成度，验证工程落地的可行性。

#### 4. 学习速度与策略收敛证明 (Learning & Convergence Analysis)
- **奖励增长斜率**：使用线性回归（Linear Regression）量化奖励曲线的上升斜率。
- **策略收敛性量化证明**：通过计算训练后期权重参数的滚动统计量（Rolling Std/Variance），量化证明策略已趋于稳定，而非发散震荡。
- **收敛状态判定**：系统自动根据波动强度判定“已收敛”或“震荡中”，为模型可用性提供科学依据。

#### 5. 飞行姿态稳定性证明 (Attitude Stability Analysis)
- **姿态波动监控**：在 `flight_attitude_stability.png` 中记录 Roll/Pitch 欧拉角随时间的变化。
- **安全性判定**：自动计算姿态抖动（Jitter）统计值。若波动长期处于低位且无剧烈抖动，即证明飞控系统在算法驱动下具备极高鲁棒性，无失控风险。

#### 6. 多机协作加速比证明 (Collaboration Speedup)
- **加速比 (Speedup)**：计算公式为 $S = T_{single} / T_{multi}$。
- **协同证明**：通过对比 1 台、3 台、5 台无人机的任务达成耗时，量化证明多机系统相比单机系统的效能提升倍数。

#### 7. DDPG vs DQN 算法对比 (Algorithm Comparison) ✨
- **算法对比**：自动对比 DDPG（连续动作空间，权重预测）与 DQN（离散动作空间，移动控制）的训练效果。
- **多维度分析**：
  - **奖励曲线对比**：可视化两种算法的学习曲线，观察收敛速度差异。
  - **收敛速度对比**：通过线性回归计算奖励增长斜率和 R² 值，量化学习效率。
  - **最终性能对比**：对比最后 10 个 Episode 的平均奖励，评估算法稳态表现。
  - **学习稳定性对比**：分析奖励方差，评估算法在训练过程中的波动程度。
- **对比报告生成**：自动生成文本格式的详细对比报告，包含算法简介、收敛统计、性能指标和应用建议。

### 跨格式兼容性支持

系统已实现对**虚拟仿真**与**实体飞行**数据格式的“透明对齐”：
- **混合对比**：支持同时选取虚拟训练产出的 `training_stats_*.csv` 和实体训练产出的 `*.json` 进行奖励曲线对比。
- **字段对齐**：自动将实体 JSON 中的 `length` 映射为虚拟 CSV 中的 `steps`，消除数据孤岛。

### 使用方法

#### 方式一：使用主菜单 (推荐)
1. 启动 `start.bat`。
2. 选择选项 `[A] 数据可视化分析`。
3. 进入子菜单选择：
   - `[5] 多个实验数据对比分析`（用于生成加速比和学习速度对比）
   - 或选择主菜单 `[B] DDPG vs DQN 算法对比`（专门的算法对比分析）

#### 方式二：直接运行批处理
- 运行 `scripts/数据可视化分析.bat`。

#### 方式三：命令行
```bash
# 生成单个实验的详尽分析图表（含活跃度、续航、同步奖励）
python multirotor/Algorithm/visualize_training_data.py --csv scan_data_XYZ.csv --show

# 进行多个实验的量化对比分析（生成加速比、学习速率对比）
python multirotor/Algorithm/visualize_training_data.py --auto --compare

# DDPG vs DQN 算法对比分析 ✨
python multirotor/Algorithm/visualize_training_data.py --auto --compare-algorithms
```

### 输出结果

分析结果统一保存至 `analysis_results/` 目录下：
- `comparison_results/`：包含多机加速比、覆盖进度对比。
- `comparison_training/`：包含奖励曲线叠加对比、学习速率斜率对比。
- `algorithm_comparison_ddpg_vs_dqn/`：✨ **DDPG vs DQN 算法对比结果**（奖励曲线、收敛速度、最终性能、稳定性分析、对比报告）。
- `dqn_training_*/`：✨ **DQN 训练专用分析**（奖励曲线、Episode 长度、学习速度、总结统计）。
- `scan_data_*/`：包含该次实验的活跃度、续航分析、实时同步奖励等 10+ 张专业图表。

---

## ⚡ DDPG 强化学习

### 统一配置文件系统 ✨

**新特性**：从 v1.2.0 开始，所有训练模式统一使用 `unified_train_config.json` 配置文件。

**配置文件位置**：
- `multirotor/DDPG_Weight/configs/unified_train_config.json`

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

### DQN 移动控制 ⭐

**功能**：使用 DQN 直接控制无人机移动（离散动作空间）

**算法特点**：
- **动作空间**：6 个离散方向（上/下/左/右/前/后）
- **观察空间**：21 维向量（位置、速度、朝向、熵值统计、Leader 信息、扫描进度等）
- **应用场景**：适合离散决策任务，可与 DDPG 进行对比实验

#### 训练 DQN 模型

**方式一：使用主菜单**（推荐）
```bash
start.bat
# 选择 [8] 训练移动DQN (真实AirSim环境)
```

**方式二：使用批处理脚本**
```bash
# 中文版
scripts\训练移动DQN-真实环境.bat

# 英文版
scripts\Train_DQN_Movement_Real_Environment.bat
```

**方式三：命令行**
```bash
cd multirotor/DQN_Movement

# 使用默认配置训练
python train_movement_dqn.py

# 使用 AirSim 环境训练
python train_movement_with_airsim.py
```

**前置条件**：
- Unity AirSim 仿真场景已启动（如使用 AirSim 训练）
- 配置文件 `movement_dqn_config.json` 已正确设置

**训练输出**：
- **模型文件**：`multirotor/DQN_Movement/models/movement_dqn_final.zip`
- **训练日志**：`multirotor/DQN_Movement/logs/dqn_training_YYYYMMDD_HHMMSS/`
  - `dqn_training_stats.csv` - Episode 奖励、长度、时间统计
  - `dqn_training_metadata.json` - 完整的训练配置和环境信息

#### 测试 DQN 模型

**方式一：使用主菜单**（推荐）
```bash
start.bat
# 选择 [D] 测试移动DQN模型
```

**方式二：使用批处理脚本**
```bash
# 中文版
scripts\测试移动DQN.bat

# 英文版
scripts\Test_DQN_Movement.bat
```

**方式三：命令行**
```bash
cd multirotor/DQN_Movement
python test_movement_dqn.py
```

#### 分析 DQN 训练数据

```bash
# 分析指定 DQN 训练日志
python multirotor/Algorithm/visualize_training_data.py --dir multirotor/DQN_Movement/logs --show

# 自动扫描并分析所有 DQN 数据
python multirotor/Algorithm/visualize_training_data.py --auto
```

**生成的分析图表**：
1. **奖励曲线**：Episode 总奖励 + 移动平均平滑曲线
2. **Episode 长度**：每个 Episode 的步数变化
3. **学习速度分析**：奖励增长斜率（Learning Rate）
4. **总结统计**：平均奖励、最大/最小奖励、总 Episode 数等

#### DDPG vs DQN 对比分析

**方式一：使用主菜单**（推荐）
```bash
start.bat
# 选择 [B] DDPG vs DQN 算法对比
```

**方式二：直接运行批处理脚本**
```bash
# 需要先训练 DDPG 和 DQN 模型
start.bat  # 选择 [4] 训练 DDPG，选择 [8] 训练 DQN

# 然后进行对比分析
start.bat  # 选择 [B]
```

**方式三：命令行（灵活控制）**
```bash
# 基础对比（4图+报告）
python multirotor/Algorithm/visualize_training_data.py --auto --compare-algorithms --out analysis_results

# 全方位对比（6图+详细报告）
python multirotor/Algorithm/visualize_training_data.py --auto --compare-algorithms --compare-algorithms-full --out analysis_results

# 仅分析单个算法数据
python multirotor/Algorithm/visualize_training_data.py --auto --out analysis_results
```

**对比内容**：

**基础对比** (`--compare-algorithms`)：
1. **奖励曲线对比**：DDPG（红色）vs DQN（蓝绿色）学习曲线叠加
2. **收敛速度对比**：奖励增长斜率（Learning Rate）+ R² 拟合优度
3. **最终性能对比**：最后 10 个 Episode 的平均奖励（带误差棒）
4. **学习稳定性对比**：10-Episode 滚动标准差分析（波动性评估）
5. **文本对比报告**：`comparison_report.txt` 包含算法简介、统计数据、结论建议

**全方位对比** (`--compare-algorithms-full`，包含基础对比所有内容 + 以下扩展）：
6. **参数敏感性对比**：不同超参数设置下的性能变化
7. **长期趋势对比**：整个训练周期的奖励增长趋势
8. **样本效率对比**：达到目标奖励所需的训练步数

**输出目录结构**：
```
analysis_results/
├── DDPG_scan_data_XXXXXX/          # DDPG 单独分析（11张图表）
├── DQN_scan_data_XXXXXX/           # DQN 单独分析（11张图表）
├── dqn_movement_XXXXXX/            # DQN 训练专用分析（4张图表）
├── algorithm_comparison_ddpg_vs_dqn/       # 基础对比（4图+报告）
└── algorithm_comparison_ddpg_vs_dqn_full/  # 全方位对比（6图+详细报告）
```

#### 应用建议

- **DDPG**：适合连续参数优化（APF 权重预测），输出平滑的权重值
- **DQN**：适合离散决策任务（移动方向选择），动作明确且易于理解
- **组合使用**：DQN 控制移动策略 + DDPG 优化 APF 权重，实现多层次智能控制
- **对比实验**：通过算法对比分析，为论文提供量化实验证据

**状态说明**：✅ **已完全集成并生产就绪**。DQN 模块现已完成数据标准化、可视化分析和算法对比功能，可用于实验和论文研究。

**模型状态**：
- ✅ DDPG 模型：`multirotor/DDPG_Weight/models/` 目录下存在多个训练版本
- ✅ DQN 模型：`multirotor/DQN_Movement/models/movement_dqn_airsim_final.zip`（已训练完成）

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
- **Episode 循环说明**：`multirotor/DDPG_Weight/docs/Episode循环说明.md`
- **图表预览与分析指南**：`multirotor/DDPG_Weight/docs/CHART_PREVIEW_USAGE.md`

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

- **当前版本**：1.2.3
- **Python 版本**：3.7+
- **最后更新**：2026-01-29 (算法对比完善版)

### 更新日志

- **v1.2.3**（2026-01-29）
  - ✨ **全方位算法对比**：扩展 DDPG vs DQN 对比分析，新增学习稳定性、最终性能、参数敏感性等 6 维度深度对比。
  - ✨ **对比报告增强**：算法对比报告新增收敛速度、稳定性分析和应用建议章节，为论文提供详实数据支撑。
  - ✨ **主菜单优化**：完善 `start.bat` 菜单，选项 [B] 现支持基础对比和全方位对比的自动化流程。
  - 🔧 **输出目录规范化**：对比结果统一保存至 `analysis_results/algorithm_comparison_*` 目录，层次清晰。
  - 📚 **文档完善**：README 新增算法对比的三种使用方式（主菜单/批处理/命令行）和输出结构说明。

- **v1.2.2**（2026-01-27）
  - ✨ **DQN 模块完全集成**：DQN 移动控制模块完成生产化改造，支持标准化数据输出。
  - ✨ **算法对比分析体系**：新增 DDPG vs DQN 算法性能对比功能，生成多维度对比图表和详细报告。
  - ✨ **DQN 专用可视化分析器**：在 `visualize_training_data.py` 中添加 `DQNDataVisualizer` 类，支持 4 种分析图表。
  - ✨ **主菜单扩展**：在 `start.bat` 中添加 DQN 训练、测试和算法对比选项（选项 [8]、[D]、[B]）。
  - ✨ **批处理脚本完善**：新增 `测试移动DQN.bat` 和 `Test_DQN_Movement.bat` 测试脚本。
  - 🔧 **数据格式标准化**：DQN 训练数据输出为 CSV（episode, reward, length, elapsed_time）+ JSON（元数据）。
  - 📚 **README 文档更新**：补充完整的 DQN 使用指南、对比分析说明和应用建议。

- **v1.2.1**（2026-01-26）
  - ✨ **飞行姿态稳定性证明**：新增欧拉角波动分析，支持多机 Roll/Pitch 抖动（Jitter）量化判定与平稳性验证。
  - ✨ **策略收敛性量化增强**：引入滚动方差（Rolling Variance）分析，提供更严谨的算法稳态判定判据。
  - 🔧 **数据底座扩充**：在 CSV 详志中集成姿态、电量、奖励和权重，实现环境-执行-决策全时域同步存储。

- **v1.2.0**（2026-01-26）
  - ✨ **科学论证 (PoC) 体系建立**：新增系统活跃度、多机加速比、学习速率斜率及续航闭环量化分析工具。
  - ✨ **全链路数据打通**：实现 `DataCollector` 与强化学习环境的同步，将 Reward/Episode 注入扫描详志。
  - ✨ **跨格式混合分析**：支持 JSON（实体）与 CSV（虚拟）数据的字段对齐与叠加对比分析。
  - ✨ **统一配置文件系统**：所有训练模式迁移至 `unified_train_config.json`，支持虚实融合训练模式。
  - ✨ **模型覆盖控制**：新增 `--overwrite-model` 参数，支持固定名称覆盖或时间戳版本控制。
  - ✨ **训练实时可视化**：增强版 `training_visualizer` 支持平滑奖励曲线与收敛自动判定。
  - 🔧 **目录布局重构**：优化 `DDPG_Weight` 结构，划分为 `configs/`, `envs/`, `docs/`, `tests/` 模块。
  - 🔧 **故障自诊断**：更新分析脚本，支持自动识别异常数据点并标注任务里程碑。

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

