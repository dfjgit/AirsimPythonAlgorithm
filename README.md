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
- ✅ **数据采集系统**：自动采集扫描数据和权重值
- ✅ **可视化工具**：2D 实时可视化（熵值、无人机位置等）
- ✅ **配置管理**：JSON 配置文件，参数可调

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
│   │   └── data_collector.py        # 数据采集模块
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
│   │   ├── test_trained_model.py    # 模型测试
│   │   ├── models/                  # 训练好的模型
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
│   └── data_logs/                    # 数据采集输出
│       └── scan_data_YYYYMMDD_HHMMSS.csv
│
├── scripts/                          # 批处理脚本
│   ├── 运行系统-固定权重.bat
│   ├── 运行系统-DDPG权重.bat
│   ├── 训练权重DDPG-真实环境.bat
│   ├── 训练权重DDPG-实体机日志.bat
│   ├── 训练权重DDPG-实体机在线.bat
│   └── 训练移动DQN-真实环境.bat
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

**功能**：自动采集扫描数据和权重值

**采集内容**：
- 时间戳和运行时间
- AOI 区域内栅格状态（已侦察/未侦察）
- 扫描比例
- 5 个权重系数值

**输出格式**：CSV 文件
- 位置：`multirotor/data_logs/scan_data_YYYYMMDD_HHMMSS.csv`
- 频率：每秒一次

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
- `multirotor/data_logs/scan_data_YYYYMMDD_HHMMSS.csv`

**CSV 格式**：
```csv
timestamp,elapsed_time,scanned_count,unscanned_count,total_count,scan_ratio,repulsion_coefficient,entropy_coefficient,distance_coefficient,leader_range_coefficient,direction_retention_coefficient
2025-11-28 15:10:33,0.00,0,25,25,0.00%,4.0,2.0,2.0,2.0,2.0
2025-11-28 15:10:34,1.00,3,22,25,12.00%,4.0,2.0,2.0,2.0,2.0
```

---

## 🧠 DDPG 强化学习

### DDPG 权重预测

**功能**：使用 DDPG 强化学习动态调整 APF 算法权重

**训练模型**：
```bash
# 进入 DDPG_Weight 目录
cd multirotor/DDPG_Weight

# 训练模型（真实 AirSim 环境）
python train_with_airsim_improved.py

# 或使用批处理脚本
..\..\scripts\训练权重DDPG-真实环境.bat
```

**注意**：虽然批处理文件名仍包含"DQN"，但实际使用的是 DDPG 算法。

**可选：使用配置文件训练**：
```bash
cd multirotor/DDPG_Weight
python train_with_airsim_improved.py --config airsim_train_config_template.json
```

**配置文件模板**：
- `multirotor/DDPG_Weight/airsim_train_config_template.json`
- `multirotor/DDPG_Weight/last_weights_template.json`

**配置字段说明（AirSim 训练）**：
- `drone_names`：训练无人机名称列表
- `total_timesteps`：训练总步数
- `step_duration`：每步飞行时长（秒）
- `checkpoint_freq`：检查点保存频率
- `enable_visualization`：是否启用训练可视化
- `use_initial_weights`：是否继承初始权重
- `initial_model_path`：初始模型路径（不含 `.zip`），会自动匹配同名权重文件
- `safety_limit`：是否启用权重变化安全限制
- `max_weight_delta`：每步权重变化最大幅度

**模型与权重文件命名**：
- 训练保存的模型文件名包含时间戳：`weight_predictor_airsim_<timestamp>.zip`
- 初始权重文件与模型同名：`weight_predictor_airsim_<timestamp>.last_weights.json`

**初始权重模板说明**：
- `last_weights_template.json` 用于提供初始权重的示例格式
- 支持按无人机名称映射权重（如 `UAV1`、`UAV2`）
- 字段包括：`repulsionCoefficient`、`entropyCoefficient`、`distanceCoefficient`、`leaderRangeCoefficient`、`directionRetentionCoefficient`

**使用模型**：
```bash
# 使用训练好的模型
python AlgorithmServer.py --use-learned-weights \
    --model-path DDPG_Weight/models/best_model
```

### Crazyflie 实体无人机训练

**配置文件**：
- `multirotor/DDPG_Weight/crazyflie_logs_train_config.json`
- `multirotor/DDPG_Weight/crazyflie_online_train_config.json`
- `multirotor/DDPG_Weight/crazyflie_reward_config.json`

**配置字段说明（日志训练）**：
- `log_path`：日志文件路径（.json/.csv）
- `total_timesteps`：训练总步数
- `reward_config`：奖励配置文件路径，`null` 表示使用默认
- `save_dir`：模型保存目录
- `continue_model`：继续训练模型路径（不含 `.zip`），`null` 表示从头训练
- `max_steps`：每个 episode 最大步数，`null` 表示不限制
- `random_start`：是否随机起始位置
- `step_stride`：日志步进间隔（每隔 N 条取一条）
- `progress_interval`：进度打印间隔（步）

**配置字段说明（在线训练）**：
- `drone_name`：训练无人机名称
- `total_timesteps`：训练总步数
- `step_duration`：每步飞行时长（秒）
- `reward_config`：奖励配置文件路径，`null` 表示使用默认
- `save_dir`：模型保存目录
- `continue_model`：继续训练模型路径（不含 `.zip`），`null` 表示从头训练
- `reset_unity`：每个 episode 是否重置 Unity 环境
- `safety_max_delta`：权重变化最大幅度（安全限制）
- `no_safety_limit`：是否关闭权重变化限制
- `progress_interval`：进度打印间隔（步）

**奖励配置字段说明**（`crazyflie_reward_config.json`）：
- `reward_coefficients`：奖励系数
  - `speed_reward`：速度奖励系数
  - `speed_penalty_threshold`：速度惩罚阈值
  - `speed_penalty`：速度惩罚系数
  - `accel_penalty`：加速度惩罚系数
  - `angular_rate_penalty`：角速度惩罚系数
  - `scan_reward`：扫描奖励系数
  - `out_of_range_penalty`：超出范围惩罚系数
  - `action_change_penalty`：动作变化惩罚系数
  - `action_magnitude_penalty`：动作幅度惩罚系数
  - `battery_optimal_reward`：电池电压在最佳范围的奖励系数
  - `battery_low_penalty`：电池电压过低惩罚系数
- `thresholds`：阈值配置
  - `scan_entropy_threshold`：扫描熵值阈值
  - `leader_range_buffer`：Leader 范围缓冲
  - `battery_optimal_min`：电池最佳电压下限
  - `battery_optimal_max`：电池最佳电压上限
  - `battery_low_threshold`：电池低电压阈值
- `episode`：训练 episode 配置
  - `max_steps`：单个 episode 最大步数
- `action_space`：动作空间范围
  - `weight_min`：权重最小值
  - `weight_max`：权重最大值

**离线日志训练（不影响状态转移）**：
```bash
cd multirotor/DDPG_Weight
python train_with_crazyflie_logs.py --config crazyflie_logs_train_config.json
```

**在线实体训练（实时日志）**：
```bash
cd multirotor/DDPG_Weight
python train_with_crazyflie_online.py --config crazyflie_online_train_config.json
```

**Windows 脚本**：
```bat
scripts\Train_DDPG_Weights_Crazyflie_Logs.bat
scripts\Train_DDPG_Weights_Crazyflie_Online.bat
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

- **位置**：`multirotor/data_logs/`
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

- **当前版本**：1.1.0
- **Python 版本**：3.7+
- **最后更新**：2026-01-21

### 更新日志

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

