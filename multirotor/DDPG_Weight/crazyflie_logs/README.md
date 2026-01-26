# Crazyflie 实体无人机训练数据日志

本目录用于存储实体无人机（Crazyflie）在训练过程中记录的飞行数据和权重变化历史。

## 📁 数据格式

### 1. JSON 完整数据文件

**文件命名**: `crazyflie_training_log_YYYYMMDD_HHMMSS.json`

**数据结构**:
```json
{
  "metadata": {
    "session_id": "20260126_153022",
    "start_time": "2026-01-26 15:30:22",
    "duration_seconds": 1850.5,
    "drone_names": ["UAV1", "UAV2"],
    "total_episodes": 25,
    "data_format": "crazyflie_training_log_v1.0"
  },
  "flight_data": {
    "UAV1": [
      {
        "id": 1,
        "x": -0.5,
        "y": 0.51,
        "z": -0.038,
        "time": 0.0,
        "speed": 0.036605,
        "xspeed": 0.029988,
        "yspeed": -0.000625,
        "zspeed": 0.020983,
        "battery": 4.2,
        "elapsed_time": 0.0,
        "session_id": "20260126_153022",
        "drone_name": "UAV1",
        ...
      },
      ...
    ]
  },
  "weight_history": [
    {
      "timestamp": "2026-01-26 15:30:25",
      "elapsed_time": 3.5,
      "session_id": "20260126_153022",
      "drone_name": "UAV1",
      "episode": 1,
      "step": 10,
      "repulsionCoefficient": 4.2,
      "entropyCoefficient": 2.5,
      "distanceCoefficient": 2.1,
      "leaderRangeCoefficient": 2.3,
      "directionRetentionCoefficient": 2.0
    },
    ...
  ],
  "episode_stats": [
    {
      "timestamp": "2026-01-26 15:30:50",
      "elapsed_time": 28.0,
      "session_id": "20260126_153022",
      "episode": 1,
      "reward": 125.5,
      "length": 50
    },
    ...
  ]
}
```

### 2. CSV 飞行数据文件

**文件命名**: `crazyflie_flight_UAV1_YYYYMMDD_HHMMSS.csv`

**字段说明**:
- `id`: 无人机 ID
- `x, y, z`: 位置坐标（米）
- `time`: 时间戳（秒）
- `qx, qy, qz, qw`: 四元数姿态
- `speed`: 总速度（米/秒）
- `xspeed, yspeed, zspeed`: 各轴速度（米/秒）
- `acceleratedspeed`: 总加速度
- `xacceleratedspeed, yacceleratedspeed, zacceleratedspeed`: 各轴加速度
- `xeulerangle, yeulerangle, zeulerangle`: 欧拉角（度）
- `xpalstance, ypalstance, zpalstance`: 角速度
- `xaccfpalstance, yaccfpalstance, zaccfpalstance`: 角加速度
- `battery`: 电池电压（伏特）
- `elapsed_time`: 训练开始后的经过时间（秒）
- `session_id`: 训练会话 ID
- `drone_name`: 无人机名称

### 3. CSV 权重历史文件

**文件命名**: `crazyflie_weights_YYYYMMDD_HHMMSS.csv`

**字段说明**:
- `timestamp`: 时间戳
- `elapsed_time`: 训练开始后的经过时间（秒）
- `session_id`: 训练会话 ID
- `drone_name`: 无人机名称
- `episode`: Episode 编号
- `step`: 步数
- `repulsionCoefficient`: 排斥力系数
- `entropyCoefficient`: 熵值系数
- `distanceCoefficient`: 距离系数
- `leaderRangeCoefficient`: Leader 范围系数
- `directionRetentionCoefficient`: 方向保持系数

## 🚀 使用场景

### 1. Crazyflie 在线训练

训练脚本会自动记录实体无人机的飞行数据：

```bash
cd multirotor/DDPG_Weight
python train_with_crazyflie_online.py --drone-name UAV1
```

训练结束后，数据会自动保存到 `crazyflie_logs/` 目录。

### 2. 虚实融合训练

如果在虚实融合训练中使用实体无人机作为镜像：

```bash
cd multirotor/DDPG_Weight
python train_with_hybrid.py --mirror-drones UAV1 UAV2
```

系统会记录所有镜像无人机（UAV1 和 UAV2）的数据。

## 📊 数据可视化

可以使用现有的 CSV 可视化工具来分析数据：

```bash
cd multirotor/Algorithm
python visualize_scan_csv.py ../../DDPG_Weight/crazyflie_logs/crazyflie_flight_UAV1_20260126_153022.csv
```

或者使用 Python 脚本直接读取 JSON 数据进行自定义分析：

```python
import json
import pandas as pd
import matplotlib.pyplot as plt

# 读取 JSON 数据
with open('crazyflie_training_log_20260126_153022.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 转换为 DataFrame
flight_data_uav1 = pd.DataFrame(data['flight_data']['UAV1'])
weight_history = pd.DataFrame(data['weight_history'])

# 绘制飞行轨迹
plt.figure(figsize=(10, 8))
plt.plot(flight_data_uav1['x'], flight_data_uav1['y'])
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('UAV1 Flight Trajectory')
plt.grid(True)
plt.show()

# 绘制权重变化
plt.figure(figsize=(12, 6))
for col in ['repulsionCoefficient', 'entropyCoefficient', 'distanceCoefficient']:
    plt.plot(weight_history['step'], weight_history[col], label=col)
plt.xlabel('Training Step')
plt.ylabel('Coefficient Value')
plt.title('Weight Evolution During Training')
plt.legend()
plt.grid(True)
plt.show()
```

## 📝 注意事项

1. **数据大小**: 实体无人机训练数据可能会很大，长时间训练可能产生几 MB 到几十 MB 的数据。

2. **自动保存**: 即使训练被 Ctrl+C 中断，系统也会尝试保存已收集的数据。

3. **多无人机**: 每个无人机的飞行数据会保存在单独的 CSV 文件中，但所有数据都会合并到一个 JSON 文件中。

4. **数据清理**: 定期清理旧的训练日志文件，避免占用过多磁盘空间。

5. **数据完整性**: JSON 文件包含完整的训练会话信息，包括元数据、飞行数据、权重历史和 Episode 统计。建议保留 JSON 文件用于完整分析。

## 🔍 数据分析建议

### 飞行轨迹分析
- 检查飞行路径是否平滑
- 分析速度变化是否合理
- 检查是否有异常的加速度或角速度

### 权重变化分析
- 观察权重是否收敛
- 分析不同权重系数的变化趋势
- 检查权重变化与奖励的关系

### Episode 表现分析
- 绘制奖励曲线
- 分析 Episode 长度的变化
- 评估训练效果和收敛速度

### 电池性能分析
- 监控电池电压变化
- 评估飞行时间与电量的关系
- 检查是否有电量过低的情况
