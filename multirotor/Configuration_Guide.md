# 配置文件参数说明

本文档描述了 `scanner_config.json` 配置文件中各个参数的作用。

## 人工势场算法参数

### 权重系数
- **repulsionCoefficient** (2.0): 排斥力权重，控制无人机间避障强度
- **entropyCoefficient** (2.0): 熵权重，控制探索未知区域的倾向
- **distanceCoefficient** (2.0): 距离权重，控制向目标移动的倾向
- **leaderRangeCoefficient** (2.0): Leader范围权重，控制保持在Leader范围内的倾向
- **directionRetentionCoefficient** (2.0): 方向保持权重，控制飞行稳定性

### 基础参数
- **updateInterval** (1): 算法更新间隔，单位：秒
- **moveSpeed** (2.0): 无人机移动速度，单位：米/秒
- **rotationSpeed** (120.0): 无人机旋转速度，单位：度/秒
- **scanRadius** (1.0): 扫描半径，单位：米
- **altitude** (2.0): 飞行高度，单位：米

### 安全参数
- **maxRepulsionDistance** (5.0): 最大排斥距离，单位：米
- **minSafeDistance** (1.0): 最小安全距离，单位：米

### 目标选择策略
- **avoidRevisits** (true): 是否避免重复访问已扫描区域
- **targetSearchRange** (20.0): 目标搜索范围，单位：米
- **revisitCooldown** (10.0): 重复访问冷却时间，单位：秒

## DQN学习参数

### 基础配置
- **enabled** (false): 是否启用DQN学习
- **learning_rate** (0.001): 学习率，控制模型学习速度
- **gamma** (0.99): 折扣因子，控制未来奖励的重要性
- **epsilon** (1.0): 初始探索率，控制随机探索的概率
- **epsilon_min** (0.01): 最小探索率
- **epsilon_decay** (0.995): 探索率衰减因子

### 训练参数
- **batch_size** (64): 训练批次大小
- **target_update** (10): 目标网络更新频率，单位：步数
- **memory_capacity** (10000): 经验回放缓冲区大小
- **model_save_interval** (1000): 模型保存间隔，单位：步数

## 奖励函数参数

- **exploration_weight** (1.0): 探索奖励权重
- **efficiency_weight** (0.5): 效率奖励权重
- **collision_penalty** (-5.0): 碰撞惩罚值
- **boundary_penalty** (-2.0): 越界惩罚值
- **energy_penalty** (-0.1): 能耗惩罚值
- **completion_reward** (100.0): 任务完成奖励值

## 学习环境参数

### 权重调整
- **coefficient_step** (0.5): 权重调整步长

### 权重范围限制
- **repulsionCoefficient**: [0.1, 10.0] - 排斥力系数范围
- **entropyCoefficient**: [0.1, 10.0] - 熵系数范围
- **distanceCoefficient**: [0.1, 10.0] - 距离系数范围
- **leaderRangeCoefficient**: [0.1, 10.0] - Leader范围系数范围
- **directionRetentionCoefficient**: [0.1, 10.0] - 方向保持系数范围

## 系统参数

- **name** ("ScannerConfigData"): 配置名称标识
- **hideFlags** (0): 隐藏标志（系统内部使用）

## 使用建议

1. **首次使用**: 建议使用默认参数值
2. **性能调优**: 根据实际环境调整权重系数
3. **DQN训练**: 启用DQN学习时，建议先使用较小的学习率
4. **安全设置**: 根据无人机尺寸调整安全距离参数
5. **环境适应**: 根据扫描区域大小调整搜索范围参数
