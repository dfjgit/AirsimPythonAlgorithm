# DQN奖励配置系统更新日志

**更新日期**: 2025-10-13  
**版本**: v1.0

---

## 📋 更新概述

将DQN训练过程中的所有奖励系数和阈值参数提取到独立的配置文件中，实现灵活的参数调整和实验管理。

---

## 🎯 主要变更

### 1. 新增文件

#### `dqn_reward_config.json` - 奖励配置文件
包含所有奖励系数、阈值参数、Episode设置和动作空间参数。

**位置**: `multirotor/DQN/dqn_reward_config.json`

**配置分类**:
- **reward_coefficients**: 奖励系数（探索、碰撞、越界、平滑运动）
- **thresholds**: 阈值参数（碰撞距离、移动范围、熵值阈值等）
- **episode**: Episode设置（最大步数）
- **action_space**: 动作空间参数（权重范围、平滑参数）

---

#### `dqn_reward_config_data.py` - 配置数据类
提供配置的加载、保存和管理功能。

**主要功能**:
```python
# 加载配置
config = DQNRewardConfig("dqn_reward_config.json")

# 修改参数
config.exploration_reward = 2.0
config.collision_penalty = 10.0

# 保存配置
config.save_to_file("custom_config.json")

# 打印配置
print(config)
```

---

#### `DQN_CONFIG_GUIDE.md` - 配置使用指南
详细的配置参数说明和调参策略文档。

**内容包括**:
- 所有配置项的详细说明
- 参数调整建议和范围
- 不同训练阶段的推荐配置
- 常见问题解决方案

---

### 2. 修改文件

#### `simple_weight_env.py` - DQN环境类

**主要修改**:

1. **导入配置模块**:
```python
from dqn_reward_config_data import DQNRewardConfig
```

2. **初始化加载配置**:
```python
def __init__(self, server=None, drone_name="UAV1", reward_config_path=None):
    # 加载配置
    self.reward_config = DQNRewardConfig(reward_config_path)
```

3. **使用配置参数**:

所有硬编码的魔法数字都替换为配置参数：

| 原硬编码值 | 配置参数 | 说明 |
|-----------|---------|------|
| `0.5, 5.0` | `weight_min, weight_max` | 权重范围 |
| `1.5` | `std_threshold` | 标准差阈值 |
| `0.7` | `std_smoothing` | 平滑系数 |
| `5` | `max_min_ratio` | 最大最小比 |
| `200` | `max_steps` | 最大步数 |
| `1.0` | `exploration_reward` | 探索奖励 |
| `5.0` | `collision_penalty` | 碰撞惩罚 |
| `2.0` | `out_of_range_penalty` | 越界惩罚 |
| `0.1` | `smooth_movement_reward` | 平滑奖励 |
| `2.0` | `collision_distance` | 碰撞距离 |
| `0.5, 3.0` | `movement_min, movement_max` | 移动范围 |
| `30` | `scanned_entropy_threshold` | 熵值阈值 |
| `10.0` | `nearby_entropy_distance` | 附近距离 |

---

## 🔧 使用方法

### 快速开始

**默认使用（自动加载配置）**:
```python
env = SimpleWeightEnv(server=server, drone_name="UAV1")
```

**指定配置文件**:
```python
env = SimpleWeightEnv(
    server=server,
    drone_name="UAV1",
    reward_config_path="custom_reward_config.json"
)
```

---

### 调整参数

**方法1: 直接编辑JSON文件**
```json
{
    "reward_coefficients": {
        "exploration_reward": 2.0,  // 修改这里
        "collision_penalty": 10.0
    }
}
```

**方法2: 程序化修改**
```python
from dqn_reward_config_data import DQNRewardConfig

config = DQNRewardConfig("dqn_reward_config.json")
config.exploration_reward = 2.0
config.save_to_file("new_config.json")
```

---

## ✅ 测试验证

### 配置类测试
```bash
python multirotor/DQN/dqn_reward_config_data.py
```

**预期输出**:
- 创建默认配置
- 保存到文件
- 从文件加载
- 修改并保存

---

### 环境测试
```bash
python multirotor/DQN/simple_weight_env.py
```

**预期输出**:
- 成功加载配置
- 环境初始化成功
- 执行多个步骤
- 显示奖励计算结果

---

## 📊 优势

### 1. 灵活性
- 无需修改代码即可调整参数
- 支持多套配置快速切换
- 便于A/B测试和参数对比

### 2. 可维护性
- 集中管理所有奖励相关参数
- 消除魔法数字，提高代码可读性
- 配置变更可追踪版本

### 3. 可扩展性
- 易于添加新的奖励项
- 支持不同场景的配置
- 便于团队协作和实验

### 4. 可重现性
- 完整记录训练参数
- 保存实验配置便于复现
- 配合文档详细说明每个参数

---

## 🔄 向后兼容

**完全兼容**: 现有代码无需修改，配置系统会自动使用默认值。

```python
# 旧代码仍然有效
env = SimpleWeightEnv(server=server, drone_name="UAV1")
# 自动加载 dqn_reward_config.json
```

---

## 📚 相关文档

- **配置使用指南**: [DQN_CONFIG_GUIDE.md](DQN_CONFIG_GUIDE.md)
- **DQN主文档**: [README.md](README.md)
- **权重平衡说明**: [WEIGHT_BALANCING.md](WEIGHT_BALANCING.md)

---

## 🎓 调参建议

### 初期训练（关注探索）
```json
{
    "reward_coefficients": {
        "exploration_reward": 3.0,
        "collision_penalty": 2.0
    }
}
```

### 中期训练（平衡性能）
```json
{
    "reward_coefficients": {
        "exploration_reward": 1.5,
        "collision_penalty": 5.0
    }
}
```

### 后期训练（精细调优）
```json
{
    "reward_coefficients": {
        "exploration_reward": 1.0,
        "collision_penalty": 10.0
    },
    "thresholds": {
        "scanned_entropy_threshold": 20
    }
}
```

---

## ⚠️ 注意事项

1. **配置修改后需要重新训练模型**
2. **建议保留配置文件备份**
3. **每次实验记录使用的配置**
4. **逐步调整参数，避免一次改动过多**
5. **使用版本控制管理配置文件**

---

## 🐛 已知问题

**无**

---

## 🔮 未来计划

- [ ] 添加配置验证和范围检查
- [ ] 支持配置文件版本管理
- [ ] 添加配置对比工具
- [ ] 集成到训练日志系统
- [ ] 支持动态配置热加载

---

**维护者**: 项目团队  
**最后更新**: 2025-10-13

