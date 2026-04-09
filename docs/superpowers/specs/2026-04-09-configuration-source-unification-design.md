# Configuration Source Unification Design

## Goal

在不把各算法重新耦合到一起的前提下，统一系统级基础配置来源，形成清晰的配置边界：

- 系统级基础信息只由一个小型共享配置承载
- 各算法继续各管各的训练与奖励参数
- 迁移过程保持兼容，优先保证现有系统运行行为不变

## Design Summary

采用“小共享核心 + 各算法独立配置”的方案。

- 新增 `multirotor/system_config.json`
- 保留 `multirotor/drones_config.json`
- 保留 `multirotor/apf_algorithm_config.json`
- 保留各算法自己的 `configs/*.json`

共享配置只承载：

- 无人机清单
- 基础环境规则

不进入共享层的内容：

- APF 系数
- DDPG 训练模式与超参数
- DDPG 权重环境奖励配置
- DQN / HRL 奖励、阈值、训练超参数

## Target Responsibilities

### 1. `multirotor/system_config.json`

职责：系统级基础信息。

建议内容：

```json
{
  "drones": {
    "UAV1": {
      "enabled": true,
      "type": "virtual",
      "isCrazyflieMirror": false
    }
  },
  "environment": {
    "termination": {
      "target_scan_ratio": 0.25,
      "max_collision_count": 6,
      "max_elapsed_time_sec": 300.0,
      "stagnation_timeout_sec": 30.0,
      "out_of_range_reset_enabled": true,
      "out_of_range_continuous_count": 12
    },
    "battery": {
      "low_threshold": 3.5,
      "optimal_min": 3.7,
      "optimal_max": 4.1
    }
  }
}
```

只允许放：

- `drones`
- `environment.termination`
- `environment.battery`

明确不放：

- `base_rewards`
- APF 系数
- 各算法训练参数

### 2. `multirotor/drones_config.json`

职责：训练选择，而不是系统清单。

目标结构：

```json
{
  "training": {
    "dqn": { "use_all_drones": true },
    "ddpg": { "use_all_drones": true },
    "hierarchical": { "use_all_drones": true }
  }
}
```

说明：

- 无人机定义迁出到 `system_config.json`
- 本文件只描述“某个算法训练时使用哪些无人机”

### 3. `multirotor/apf_algorithm_config.json`

职责：APF 控制参数。

保留：

- `repulsionCoefficient`
- `entropyCoefficient`
- `distanceCoefficient`
- `leaderRangeCoefficient`
- `directionRetentionCoefficient`
- `groundRepulsionCoefficient`
- `updateInterval`
- `moveSpeed`
- `rotationSpeed`
- `scanRadius`
- `maxRepulsionDistance`
- `minSafeDistance`
- `avoidRevisits`
- `targetSearchRange`
- `revisitCooldown`
- `altitude`

迁出：

- `env_config`

### 4. Algorithm-Specific Configs

以下配置继续各管各的：

- `multirotor/DDPG_Weight/configs/unified_train_config.json`
- `multirotor/DDPG_Weight/configs/crazyflie_reward_config.json`
- `multirotor/DQN_Movement/configs/movement_dqn_config.json`
- `multirotor/DQN_Movement/configs/hierarchical_dqn_config.json`

这些文件不再承载系统级共享配置，只承载算法自身需要的参数。

## Read Priority

目标读取优先级如下：

1. 系统级基础信息
   - 只认 `multirotor/system_config.json`
2. 训练选择
   - 只认 `multirotor/drones_config.json`
3. APF 参数
   - 只认 `multirotor/apf_algorithm_config.json`
4. 各算法配置
   - 只认各自配置文件

## Migration Strategy

迁移分两阶段完成。

### Phase 1: Compatible Introduction

先引入共享配置，不立即删除旧位置字段。

实现要求：

- 新增 `multirotor/system_config.json`
- 新增共享 loader，例如 `multirotor/Algorithm/system_config.py`
- 所有需要系统清单的代码优先读 `system_config.json.drones`
- 所有需要基础环境规则的代码优先读 `system_config.json.environment`
- 如果新位置缺失，则兼容回退：
  - 无人机清单回退到旧的 `drones_config.json.drones`
  - 环境规则回退到旧的 `apf_algorithm_config.json.env_config`

这一步目标是“迁移入口”，不是“清理旧字段”。

### Phase 2: Cleanup

当所有入口都已经稳定切到新来源后，再删除旧位置共享字段：

- 删除 `drones_config.json.drones`
- 删除 `apf_algorithm_config.json.env_config`

这一步才真正完成配置源统一。

## Loader Design

建议新增两个小型入口，而不是在旧类里继续堆职责。

### `SystemConfig`

职责：

- 加载 `system_config.json`
- 提供 `get_all_drones()` / `get_enabled_drones()` / `get_drone_info()`
- 提供 `get_environment_rules()`
- 在兼容阶段负责旧字段回退

### `DronesTrainingConfig`

职责：

- 加载 `drones_config.json`
- 提供 `get_training_drones(algorithm)`
- 依赖 `SystemConfig` 校验无人机名称和启用状态

这样可以避免当前 `DronesConfig` 同时承担“系统清单 + 训练选择”的双重职责。

## Non-Goals

本设计明确不做这些事情：

- 不把所有配置折叠成单一总配置
- 不把 DQN / DDPG / HRL 奖励参数抽成共享配置
- 不统一训练输出目录、日志目录、模型目录
- 不改变任何算法默认参数或训练行为

## Error Handling

兼容阶段：

- 如果 `system_config.json` 缺失，允许回退并打印一次迁移提示
- 如果 `system_config.json` 存在但结构非法，直接报错

清理阶段：

- 移除回退逻辑
- 共享配置缺失或非法都应成为硬错误

## Testing Strategy

必须覆盖以下验证：

1. 共享配置解析测试
   - 新配置存在时正确读取
   - 新配置缺失时正确回退

2. 训练选择测试
   - `use_all_drones=true` 行为保持不变
   - `use_all_drones=false` + `drone_list` 行为保持不变

3. 环境规则测试
   - DQN / DDPG / HRL 读取到的新环境规则与旧逻辑等价

4. 回归测试
   - 迁移前后关键配置对象的有效值相同

## Risks

主要风险有三类：

1. 配置来源切换后，某些入口仍在偷偷读旧字段
2. 兼容阶段同时存在新旧来源，容易出现优先级不清
3. 若过早删除旧字段，会造成隐蔽回归

缓解方式：

- 先引入共享 loader，再集中切入口
- 兼容阶段明确“新优先，旧回退”
- 通过测试比较迁移前后的有效配置值

## Recommended Rollout

推荐落地顺序：

1. 新增 `system_config.json`
2. 新增共享 loader
3. 改 `drones_config.py`，拆成“系统清单”和“训练选择”两个职责
4. 改所有消费 `env_config` 的环境入口，先读共享配置
5. 跑兼容测试与回归测试
6. 删除旧字段
7. 更新文档与样例

## Decision Record

本设计选择：

- 以“每个算法各管各的”为主
- 只抽一个很小的共享公共配置
- 共享配置采用新文件 `multirotor/system_config.json`
- 无人机清单迁入共享配置
- 基础环境规则迁入共享配置
- APF 参数保留在 `apf_algorithm_config.json`
- 采用两阶段迁移，先兼容、后清理
