# 实飞高权重在线训练设计

日期：2026-04-07

## 摘要

本设计用于在现有 DDPG 权重训练链路中加入“实飞数据高影响力更新”能力。

整体训练主体仍然是基于 AirSim 的预训练。进入后续实飞阶段后，在线采集到的实飞 transition 不应只按样本数量自然影响模型，而应以更高权重参与模型更新。第一阶段的实现目标是：

- 在飞行过程中在线采集实飞数据
- 在每个 episode 结束后统一更新模型
- 让更新过程明显偏向实飞数据
- 将更新后的模型用于下一次实飞 episode

同时，设计上要为后续的 `inflight` 模式预留路径，也就是未来允许在一次飞行过程中更新模型，而不是只能在 episode 结束后更新。

## 问题背景

当前系统已经具备以下训练能力：

- AirSim 虚拟训练：`multirotor/DDPG_Weight/train_with_airsim_improved.py`
- 实飞在线训练：`multirotor/DDPG_Weight/train_with_crazyflie_online.py`
- 实飞日志训练：`multirotor/DDPG_Weight/train_with_crazyflie_logs.py`
- 虚实镜像混合训练：`multirotor/DDPG_Weight/train_with_hybrid.py`

但系统目前还没有一个明确机制，能够表达下面这个需求：

- 实飞样本数量很少
- 但每一条实飞样本对模型的影响要明显大于仿真样本

现阶段能做到的近似方式是分阶段训练：

1. 先在仿真中预训练
2. 再在实飞阶段继续训练

这确实有帮助，但它还不是一个显式、可配置、可观测的“实飞高权重影响”能力。

## 目标

- 保持 AirSim 预训练仍然是主训练阶段
- 支持在实体机飞行过程中在线采集实飞数据
- 增加显式的实飞高权重更新机制
- 第一阶段采用 `episode_end` 模式以降低安全风险
- 与现有 DDPG 环境、AlgorithmServer、Crazyflie 数据链路兼容
- 为未来切换到 `inflight` 更新模式预留清晰扩展点

## 非目标

- 第一阶段不替换 Stable-Baselines3 DDPG
- 第一阶段不重做 `AlgorithmServer`、Unity 通信或 Crazyflie 日志格式
- 第一阶段不顺带修改奖励函数定义
- 第一阶段不在同一次飞行中热切换模型参数

## 推荐方案

推荐采用“实飞优先的双层缓冲更新设计”，并基于当前在线 DDPG 流程落地。

核心思想如下：

- 保持仿真预训练链路不变
- 在实飞阶段，把在线采集的实飞 transition 存入专用的 `real_buffer`
- 在每个 episode 结束后，执行额外的实飞优先更新轮次
- 如果未来需要，也允许引入少量仿真 transition 参与混合 batch，但默认仍由实飞数据主导

相比一上来就做 `inflight` 在线热更新，这个方案更适合当前代码结构，也更安全。它仍然能够让少量实飞数据在下一次飞行前对模型产生远超样本数量占比的影响。

## 为什么符合当前架构

当前代码已经具备合适的分层边界：

- 训练入口：`multirotor/DDPG_Weight/train_with_crazyflie_online.py`
- 实飞环境：`multirotor/DDPG_Weight/envs/crazyflie_weight_env.py`
- 仿真预训练入口：`multirotor/DDPG_Weight/train_with_airsim_improved.py`

在线环境已经完整暴露了一个标准 transition 流程：

- 从实飞运行时状态读取当前 observation
- 用策略输出 action
- 飞行一步后计算 reward
- 再读取新的 real state 作为 next observation

这说明当前缺的不是数据入口，而是训练阶段的加权策略和更新调度机制。

## 总体设计

### 1. 更新时机模式

引入明确的更新时机概念：

- `episode_end`：飞行过程中只采集 transition，不更新模型；episode 结束后统一更新，并把新模型用于下一次飞行
- `inflight`：预留给未来，在飞行过程中按更严格的安全策略进行更新

第一阶段只实现 `episode_end`，但新增接口和配置时都要兼容时机模式，避免后续扩展时推翻现有设计。

### 2. 实飞优先训练器

新增一个训练器侧组件，专门负责“实飞高权重”逻辑。

建议职责如下：

- 接收在线环境产生的 transition
- 为 transition 打上数据来源标签，例如 `real`
- 在 episode 结束时触发额外训练轮次
- 根据配置执行实飞优先的更新策略

建议命名：

- `RealFlightPriorityTrainer`

该组件应放在 `multirotor/DDPG_Weight/` 下，保持为训练层组件，不直接拥有飞控逻辑。

### 3. 感知数据来源的 transition 存储

新增感知数据来源的 transition 存储设计。

最小实现如下：

- `real_buffer`：存储当前及最近若干个实飞 episode 采集到的 transition
- `sim_buffer`：可选，未来如需做混合 batch 再引入

第一阶段即使只实现 `real_buffer` 也可以满足需求。因为仿真预训练得到的模型参数本身已经携带了仿真知识，所以第一阶段没有必要把仿真 transition 再导入实飞高权重训练器，这是最简单也最稳妥的版本。

每条 transition 至少包含：

- `observation`
- `action`
- `reward`
- `next_observation`
- `done`
- `source`
- `episode_index`
- `step_index`
- `timestamp`

## 加权策略

第一阶段需要通过训练器配置显式表达“实飞数据占主导”。

建议新增以下控制项：

- `real_update_multiplier`
  - 每个实飞 episode 结束后，基于实飞数据额外执行多少轮更新
- `real_batch_ratio`
  - 混合 batch 中实飞样本的目标占比
- `min_real_samples_before_update`
  - 实飞样本数低于阈值时跳过更新，避免样本太少导致不稳定
- `max_real_updates_per_episode`
  - 每个 episode 最多额外执行多少轮更新，避免计算量过大或过拟合
- `real_buffer_capacity`
  - `real_buffer` 最大容量

第一阶段推荐默认策略：

- episode 结束后的高权重更新只使用实飞 transition
- 当实飞样本数不足时，本次高权重更新直接跳过

这样实现最简单，也最容易解释“为什么这次模型会被实飞数据强力拉动”。

## 数据流

### AirSim 预训练阶段

1. 通过现有虚拟训练链路在 AirSim 中完成预训练
2. 保存预训练 DDPG 模型
3. 进入实飞训练阶段时，以该模型作为初始化模型

### 实飞 episode 阶段

1. `train_with_crazyflie_online.py` 读取预训练模型
2. `CrazyflieOnlineWeightEnv` 正常执行一次实飞 episode
3. 每一步交互产生的 transition 被复制到 `real_buffer`
4. 本次飞行过程中，控制侧始终使用该 episode 开始时的稳定模型
5. episode 结束后，由优先训练器执行配置好的实飞高权重更新
6. 如果更新验证通过，更新后的模型用于下一次实飞 episode

## 组件边界

### `train_with_crazyflie_online.py`

改造后的职责：

- 解析新增的加权配置
- 构建优先训练器
- 读取预训练模型
- 负责整体编排：
  - 执行飞行 episode
  - 收集实飞 transition
  - 触发 episode 结束后的高权重更新
  - 保存模型和训练元数据

它仍然是顶层编排入口，不应把 buffer 和加权细节直接塞进这个脚本里。

### `CrazyflieOnlineWeightEnv`

改造后的职责：

- 继续以标准 Gym 风格产生 transition
- 暴露足够的每步数据，便于训练器干净地存储 transition
- 不直接负责加权策略和更新调度

环境本身应继续聚焦“与实体平台交互”这一件事。

### 新增训练器辅助组件

改造后的职责：

- 管理带来源标签的 transition 存储
- 执行 episode 结束后的高权重更新
- 返回更新结果与统计信息
- 为未来的 `inflight` 模式保留统一接口

## 配置设计

建议在在线训练配置中增加一个嵌套块，例如：

```json
{
  "crazyflie_online": {
    "update_timing": "episode_end",
    "enable_real_weighting": true,
    "real_update_multiplier": 4,
    "real_batch_ratio": 1.0,
    "min_real_samples_before_update": 32,
    "max_real_updates_per_episode": 8,
    "real_buffer_capacity": 5000,
    "rollback_on_bad_update": true
  }
}
```

含义如下：

- `real_batch_ratio = 1.0` 表示 episode 结束后的高权重更新只使用实飞数据
- 后续如果需要混合 batch，可以用 `0.7` 之类的值
- `rollback_on_bad_update` 表示更新后如发现异常则自动回滚

## 安全与失败处理

之所以第一阶段先做 `episode_end`，核心原因就是安全。

必须具备的保护措施：

- 第一阶段不允许在同一次飞行中热切换模型参数
- 如果某个 episode 产出的有效实飞 transition 太少，则跳过更新
- 开始高权重更新前保存一份更新前检查点
- 出现以下情况时自动回滚到更新前模型：
  - loss 出现非有限值
  - 更新过程异常崩溃
  - 更新后的 sanity check 不通过

建议的 sanity check：

- 策略参数中没有 NaN 或 Inf
- 参数变化幅度不超过配置阈值
- 在一组固定 probe state 上采样得到的 action 仍在合理范围内

## 可观测性

系统需要输出足够的日志，让我们能够解释“本次实飞数据到底对模型产生了多大影响”。

必须记录的内容：

- 当前 episode 采集到的实飞样本数
- 本次高权重更新是否执行
- 额外更新轮数
- 实际 batch 构成
- 参数变化摘要
- 是否触发回滚

建议输出字段：

- `real_samples_collected`
- `weighted_update_rounds`
- `weighted_update_status`
- `real_buffer_size`
- `policy_param_delta_norm`
- `rollback_triggered`

## 测试策略

### 单元测试

- transition 存储是否正确
- 加权配置解析是否正确
- 样本不足时是否正确跳过更新
- 回滚路径是否正确

### 集成测试

- 能否把 AirSim 预训练模型加载到在线训练入口
- 使用模拟实飞 episode 时，transition 是否进入 `real_buffer`
- episode 结束后是否正确触发高权重更新
- 更新后的模型是否在下一次 episode 中生效

### 安全回归测试

- 验证 `episode_end` 模式下不会在飞行中途更新模型
- 验证更新失败后能正确恢复到更新前模型

## 向未来 `inflight` 模式迁移的路径

第一阶段实现时应主动预留以下扩展点：

- `update_timing` 配置项
- 训练器 API 同时支持按 step 调用和按 episode 调用
- 模型切换机制与环境 step 过程解耦

未来从 `episode_end` 升级到 `inflight` 时，主要变化应该是：

- 增加按步数或按时间间隔触发的更新策略
- 增加更严格的 action 变化幅度与参数变化幅度保护
- 必要时引入 shadow model 提升流程，而不是直接热切换

只要第一阶段守住这些边界，后续从 `episode_end` 切换到 `inflight` 应该属于中等扩展，而不是重新设计。

## 风险

- 单个实飞 episode 的样本数可能过少，难以支撑稳定更新
- 实飞数据权重过高时，可能过拟合到某一次飞行条件、电量状态或环境扰动
- 如果实现过度耦合 SB3 内部私有接口，后续升级库版本会变得困难

## 最终决策

第一阶段按以下方式实现：

- 保持 AirSim 预训练链路不变
- 保持实飞在线采集链路不变
- 新增带来源标签的 `real_buffer`
- 在 episode 结束后执行实飞主导的高权重更新
- 配置层提前为未来的 `inflight` 模式预留接口

这样可以在第一阶段就让系统具备真正意义上的“实飞数据高影响力”能力，同时避免把同一次飞行中的热更新风险一起引入。
