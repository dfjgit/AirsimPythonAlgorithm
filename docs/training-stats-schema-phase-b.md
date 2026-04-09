# 训练统计 Schema 收口说明（方案 B）

## 背景

在外部可视化进程模式下，DDPG、DQN、HRL 都需要通过 `server.current_training_stats`
把训练状态传给快照和页面。

之前的问题是：

- 不同模块对同一含义使用了不同字段名
  - 例如 `current_episode_time` 和 `episode_elapsed_time`
- 不同训练脚本/环境写入字段的完整度不同
  - 有的只写 `step`
  - 有的写 `total_steps`
  - 有的没有稳定维护 `current_episode_steps`
- 可视化器各自做字段兜底，久而久之出现漂移

## 本次方案 B 做了什么

本次没有引入全链路强类型模型，而是先统一“通用训练统计协议”。

新增共享模块：

- `multirotor/training_stats_schema.py`

这个模块负责两件事：

1. 提供一套通用训练统计默认结构
2. 把历史字段名和不完整字段归一化为统一 schema

## 当前统一后的通用字段

以下字段视为可视化通用训练统计字段：

- `episode_count`
- `total_steps`
- `current_episode_steps`
- `current_step_reward`
- `current_episode_reward`
- `steps_per_sec`
- `current_episode_time`
- `episode_elapsed_time`
- `last_episode_duration`
- `total_training_time`
- `reward_history`
- `episode_reward_history`
- `avg_reward`
- `max_reward`
- `min_reward`

说明：

- `current_episode_time` 是页面统一读取字段
- `episode_elapsed_time` 继续保留，作为兼容别名
- 如果只有 `step` 没有 `total_steps`，会自动折叠成 `total_steps`
- 如果只有 `total_reward` 没有 `current_episode_reward`，会自动折叠

## 当前接入点

### 写入端

- `multirotor/AlgorithmServer.py`
  - `current_training_stats` 初始化
  - `set_training_stats()`
  - `reset_episode_timer()`

- `multirotor/DQN_Movement/scripts/train_movement_with_airsim.py`
  - DQN 可视化回调写回 `server.current_training_stats`

### 快照端

- `multirotor/AlgorithmServer.py`
  - `get_visualization_snapshot()`

- `multirotor/Visualization/external_visualizer_client.py`
  - 接收快照后再次归一化，保证客户端读到的是统一字段

### 读取端

- `multirotor/Visualization/ddpg_training_visualizer.py`
- `multirotor/Visualization/dqn_movement_visualizer.py`
- `multirotor/Visualization/hierarchical_training_visualizer.py`
- `multirotor/Visualization/panels/training_stats_panel.py`

## 方案 B 的边界

本次只统一“通用训练统计字段”，没有动算法专属字段的组织方式。

例如这些字段仍然保持专属透传：

- DQN:
  - `action_counts`
  - `last_action`
  - `per_drone_actions`
  - `drone_name`
  - `leader_distance`
  - `out_of_range_steps`
  - `last_done_reason`

- DDPG:
  - `weights`
  - `weight_history`

也就是说：

- 通用训练信息已经统一
- 算法专属展示信息仍由各自页面维护

## 如果后续升级到方案 C

方案 C 不是再加更多别名，而是把这套通用 schema 升级为强类型模型。

建议升级顺序：

1. 在 `multirotor/training_stats_schema.py` 基础上引入明确的数据类型
   - 可选：`dataclass`
   - 可选：`TypedDict`
   - 可选：校验模型

2. 将以下链路改成显式构造/传递该类型，而不是自由拼装 `dict`
   - 环境 `env`
   - 训练回调 `callback`
   - `AlgorithmServer.current_training_stats`
   - 快照序列化/反序列化
   - 各可视化器

3. 把算法专属字段拆成两层
   - `common_training_stats`
   - `algorithm_specific_stats`

这样做完后，可以进一步避免：

- 字段名被手写漂移
- 0 值/默认值覆盖真实值
- 新页面复制旧页面逻辑时继续引入分叉

## 当前结论

目前已经完成方案 B：

- DDPG / DQN / HRL 的通用训练统计读取方式已经收口到统一 schema
- 外部可视化快照中的 `current_training_stats` 已经过统一归一化
- 后续如需方案 C，可在当前文件基础上继续向强类型推进，而不需要重做方案 B
