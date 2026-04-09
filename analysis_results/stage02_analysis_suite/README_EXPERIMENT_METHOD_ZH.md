# Stage02 图表方法学与公平性说明

## 1. 本文档的用途

本文档用于回答两个在论文写作中必须说明的问题：

1. `stage02` 图表的横轴、纵轴分别是什么，来源于哪些原始字段，以及经过了哪些聚合处理。  
2. 在 `DDPG+APF` 与 `纯 DQN` 步长、动作语义不一致的情况下，哪些指标可以直接比较，哪些指标只能谨慎解释。

## 2. 图表元数据总表

| 图名 | 图类型 | 横轴中文名 | 横轴字段 | 纵轴中文名 | 纵轴字段 | 原始数据来源 | 聚合方式 | 可比性 |
|---|---|---|---|---|---|---|---|---|
| `ddpg_stage02/episode_reward.png` | 单算法过程图 | 训练轮次 | `episode` | 单轮累计奖励 | `reward` | `training csv` | 每个 episode 一行 | 单算法可用 |
| `ddpg_stage02/episode_length.png` | 单算法过程图 | 训练轮次 | `episode` | 单轮步数长度 | `length` | `training csv` | 每个 episode 一行 | 单算法可用 |
| `ddpg_stage02/collision_stability.png` | 单算法稳定性图 | 训练轮次 | `episode` | 碰撞终止占比(%) | `collision_rate` | `training csv` | 每个 episode 一行 | 单算法可用 |
| `ddpg_stage02/collision_count_trend.png` | 单算法碰撞图 | 训练轮次 | `episode` | 碰撞次数 | `collision_count` | `training csv` | 每个 episode 一行 | 单算法可用 |
| `ddpg_stage02/global_scan_ratio.png` | 单算法结果图 | 训练轮次 | `episode` | 最终全局扫描率(%) | `final_global_scan_ratio` | `training csv` | 每个 episode 一行 | 单算法可用 |
| `ddpg_stage02/global_avg_entropy.png` | 单算法结果图 | 训练轮次 | `episode` | 最终全局平均熵 | `final_global_avg_entropy` | `training csv` | 每个 episode 一行 | 单算法可用 |
| `ddpg_stage02/scan_efficiency.png` | 单算法效率图 | 训练轮次 | `episode` | 扫描效率(Cell/Step) | `scan_efficiency` | `training csv` | 每个 episode 一行 | 单算法可用 |
| `ddpg_stage02/trajectories_xz.png` | 单算法轨迹图 | X坐标 | `UAVx_x` | Z坐标 | `UAVx_z` | `scan_data csv` | 按时间连接 | 单算法可用 |
| `dqn_stage02/episode_reward.png` | 单算法过程图 | 训练轮次 | `episode` | 单轮累计奖励 | `reward` | `training csv` | 每个 episode 一行 | 单算法可用 |
| `dqn_stage02/episode_length.png` | 单算法过程图 | 训练轮次 | `episode` | 单轮步数长度 | `length` | `training csv` | 每个 episode 一行 | 单算法可用 |
| `dqn_stage02/collision_stability.png` | 单算法稳定性图 | 训练轮次 | `episode` | 碰撞终止占比(%) | `collision_rate` | `training csv` | 每个 episode 一行 | 单算法可用 |
| `dqn_stage02/collision_count_trend.png` | 单算法碰撞图 | 训练轮次 | `episode` | 碰撞次数 | `collision_count` | `training csv` | 每个 episode 一行 | 单算法可用 |
| `dqn_stage02/global_scan_ratio.png` | 单算法结果图 | 训练轮次 | `episode` | 最终全局扫描率(%) | `final_global_scan_ratio` | `training csv` | 每个 episode 一行 | 单算法可用 |
| `dqn_stage02/global_avg_entropy.png` | 单算法结果图 | 训练轮次 | `episode` | 最终全局平均熵 | `final_global_avg_entropy` | `training csv` | 每个 episode 一行 | 单算法可用 |
| `dqn_stage02/scan_efficiency.png` | 单算法效率图 | 训练轮次 | `episode` | 扫描效率(Cell/Step) | `scan_efficiency` | `training csv` | 每个 episode 一行 | 单算法可用 |
| `dqn_stage02/trajectories_xz.png` | 单算法轨迹图 | X坐标 | `UAVx_x` | Z坐标 | `UAVx_z` | `scan_data csv` | 按时间连接 | 单算法可用 |
| `comparison/comparison_training_reward.png` | 横向过程图 | 训练轮次 | `episode` | 累计奖励 | `reward` | `training csv` | 同算法按 episode 统计 | 弱可比 |
| `comparison/comparison_training_scan_efficiency.png` | 横向过程图 | 训练轮次 | `episode` | 扫描效率(Cell/Step) | `scan_efficiency` | `training csv` | 同算法按 episode 统计 | 弱可比 |
| `comparison/comparison_training_collision_rate.png` | 横向过程图 | 训练轮次 | `episode` | 碰撞终止占比(%) | `collision_rate` | `training csv` | 同算法按 episode 统计 | 中等可比 |
| `comparison/comparison_training_collision_count.png` | 横向过程图 | 训练轮次 | `episode` | 碰撞次数 | `collision_count` | `training csv` | 同算法按 episode 统计 | 中等可比 |
| `comparison/comparison_scan_scan_ratio.png` | 横向结果图 | 训练轮次 | `episode` | 最终全局扫描率(%) | `final_global_scan_ratio` | `training csv` | 同算法按 episode 统计 | 强可比 |
| `comparison/comparison_scan_global_avg_entropy.png` | 横向结果图 | 训练轮次 | `episode` | 最终全局平均熵 | `final_global_avg_entropy` | `training csv` | 同算法按 episode 统计 | 强可比 |
| `comparison/comparison_scan_per_second.png` | 横向归一化结果图 | 训练轮次 | `episode` | 按时间归一化扫描产出(Cell/s) | `avg_scan_cells_per_second` | `training csv` | 同算法按 episode 统计 | 中等可比 |
| `comparison/comparison_scan_per_volt_drop.png` | 横向归一化结果图 | 训练轮次 | `episode` | 按电量归一化扫描产出(Cell/V) | `avg_scan_cells_per_volt_drop` | `training csv` | 同算法按 episode 统计 | 中等可比 |

## 3. 为什么 Stage02 横向图大多使用 Episode 轴

当前 `stage02` 结果图主要服务于“阶段结果比较”，其重点是比较：

- 每个 episode 的最终扫描结果
- 每个 episode 的最终熵值
- 每个 episode 经过时间、电量归一化后的平均产出

这些量本质上都是“单轮摘要量”，因此最自然的横轴是 `episode`。

需要注意的是：

- `episode` 轴更适合表达“每轮训练结果怎么变化”
- 它不是“绝对等物理预算”的保证
- 因此在论文中必须配合“强可比 / 弱可比”说明一起解释

## 4. 为什么 Stage02 要引入按时间、按电量归一化指标

### 4.1 步长不同带来的问题

当前两种算法的“单步”并不是完全同义：

- `纯 DQN` 的有效物理步长约为 `1.5s / step`
- `DDPG+APF` 在当前 `stage02` 实验口径下约为 `5.0s / step`

因此：

- `Cell/Step` 会天然偏向步长更长、单步控制更重的算法
- 如果直接拿 `Cell/Step` 做最终结论，会放大步长差异带来的偏差

### 4.2 时间归一化的意义

`Cell/s` 的引入，是为了回答：

- 单位时间内，谁扫得更多

这比 `Cell/Step` 更接近真实系统效率。

### 4.3 电量归一化的意义

`Cell/V` 的引入，是为了回答：

- 单位电压下降量下，谁扫得更多

这比单纯看最终电压或单纯看步数更适合作为“省电性”的近似比较指标。

## 5. 为什么 Reward 与 Cell/Step 只能弱可比

### 5.1 Reward

`Reward` 受以下因素影响：

- 奖励结构设计
- 控制方式
- 终止机制
- 状态空间与动作空间

因此在 `DDPG+APF` 与 `纯 DQN` 的横向比较中：

- Reward 可以解释训练趋势
- 不能单独承担最终性能判断

### 5.2 Cell/Step

`Cell/Step` 受以下因素影响：

- 单步物理时间
- 动作持续时长
- 控制策略是否偏向大步推进还是细粒度控制

因此：

- 它适合解释“动作效率”
- 不适合单独解释“谁最终更强”

## 6. Stage02 当前最合理的论文口径

建议在论文中采用如下判断层级：

### 第一层：主结论

使用强可比指标：

- 最终全局扫描率
- 最终全局平均熵

### 第二层：补充效率解释

使用中等可比指标：

- 按时间归一化扫描产出
- 按电量归一化扫描产出

### 第三层：过程解释

使用弱可比指标：

- Reward
- Cell/Step

## 7. 当前 Stage02 的方法学边界

当前 `stage02` 已经足以支持“阶段性对比分析”，但仍不是严格意义上的完全同构 benchmark。主要原因在于：

- `DDPG+APF` 与 `纯 DQN` 的控制结构不同
- 奖励函数细节不同
- 单步动作语义不同

因此，最适合的论文表述应为：

> 当前 `stage02` 对比可用于说明两种方法在现有工程口径下的阶段性结果差异，其中最终扫描率、最终熵值以及归一化扫描产出可作为主要比较依据；而 Reward 与 Cell/Step 仅用于解释训练过程与动作风格差异。
