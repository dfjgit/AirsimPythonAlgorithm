# DQN Stage02 单算法分析说明

## 1. 本目录的目标

本目录用于分析 `纯 DQN` 在 `stage02_finetune` 中的单算法训练表现。

对应图和报告包括：

- [episode_reward.png](/D:/Work/Python%20Project/airsim-python-algorithm/analysis_results/stage02_analysis_suite/dqn_stage02/episode_reward.png)
- [episode_length.png](/D:/Work/Python%20Project/airsim-python-algorithm/analysis_results/stage02_analysis_suite/dqn_stage02/episode_length.png)
- [collision_stability.png](/D:/Work/Python%20Project/airsim-python-algorithm/analysis_results/stage02_analysis_suite/dqn_stage02/collision_stability.png)
- [global_scan_ratio.png](/D:/Work/Python%20Project/airsim-python-algorithm/analysis_results/stage02_analysis_suite/dqn_stage02/global_scan_ratio.png)
- [global_avg_entropy.png](/D:/Work/Python%20Project/airsim-python-algorithm/analysis_results/stage02_analysis_suite/dqn_stage02/global_avg_entropy.png)
- [scan_efficiency.png](/D:/Work/Python%20Project/airsim-python-algorithm/analysis_results/stage02_analysis_suite/dqn_stage02/scan_efficiency.png)
- [trajectories_xz.png](/D:/Work/Python%20Project/airsim-python-algorithm/analysis_results/stage02_analysis_suite/dqn_stage02/trajectories_xz.png)
- [summary_report.md](/D:/Work/Python%20Project/airsim-python-algorithm/analysis_results/stage02_analysis_suite/dqn_stage02/summary_report.md)
- [summary_report.csv](/D:/Work/Python%20Project/airsim-python-algorithm/analysis_results/stage02_analysis_suite/dqn_stage02/summary_report.csv)

## 2. 数据来源

本目录主要基于以下 `DQN stage02` 数据：

- 训练汇总：
  [dqn_training_pure_dqn_20260330_005101_stage02_20260402_005952.csv](/D:/Work/Python%20Project/airsim-python-algorithm/multirotor/DQN_Movement/logs/dqn_scan_data/dqn_training_pure_dqn_20260330_005101_stage02_20260402_005952.csv)
- 扫描明细：
  [scan_data_pure_dqn_20260330_005101_stage02_20260402_005952.csv](/D:/Work/Python%20Project/airsim-python-algorithm/multirotor/DQN_Movement/logs/dqn_scan_data/scan_data_pure_dqn_20260330_005101_stage02_20260402_005952.csv)
- 终止原因辅助参考：
  [reset_trace_20260402_005952.jsonl](/D:/Work/Python%20Project/airsim-python-algorithm/multirotor/DQN_Movement/logs/movement_dqn_airsim/reset_diagnostics/reset_trace_20260402_005952.jsonl)

## 3. 各图横轴、纵轴与用途

| 图名 | 横轴（中文） | 横轴字段 | 纵轴（中文） | 纵轴字段 | 数据来源 |
|---|---|---|---|---|---|
| `episode_reward.png` | 训练轮次 | `episode` | 单轮累计奖励 | `reward` | `training csv` |
| `episode_length.png` | 训练轮次 | `episode` | 单轮步数长度 | `length` | `training csv` |
| `collision_stability.png` | 训练轮次 | `episode` | 碰撞终止占比(%) | `collision_rate` | `training csv` |
| `global_scan_ratio.png` | 训练轮次 | `episode` | 最终全局扫描率(%) | `final_global_scan_ratio` | `training csv` |
| `global_avg_entropy.png` | 训练轮次 | `episode` | 最终全局平均熵 | `final_global_avg_entropy` | `training csv` |
| `scan_efficiency.png` | 训练轮次 | `episode` | 扫描效率(Cell/Step) | `scan_efficiency` | `training csv` |
| `trajectories_xz.png` | X坐标 | `UAVx_x` | Z坐标 | `UAVx_z` | `scan_data csv` |

## 4. 每张图分别解释什么

### 4.1 `episode_reward.png`

用于观察：

- `stage02` 追训后奖励是否继续上升
- 是否已经摆脱 `stage01` 的高波动前期
- 是否开始进入平台区

### 4.2 `episode_length.png`

用于观察：

- 追训后单轮能否维持更长时间
- 是否仍存在大量早停
- OOR 与超时终止是否成为主要模式

### 4.3 `collision_stability.png`

用于观察：

- 追训后是否越来越少因碰撞终止
- 是否已从高碰撞不稳定阶段进入低碰撞稳定阶段

### 4.4 `global_scan_ratio.png`

用于观察：

- 追训后最终覆盖率是否继续提升
- 是否已经稳定维持在较高水平

### 4.5 `global_avg_entropy.png`

用于观察：

- DQN 是否在追训后进一步降低环境不确定性
- 覆盖率增长是否伴随有效的信息增益

### 4.6 `scan_efficiency.png`

用于观察：

- 单位动作层面的扫描产出变化
- DQN 是否出现“总量更高但单步效率不再继续增长”的平台迹象

### 4.7 `trajectories_xz.png`

用于观察：

- 空间覆盖范围
- 是否出现更稳定、更连续的轨迹组织
- 是否仍存在局部区域反复徘徊

## 5. 当前 Stage02 对 DQN 的阶段判断

根据当前汇总结果：

- 平均奖励约 `11945.97`
- 平均步数长度约 `195.08`
- 平均最终扫描率约 `20.03%`
- 最佳最终扫描率约 `23.12%`
- 平均最终熵约 `75.22`
- 平均扫描效率约 `1.6618 Cell/Step`

当前更合理的阶段判断是：

- `纯 DQN` 在 `stage02` 中已经形成较稳定的高覆盖表现
- 相比 `stage01`，结果指标明显更强
- 但后 20 轮相对中段已有一定平台波动，说明当前阶段可能已接近阶段上限附近

## 6. 当前数据口径的一个保留说明

当前 `DQN stage02` 的主录入链已经恢复正常：

- `training csv`
- `scan_data csv`
- `reset_trace`

三者在主结论层面是一致可用的。

目前仅保留一个小问题：

- `scan_data` 每轮最后一帧的 `reset_reason` 仍为空

但：

- `training csv` 中的 `reset_reason`
- `reset_trace` 中的终止原因

是正常的，因此不会影响本目录的主分析结论。

## 7. 该目录能支撑什么，不能支撑什么

### 可以支撑的

- `纯 DQN stage02` 已经是可用的正式阶段数据
- DQN 在结果层面已经达到较高覆盖水平
- DQN 当前阶段相较 `stage01` 明显增强

### 不能单独支撑的

- `纯 DQN` 已经绝对达到最终最优上限
- 不需要再做任何统一评测或更高阶段验证

因为最终方法结论仍应结合横向对比与后续实验设计。

## 8. 在论文中的适合位置

本目录最适合用于：

- `纯 DQN` 单算法追训阶段结果分析
- `纯 DQN` 由学习增强带来的覆盖能力提升说明
- 为横向对比章节提供单算法结果背景
