# DDPG Stage02 单算法分析说明

## 1. 本目录的目标

本目录用于分析 `DDPG+APF` 在 `stage02_finetune` 中的单算法训练表现。

对应图和报告包括：

- [episode_reward.png](/D:/Work/Python%20Project/airsim-python-algorithm/analysis_results/stage02_analysis_suite/ddpg_stage02/episode_reward.png)
- [episode_length.png](/D:/Work/Python%20Project/airsim-python-algorithm/analysis_results/stage02_analysis_suite/ddpg_stage02/episode_length.png)
- [global_scan_ratio.png](/D:/Work/Python%20Project/airsim-python-algorithm/analysis_results/stage02_analysis_suite/ddpg_stage02/global_scan_ratio.png)
- [global_avg_entropy.png](/D:/Work/Python%20Project/airsim-python-algorithm/analysis_results/stage02_analysis_suite/ddpg_stage02/global_avg_entropy.png)
- [scan_efficiency.png](/D:/Work/Python%20Project/airsim-python-algorithm/analysis_results/stage02_analysis_suite/ddpg_stage02/scan_efficiency.png)
- [trajectories_xz.png](/D:/Work/Python%20Project/airsim-python-algorithm/analysis_results/stage02_analysis_suite/ddpg_stage02/trajectories_xz.png)
- [summary_report.md](/D:/Work/Python%20Project/airsim-python-algorithm/analysis_results/stage02_analysis_suite/ddpg_stage02/summary_report.md)
- [summary_report.csv](/D:/Work/Python%20Project/airsim-python-algorithm/analysis_results/stage02_analysis_suite/ddpg_stage02/summary_report.csv)

## 2. 数据来源

本目录主要基于以下两份 `DDPG stage02` 数据：

- 训练汇总：
  [ddpg_training_ddpg_apf_20260326_234951_stage02_20260331_003640.csv](/D:/Work/Python%20Project/airsim-python-algorithm/multirotor/DDPG_Weight/airsim_training_logs/ddpg_training_ddpg_apf_20260326_234951_stage02_20260331_003640.csv)
- 扫描明细：
  [scan_data_ddpg_apf_20260326_234951_stage02_20260331_003640.csv](/D:/Work/Python%20Project/airsim-python-algorithm/multirotor/DDPG_Weight/airsim_training_logs/scan_data_ddpg_apf_20260326_234951_stage02_20260331_003640.csv)

## 3. 各图横轴、纵轴与用途

| 图名 | 横轴（中文） | 横轴字段 | 纵轴（中文） | 纵轴字段 | 数据来源 |
|---|---|---|---|---|---|
| `episode_reward.png` | 训练轮次 | `episode` | 单轮累计奖励 | `reward` | `training csv` |
| `episode_length.png` | 训练轮次 | `episode` | 单轮步数长度 | `length` | `training csv` |
| `global_scan_ratio.png` | 训练轮次 | `episode` | 最终全局扫描率(%) | `final_global_scan_ratio` | `training csv` |
| `global_avg_entropy.png` | 训练轮次 | `episode` | 最终全局平均熵 | `final_global_avg_entropy` | `training csv` |
| `scan_efficiency.png` | 训练轮次 | `episode` | 扫描效率(Cell/Step) | `scan_efficiency` | `training csv` |
| `trajectories_xz.png` | X坐标 | `UAVx_x` | Z坐标 | `UAVx_z` | `scan_data csv` |

## 4. 每张图分别解释什么

### 4.1 `episode_reward.png`

用于观察：

- 追训后奖励是否继续提高
- 是否已经进入明显平台期

### 4.2 `episode_length.png`

用于观察：

- 是否仍存在大量早停
- 是否大多数回合都能稳定运行到较长时长

### 4.3 `global_scan_ratio.png`

用于观察：

- `DDPG+APF` 追训后的最终覆盖率是否继续提升
- 是否已经在固定口径下进入平台区

### 4.4 `global_avg_entropy.png`

用于观察：

- 追训后环境不确定性是否继续下降
- 覆盖提升是否伴随着真实信息增益

### 4.5 `scan_efficiency.png`

用于观察：

- 单位动作层面的扫描产出是否稳定
- 是否出现“仍然很稳，但单位步增益不再增长”的平台期特征

### 4.6 `trajectories_xz.png`

用于观察：

- 轨迹覆盖范围
- 是否存在明显的空间偏置与局部打转

## 5. 当前 Stage02 对 DDPG+APF 的阶段判断

根据当前汇总结果：

- 平均奖励约 `2339.49`
- 平均步数长度约 `59.47`
- 平均最终扫描率约 `11.06%`
- 最佳最终扫描率约 `15.25%`
- 平均最终熵约 `77.02`
- 平均扫描效率约 `3.0141 Cell/Step`

当前更合理的阶段判断是：

- `DDPG+APF` 已经表现出良好的训练稳定性
- 当前阶段不再是“高碰撞、高早停”的异常状态
- 但在结果层面已经明显接近平台区，继续追训的边际收益相对有限

## 6. 该目录能支撑什么，不能支撑什么

### 可以支撑的

- `DDPG+APF stage02` 已经稳定
- 其结果口径可作为横向对比基线
- 其当前阶段更像平台维持，而非持续快速上升

### 不能单独支撑的

- `DDPG+APF` 最终一定优于 `纯 DQN`
- `DDPG+APF` 已经是本任务中的终极最优方法

因为这些判断必须结合横向对比结果。

## 7. 在论文中的适合位置

本目录最适合用于：

- `DDPG+APF` 单算法追训阶段结果分析
- `DDPG+APF` 稳定性与平台期说明
- 为横向对比章节提供单算法背景支撑
