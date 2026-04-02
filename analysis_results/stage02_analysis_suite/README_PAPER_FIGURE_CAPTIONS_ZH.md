# Stage02 论文图注模板（中文）

## 1. 文件用途

本文档提供 `stage02` 三套分析图的论文图注模板，可直接用于：

- 图注草稿
- 结果分析小节初稿
- 论文中“图 X 展示了什么”的标准表述

建议使用方式：

1. 先阅读 [README_STAGE02_ANALYSIS_ZH.md](/D:/Work/Python%20Project/airsim-python-algorithm/analysis_results/stage02_analysis_suite/README_STAGE02_ANALYSIS_ZH.md)
2. 再结合 [README_EXPERIMENT_METHOD_ZH.md](/D:/Work/Python%20Project/airsim-python-algorithm/analysis_results/stage02_analysis_suite/README_EXPERIMENT_METHOD_ZH.md)
3. 最后从本文复制图注模板并按章节编号轻微改写

## 2. DDPG+APF 单算法图注模板

### `episode_reward.png`

> 图 X 展示了 `DDPG+APF` 在 `stage02_finetune` 阶段中单轮累计奖励随训练轮次的变化趋势。横轴为训练轮次（Episode），纵轴为单轮累计奖励（Episode Reward）。该图用于观察 `DDPG+APF` 在追训阶段是否继续获得明显收益，或是否已经进入平台期。

### `episode_length.png`

> 图 X 展示了 `DDPG+APF` 在 `stage02_finetune` 阶段中单轮步数长度随训练轮次的变化情况。横轴为训练轮次，纵轴为单轮步数长度。该图用于观察追训阶段是否仍存在频繁早停，或是否已稳定维持较长单轮运行。

### `global_scan_ratio.png`

> 图 X 展示了 `DDPG+APF` 在 `stage02_finetune` 阶段中每轮最终全局扫描率的变化情况。横轴为训练轮次，纵轴为最终全局扫描率（%）。该图反映了 `DDPG+APF` 在追训阶段的覆盖结果表现，是判断其是否继续提升的重要结果指标之一。

### `global_avg_entropy.png`

> 图 X 展示了 `DDPG+APF` 在 `stage02_finetune` 阶段中每轮最终全局平均熵的变化情况。横轴为训练轮次，纵轴为最终全局平均熵。该图用于评价算法在追训阶段降低环境不确定性的能力，熵值越低通常表示扫描结果越充分。

### `scan_efficiency.png`

> 图 X 展示了 `DDPG+APF` 在 `stage02_finetune` 阶段中的扫描效率变化。横轴为训练轮次，纵轴为扫描效率（Cell/Step），即每一步平均带来的扫描格数增量。该图用于解释 `DDPG+APF` 在单位动作层面的效率特征，但不宜单独作为最终性能判断依据。

### `trajectories_xz.png`

> 图 X 展示了 `DDPG+APF` 在 `stage02_finetune` 阶段中的顶视轨迹分布。横轴为 X 坐标，纵轴为 Z 坐标。该图用于分析无人机群在水平平面内的空间覆盖范围与轨迹组织性。

## 3. 纯 DQN 单算法图注模板

### `episode_reward.png`

> 图 X 展示了 `纯 DQN` 在 `stage02_finetune` 阶段中单轮累计奖励随训练轮次的变化情况。横轴为训练轮次，纵轴为单轮累计奖励。该图用于观察 `纯 DQN` 在追训后是否延续了 `stage01` 后期的提升趋势，或是否已经进入平台期。

### `episode_length.png`

> 图 X 展示了 `纯 DQN` 在 `stage02_finetune` 阶段中单轮步数长度的变化情况。横轴为训练轮次，纵轴为单轮步数长度。该图用于判断 DQN 是否维持稳定运行，以及是否仍存在频繁早停或过多异常终止。

### `global_scan_ratio.png`

> 图 X 展示了 `纯 DQN` 在 `stage02_finetune` 阶段中每轮最终全局扫描率的变化情况。横轴为训练轮次，纵轴为最终全局扫描率（%）。该图直接反映了 `纯 DQN` 在追训阶段的最终覆盖能力，是本研究中最重要的结果指标之一。

### `global_avg_entropy.png`

> 图 X 展示了 `纯 DQN` 在 `stage02_finetune` 阶段中每轮最终全局平均熵的变化情况。横轴为训练轮次，纵轴为最终全局平均熵。该图用于评估 DQN 在追训后降低环境不确定性的能力，熵值越低通常表示环境状态被更充分地观测与扫描。

### `scan_efficiency.png`

> 图 X 展示了 `纯 DQN` 在 `stage02_finetune` 阶段中的扫描效率变化。横轴为训练轮次，纵轴为扫描效率（Cell/Step）。该图可用于解释 DQN 的动作效率变化趋势，但由于 `DDPG+APF` 与 `纯 DQN` 的单步语义不同，故该指标更适合作为补充解释，而非最终横向优劣判断依据。

### `trajectories_xz.png`

> 图 X 展示了 `纯 DQN` 在 `stage02_finetune` 阶段中的顶视轨迹分布。横轴为 X 坐标，纵轴为 Z 坐标。该图用于观察 DQN 轨迹在追训后是否形成更连续、更具覆盖组织性的空间分布。

## 4. Stage02 横向对比图注模板

### `comparison_training_reward.png`

> 图 X 对比了 `DDPG+APF` 与 `纯 DQN` 在 `stage02` 阶段的训练奖励变化趋势。横轴为训练轮次，纵轴为累计奖励。该图主要用于比较两种算法在追训阶段的训练趋势与稳定性，而不宜单独作为最终任务性能优劣的证据。

### `comparison_training_scan_efficiency.png`

> 图 X 对比了 `DDPG+APF` 与 `纯 DQN` 在 `stage02` 阶段的扫描效率变化趋势。横轴为训练轮次，纵轴为扫描效率（Cell/Step）。由于两种算法的单步物理时间和控制语义并不完全相同，因此该图更适合解释动作层面的效率差异，而不宜单独作为最终胜负指标。

### `comparison_scan_scan_ratio.png`

> 图 X 对比了 `DDPG+APF` 与 `纯 DQN` 在 `stage02` 阶段的最终全局扫描率变化趋势。横轴为训练轮次，纵轴为最终全局扫描率（%）。该图直接反映了两种算法在当前追训阶段的覆盖结果表现，属于本研究中的强可比指标之一。

### `comparison_scan_global_avg_entropy.png`

> 图 X 对比了 `DDPG+APF` 与 `纯 DQN` 在 `stage02` 阶段的最终全局平均熵变化趋势。横轴为训练轮次，纵轴为最终全局平均熵。该图用于比较两种算法降低环境不确定性的能力，熵值越低表示信息获取越充分，属于本研究中的强可比指标之一。

### `comparison_scan_per_second.png`

> 图 X 对比了 `DDPG+APF` 与 `纯 DQN` 在 `stage02` 阶段的按时间归一化扫描产出。横轴为训练轮次，纵轴为扫描产出（Cell/s）。该图用于缓解步长差异带来的偏差，比较两种算法在单位时间下的扫描效率，属于中等可比指标。

### `comparison_scan_per_volt_drop.png`

> 图 X 对比了 `DDPG+APF` 与 `纯 DQN` 在 `stage02` 阶段的按电量下降归一化扫描产出。横轴为训练轮次，纵轴为扫描产出（Cell/V）。该图用于比较两种算法在单位电量下降下的任务产出，是分析“省电性”与“能耗效率”的补充性指标。

## 5. 关于强可比 / 弱可比的论文表述模板

建议在正文或图注附近补充如下说明：

> 需要说明的是，由于 `DDPG+APF` 与 `纯 DQN` 在动作空间、控制结构与奖励设计上并不完全同构，故 `Reward` 与 `Cell/Step` 更适合用于解释训练过程和动作风格差异，而最终全局扫描率、最终全局平均熵以及归一化扫描产出更适合用于支撑横向性能结论。

## 6. 关于当前 Stage02 结论的论文式表述模板

建议写成：

> 从 `stage02` 阶段结果看，`纯 DQN` 在最终全局扫描率、最终全局平均熵以及按时间、按电量归一化扫描产出方面均优于当前 `DDPG+APF` 基线，说明其在结果层面已表现出更强的覆盖能力与更高的归一化任务产出；而 `DDPG+APF` 仍在训练稳定性与步级动作效率方面表现出解析先验带来的结构优势。
