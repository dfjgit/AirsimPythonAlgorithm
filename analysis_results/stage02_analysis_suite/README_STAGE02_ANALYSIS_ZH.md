# Stage02 实验分析总说明

## 1. 文件用途

本目录用于保存当前 `stage02_finetune` 阶段的三套实验分析结果：

1. `DDPG+APF` 单算法分析  
2. `纯 DQN` 单算法分析  
3. `DDPG+APF vs 纯 DQN` 横向对比分析

对应目录如下：

- [ddpg_stage02](/D:/Work/Python%20Project/airsim-python-algorithm/analysis_results/stage02_analysis_suite/ddpg_stage02)
- [dqn_stage02](/D:/Work/Python%20Project/airsim-python-algorithm/analysis_results/stage02_analysis_suite/dqn_stage02)
- [comparison](/D:/Work/Python%20Project/airsim-python-algorithm/analysis_results/stage02_analysis_suite/comparison)

本说明文件面向“论文级实验分析”使用，重点回答以下问题：

- `stage02` 相比 `stage01` 回答的是哪一类实验问题
- 三套图分别适合支撑什么结论
- 哪些指标属于强可比指标，哪些只能作为弱可比或补充指标
- 当前 `stage02` 能支持什么结论，不能支持什么结论

## 2. 推荐阅读顺序

建议按以下顺序阅读：

1. 先看单算法分析  
   先确认 `DDPG+APF` 和 `纯 DQN` 在各自 `stage02` 中是否稳定、是否进入平台期、是否仍有继续提升迹象。
2. 再看横向对比分析  
   在确认两边单算法数据可信后，再讨论 `stage02` 阶段的算法差异。
3. 最后结合方法学说明  
   避免将 `Reward`、`Cell/Step` 这类弱可比指标误用为最终性能结论。

## 3. 本套分析的统一数据来源

本套 `stage02` 分析主要使用两类原始文件：

1. `training csv`  
   用于记录每个 episode 的训练汇总结果，例如：
   - `reward`
   - `length`
   - `scan_efficiency`
   - `train_timestep_end`
   - `reset_reason`
   - `final_global_scan_ratio`
   - `min_global_avg_entropy`

2. `scan_data csv`  
   用于记录逐步采集的扫描过程明细，例如：
   - `episode`
   - `step`
   - `elapsed_time`
   - `global_scan_ratio`
   - `global_avg_entropy`
   - 多机坐标
   - 电量与越界时长

此外，`DQN stage02` 的终止原因解释还结合了：

- [reset_trace_20260402_005952.jsonl](/D:/Work/Python%20Project/airsim-python-algorithm/multirotor/DQN_Movement/logs/movement_dqn_airsim/reset_diagnostics/reset_trace_20260402_005952.jsonl)

## 4. 三套分析分别回答什么问题

### 4.1 DDPG+APF 单算法分析

对应目录：

- [ddpg_stage02](/D:/Work/Python%20Project/airsim-python-algorithm/analysis_results/stage02_analysis_suite/ddpg_stage02)

主要回答：

- DDPG+APF 在追训后是否继续提升
- DDPG+APF 是否已经进入平台期
- 其扫描率、熵值、轨迹分布是否仍然具有工程可用性

### 4.2 纯 DQN 单算法分析

对应目录：

- [dqn_stage02](/D:/Work/Python%20Project/airsim-python-algorithm/analysis_results/stage02_analysis_suite/dqn_stage02)

主要回答：

- DQN 在 `stage02` 是否延续了 `stage01` 的后期上升趋势
- DQN 是否开始进入平台期
- DQN 是否已经在结果层面超过当前 DDPG+APF 基线

### 4.3 横向对比分析

对应目录：

- [comparison](/D:/Work/Python%20Project/airsim-python-algorithm/analysis_results/stage02_analysis_suite/comparison)

主要回答：

- 两种算法在 `stage02` 阶段谁的覆盖结果更好
- 两种算法在降低环境不确定性方面谁更强
- 两种算法在按时间归一化、按电量归一化后谁更高效
- 哪些差异属于“控制风格差异”，哪些差异属于“结果差异”

## 5. 训练过程分析 与 结果表现分析

### 5.1 训练过程分析

训练过程分析关注“算法是怎么学的”，主要包括：

- `episode_reward.png`
- `episode_length.png`
- `comparison_training_reward.png`
- `comparison_training_scan_efficiency.png`

它们主要用于分析：

- 收敛趋势
- 训练稳定性
- 平台期与回落
- 单位动作层面的效率差异

### 5.2 结果表现分析

结果表现分析关注“算法最终做成了什么”，主要包括：

- `global_scan_ratio.png`
- `global_avg_entropy.png`
- `trajectories_xz.png`
- `comparison_scan_scan_ratio.png`
- `comparison_scan_global_avg_entropy.png`
- `comparison_scan_per_second.png`
- `comparison_scan_per_volt_drop.png`

它们主要用于分析：

- 最终覆盖能力
- 环境不确定性降低能力
- 按时间归一化的任务产出
- 按电量归一化的任务产出

## 6. 当前 Stage02 的可比性原则

### 6.1 强可比指标

以下指标更适合作为论文主结论依据：

- `comparison_scan_scan_ratio.png`
- `comparison_scan_global_avg_entropy.png`

原因是：

- 它们直接反映扫描覆盖结果与不确定性降低效果
- 对控制结构差异和奖励差异不敏感
- 更接近任务本身的最终表现

### 6.2 中等可比指标

以下指标适合做补充性论据：

- `comparison_scan_per_second.png`
- `comparison_scan_per_volt_drop.png`

原因是：

- 它们通过时间或电量归一化，减弱了步长差异的影响
- 但仍会受到动作强度、电池模型、终止方式差异的影响

### 6.3 弱可比指标

以下指标不适合作为单独的最终优劣结论：

- `comparison_training_reward.png`
- `comparison_training_scan_efficiency.png`

原因是：

- `Reward` 受奖励结构影响明显
- `Cell/Step` 受单步物理时间与动作语义差异影响明显
- 因此更适合解释“训练风格”和“动作效率”，不适合独立下最终结论

## 7. 当前 Stage02 的阶段性结论

基于当前 `stage02` 数据，可以较稳地给出以下判断：

### 7.1 DDPG+APF stage02

- 平均最终扫描率约 `11.06%`
- 最佳最终扫描率约 `15.25%`
- 平均最终熵值约 `77.02`
- 平均按时间归一化扫描产出约 `0.6028 Cell/s`
- 平均按电量归一化扫描产出约 `193.52 Cell/V`
- 主要终止原因以 `达到时长上限` 为主，说明训练过程稳定，但提升空间有限

### 7.2 纯 DQN stage02

- 平均最终扫描率约 `20.03%`
- 最佳最终扫描率约 `23.12%`
- 平均最终熵值约 `75.22`
- 平均按时间归一化扫描产出约 `1.1083 Cell/s`
- 平均按电量归一化扫描产出约 `366.63 Cell/V`
- 说明 DQN 在当前 `stage02` 中，已经在结果层面表现出更强的覆盖能力与更好的时间/能耗归一化产出

### 7.3 当前最稳妥的横向判断

当前最稳妥的结论不是“某算法在所有维度都绝对更优”，而是：

- `DDPG+APF` 的特点是更稳、更早进入平台期
- `纯 DQN` 的特点是结果指标更高，但训练过程中的步级效率解释要谨慎

## 8. 当前数据能做什么，不能做什么

### 8.1 可以做的

- 用作 `stage02` 的正式阶段性对比分析
- 支撑“DQN 在结果层面已超过当前 DDPG+APF stage02”这一阶段结论
- 支撑论文中的“追训后阶段结果比较”部分

### 8.2 不能直接做的

- 直接宣称两者是完全同构、完全公平的终局 benchmark
- 仅凭 `Reward` 或 `Cell/Step` 给出最终胜负结论
- 把当前 `stage02` 视为冻结模型统一评测的替代品

## 9. 当前数据口径的一个保留说明

当前 `DQN stage02` 的主录入链已经恢复正常，训练汇总与扫描明细可对齐。  
但在 [scan_data_pure_dqn_20260330_005101_stage02_20260402_005952.csv](/D:/Work/Python%20Project/airsim-python-algorithm/multirotor/DQN_Movement/logs/dqn_scan_data/scan_data_pure_dqn_20260330_005101_stage02_20260402_005952.csv) 中，每轮最后一帧的 `reset_reason` 仍为空。  
不过：

- [dqn_training_pure_dqn_20260330_005101_stage02_20260402_005952.csv](/D:/Work/Python%20Project/airsim-python-algorithm/multirotor/DQN_Movement/logs/dqn_scan_data/dqn_training_pure_dqn_20260330_005101_stage02_20260402_005952.csv)
- [reset_trace_20260402_005952.jsonl](/D:/Work/Python%20Project/airsim-python-algorithm/multirotor/DQN_Movement/logs/movement_dqn_airsim/reset_diagnostics/reset_trace_20260402_005952.jsonl)

中的终止原因是正常的，因此不影响本套 `stage02` 主结论。

## 10. 后续推荐流程

建议按以下逻辑继续推进：

1. 使用本套 `stage02` 图和说明，完成“追训阶段结果分析”
2. 在论文正文中，明确把强可比与弱可比指标分开表述
3. 若需要更强方法学说服力，再补“冻结模型统一评测”
4. 若需要进一步提升 DQN，后续可规划新的扩展阶段，而不是再回到 `stage01` 口径
