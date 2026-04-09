# Stage02 横向对比分析说明

## 1. 本目录的目标

本目录用于保存 `DDPG+APF stage02` 与 `纯 DQN stage02` 的横向对比图。

对应图包括：

- [comparison_training_reward.png](/D:/Work/Python%20Project/airsim-python-algorithm/analysis_results/stage02_analysis_suite/comparison/comparison_training_reward.png)
- [comparison_training_scan_efficiency.png](/D:/Work/Python%20Project/airsim-python-algorithm/analysis_results/stage02_analysis_suite/comparison/comparison_training_scan_efficiency.png)
- [comparison_training_collision_rate.png](/D:/Work/Python%20Project/airsim-python-algorithm/analysis_results/stage02_analysis_suite/comparison/comparison_training_collision_rate.png)
- [comparison_scan_scan_ratio.png](/D:/Work/Python%20Project/airsim-python-algorithm/analysis_results/stage02_analysis_suite/comparison/comparison_scan_scan_ratio.png)
- [comparison_scan_global_avg_entropy.png](/D:/Work/Python%20Project/airsim-python-algorithm/analysis_results/stage02_analysis_suite/comparison/comparison_scan_global_avg_entropy.png)
- [comparison_scan_per_second.png](/D:/Work/Python%20Project/airsim-python-algorithm/analysis_results/stage02_analysis_suite/comparison/comparison_scan_per_second.png)
- [comparison_scan_per_volt_drop.png](/D:/Work/Python%20Project/airsim-python-algorithm/analysis_results/stage02_analysis_suite/comparison/comparison_scan_per_volt_drop.png)

这些图不是简单用来回答“谁高谁低”，而是分别从：

- 训练过程
- 结果覆盖
- 信息不确定性下降
- 时间归一化效率
- 电量归一化效率

五个角度解释当前 `stage02` 的方法差异。

## 2. 数据来源

### 2.1 DDPG stage02

- 训练汇总：
  [ddpg_training_ddpg_apf_20260326_234951_stage02_20260331_003640.csv](/D:/Work/Python%20Project/airsim-python-algorithm/multirotor/DDPG_Weight/airsim_training_logs/ddpg_training_ddpg_apf_20260326_234951_stage02_20260331_003640.csv)
- 扫描明细：
  [scan_data_ddpg_apf_20260326_234951_stage02_20260331_003640.csv](/D:/Work/Python%20Project/airsim-python-algorithm/multirotor/DDPG_Weight/airsim_training_logs/scan_data_ddpg_apf_20260326_234951_stage02_20260331_003640.csv)

### 2.2 DQN stage02

- 训练汇总：
  [dqn_training_pure_dqn_20260330_005101_stage02_20260402_005952.csv](/D:/Work/Python%20Project/airsim-python-algorithm/multirotor/DQN_Movement/logs/dqn_scan_data/dqn_training_pure_dqn_20260330_005101_stage02_20260402_005952.csv)
- 扫描明细：
  [scan_data_pure_dqn_20260330_005101_stage02_20260402_005952.csv](/D:/Work/Python%20Project/airsim-python-algorithm/multirotor/DQN_Movement/logs/dqn_scan_data/scan_data_pure_dqn_20260330_005101_stage02_20260402_005952.csv)

### 2.3 汇总报告

- [stage02_algorithm_comparison_report.csv](/D:/Work/Python%20Project/airsim-python-algorithm/analysis_results/stage02_analysis_suite/comparison/stage02_algorithm_comparison_report.csv)

## 3. 七张图的横轴、纵轴与用途

### 3.1 `comparison_training_reward.png`

- 横轴：训练轮次（Episode）
- 纵轴：累计奖励（Reward）
- 数据来源：`training csv`
- 可比性：弱可比

用途：

- 观察训练趋势
- 判断是否出现明显平台期或回落
- 解释不同算法在训练阶段的“学习风格”

局限：

- DDPG 与 DQN 的奖励结构不同
- 不能只凭该图下最终性能结论

### 3.2 `comparison_training_scan_efficiency.png`

- 横轴：训练轮次（Episode）
- 纵轴：扫描效率（Cell/Step）
- 数据来源：`training csv`
- 可比性：弱可比

用途：

- 观察单位步长下的扫描产出
- 解释两种算法的动作风格差异

局限：

- `DDPG+APF` 与 `纯 DQN` 的“单步”并不具有完全相同的物理时间含义
- 因此 `Cell/Step` 更适合解释动作效率，不适合单独作为最终胜负指标

### 3.3 `comparison_training_collision_rate.png`

- 横轴：训练轮次（Episode）
- 纵轴：碰撞终止占比（%）
- 数据来源：`training csv`
- 可比性：中等可比

用途：

- 比较两种算法在训练过程中的碰撞稳定性
- 观察谁更早进入低碰撞区间

### 3.4 `comparison_scan_scan_ratio.png`

- 横轴：训练轮次（Episode）
- 纵轴：最终全局扫描率（%）
- 数据来源：`training csv`
- 可比性：强可比

用途：

- 反映最终覆盖能力
- 是当前论文结果分析中的核心图之一

### 3.5 `comparison_scan_global_avg_entropy.png`

- 横轴：训练轮次（Episode）
- 纵轴：最终全局平均熵
- 数据来源：`training csv`
- 可比性：强可比

用途：

- 反映算法降低环境不确定性的能力
- 熵值越低通常表示扫描结果越充分

### 3.6 `comparison_scan_per_second.png`

- 横轴：训练轮次（Episode）
- 纵轴：按时间归一化扫描产出（Cell/s）
- 数据来源：`training csv`
- 可比性：中等可比

用途：

- 解释“单位时间内谁扫得更多”
- 用于缓解步长差异导致的 `Cell/Step` 偏差

说明：

- 当前按 `seconds_per_step * episode_length` 近似得到单轮物理时长
- 因此它比 `Cell/Step` 更公平，但仍然是近似归一化

### 3.7 `comparison_scan_per_volt_drop.png`

- 横轴：训练轮次（Episode）
- 纵轴：按电量下降归一化扫描产出（Cell/V）
- 数据来源：`training csv`
- 可比性：中等可比

用途：

- 解释“单位电压下降带来的扫描产出”
- 用于回答“谁更省电、更节能”

说明：

- 当前使用 `4.2V - terminal_battery_voltage` 近似定义每轮电压下降量
- 因此这是电量效率的近似指标，不等于真实焦耳级能耗

## 4. 强可比 / 中等可比 / 弱可比

### 4.1 强可比指标

- `comparison_scan_scan_ratio.png`
- `comparison_scan_global_avg_entropy.png`

理由：

- 直接反映覆盖效果与不确定性降低效果
- 与具体奖励函数结构关系较弱
- 更接近任务结果层面的最终表现

### 4.2 中等可比指标

- `comparison_training_collision_rate.png`
- `comparison_scan_per_second.png`
- `comparison_scan_per_volt_drop.png`

理由：

- 通过时间、电量归一化，缓解了不同步长造成的偏差
- 但仍受控制强度、电池模型、终止机制等因素影响

### 4.3 弱可比指标

- `comparison_training_reward.png`
- `comparison_training_scan_efficiency.png`

理由：

- `Reward` 依赖奖励设计
- `Cell/Step` 依赖步长和动作语义
- 更适合做过程解释，而不是最终优劣结论

## 5. 当前 Stage02 的主要结果

### 5.1 DDPG+APF stage02

- 平均奖励：`2339.49`
- 平均最终扫描率：`11.06%`
- 最佳最终扫描率：`15.25%`
- 平均最终熵值：`77.02`
- 最低最终熵值：`73.16`
- 平均按时间归一化扫描产出：`0.6028 Cell/s`
- 平均按电量归一化扫描产出：`193.52 Cell/V`

### 5.2 纯 DQN stage02

- 平均奖励：`11945.97`
- 平均最终扫描率：`20.03%`
- 最佳最终扫描率：`23.12%`
- 平均最终熵值：`75.22`
- 最低最终熵值：`71.96`
- 平均按时间归一化扫描产出：`1.1083 Cell/s`
- 平均按电量归一化扫描产出：`366.63 Cell/V`

## 6. 当前最稳妥的论文结论

当前最稳妥的表述是：

- `DDPG+APF stage02` 的优势在于训练稳定、终止行为可预测、较早进入平台期。
- `纯 DQN stage02` 的优势在于最终覆盖率更高、最终熵值更低，并且在按时间与按电量归一化后仍表现出更高的扫描产出。
- 因此，在当前 `stage02` 条件下，若论文主结论关注“结果效果”和“归一化任务产出”，则 `纯 DQN` 的阶段结果更优；若关注“步级动作效率”或训练稳定性，则 `DDPG+APF` 仍体现出其解析先验带来的结构优势。

## 7. 使用建议

建议在论文正文中采用以下写法：

1. 先用强可比图给出主结论  
   即扫描率与熵值。
2. 再用中等可比图做补充说明  
   即时间归一化与电量归一化扫描产出。
3. 最后用弱可比图解释训练风格差异  
   即 `Reward` 与 `Cell/Step`。

这样可以避免把弱可比指标误写成最终性能结论。
