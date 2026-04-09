# 碰撞稳定性分析补充设计

日期：2026-04-09

## 摘要

本设计用于在当前训练分析体系中补充“碰撞稳定性”分析能力，并同时覆盖两套口径：

- 完整口径离线分析
- 论文精简口径分析

本次补充的核心目标不是增加更多碰撞诊断细节，而是引入一个可以直接反映算法训练稳定性的主指标：**碰撞终止占比随训练轮次的变化趋势**。

在完整口径中，该指标将作为新的稳定性主图加入单算法分析链，并继续与现有的碰撞热点图、碰撞对象统计图共同组成“碰撞稳定性 + 碰撞诊断”组合。在论文精简口径中，则以尽量低的图数代价，为 `DDPG+APF`、`纯 DQN` 以及两者的横向对比各增加一张碰撞稳定性主图，用于支撑“训练过程稳定性”结论。

## 问题背景

当前项目已经有两套并行存在的分析口径：

### 1. 完整口径离线分析

完整口径由以下脚本主导：

- `multirotor/Algorithm/visualize_scan_csv.py`
- `multirotor/Algorithm/visualize_training_data.py`
- `multirotor/Algorithm/training_analyzer.py`

它已经具备较完整的碰撞诊断能力，例如：

- `collision_hotspots_xy.png`
- `collision_object_breakdown.png`
- `reset_reason_rolling_ratio.png`

但这些图在当前结构下仍然存在两个问题：

1. 没有一张图被明确定位为“碰撞稳定性主图”
2. 碰撞相关信息偏诊断，论文里不够直接支撑“算法稳定性是否改善”的主结论

### 2. 论文精简口径分析

论文精简口径当前主要以 `stage02_analysis_suite` 下的说明文件为准，例如：

- `analysis_results/stage02_analysis_suite/README_STAGE02_ANALYSIS_ZH.md`
- `analysis_results/stage02_analysis_suite/ddpg_stage02/README_SINGLE_ANALYSIS_ZH.md`
- `analysis_results/stage02_analysis_suite/dqn_stage02/README_SINGLE_ANALYSIS_ZH.md`
- `analysis_results/stage02_analysis_suite/comparison/README_COMPARISON_EXPERIMENT_ZH.md`

这套口径目前聚焦：

- reward
- episode length
- scan ratio
- entropy
- scan efficiency
- trajectory

虽然能够说明训练趋势和结果表现，但还缺一类能直接回答“训练是否更稳定、更少因碰撞失败”的主图。

## 目标

- 引入统一的“碰撞稳定性”主指标，用于观察训练过程中的稳定性变化
- 在完整口径单算法分析中新增一张稳定性主图
- 在论文精简口径单算法分析中，各新增一张碰撞稳定性图
- 在论文精简口径横向对比分析中，新增一张碰撞稳定性对比图
- 保持与现有碰撞热点图、碰撞对象图兼容，不替代它们的诊断价值
- 尽量复用已有训练 CSV 字段与现有 rolling 曲线风格，避免引入新的数据采集链要求

## 非目标

- 本次不新增新的运行时碰撞采集协议
- 本次不修改 `AlgorithmServer`、AirSim 碰撞检测流程或训练日志字段定义
- 本次不把碰撞穿透深度、碰撞次数、对象分布全部提升为论文精简口径主图
- 本次不试图把“碰撞稳定性”扩展成完整安全评测体系

## 推荐方案

推荐采用“**碰撞终止占比作为稳定性主指标，碰撞热点与对象分布继续作为诊断图**”的设计。

核心思想如下：

- 对每个 episode 计算一个二值碰撞终止标记
  - 因碰撞终止记为 `1`
  - 非碰撞终止记为 `0`
- 对该序列按 episode 做滚动平均，得到最近一段训练窗口内的碰撞终止占比
- 用该滚动占比曲线作为碰撞稳定性主图

相比直接使用碰撞热点或碰撞对象频数，这个指标更适合作为“稳定性”主论据，因为：

- 它直接对应“算法是否频繁因碰撞失败”
- 它能随训练过程连续观察变化趋势
- 它既适合单算法看稳定性演化，也适合横向对比不同算法的稳定性趋势

## 为什么符合当前架构

当前训练汇总 CSV 已经包含足够字段来支持该指标：

- `reset_reason`
- `collision_count`
- `collision_count_final`
- `collision_object_name`
- `collision_penetration_depth`
- `collision_position`

这意味着：

- 单算法完整口径可以直接基于 `RunData.episode_df` 派生碰撞终止标记
- 横向对比口径可以直接基于 `UnifiedTrainingAnalyzer` 已加载的 training CSV 派生统一指标

换句话说，当前缺的不是数据，而是：

- 统一的稳定性指标定义
- 新的绘图入口
- 文档对该指标的正式定位

## 指标定义

### 1. 主指标名称

建议统一命名为：

- 中文：碰撞稳定性
- 英文：Collision Stability

其图中纵轴建议明确写成：

- `Collision Termination Ratio (%)`

### 2. Episode 级碰撞终止标记

每个 episode 先转换为一个二值标记：

- `collision_terminated = 1`
- `collision_terminated = 0`

推荐判定优先级如下：

1. 若 `reset_reason` 明确表示碰撞终止，则记为 `1`
2. 若 `reset_reason` 缺失、不规范或无法稳定判断，则当 `collision_count_final > 0` 时记为 `1`
3. 其余情况记为 `0`

### 3. 稳定性曲线

对 `collision_terminated` 序列做滚动平均后乘以 `100`，得到最近窗口内的碰撞终止占比：

\[
CollisionRate_{MA20}(t)=\frac{1}{20}\sum_{i=t-19}^{t} collision\_terminated_i \times 100\%
\]

其解释方式为：

- 越低表示最近训练窗口内因碰撞失败的比例越低
- 越低通常意味着训练过程越稳定
- 若曲线长期下降并保持低位，可支持“碰撞稳定性改善”结论

## 总体设计

### 1. 完整口径单算法分析

在 `visualize_scan_csv.py` 的完整单算法分析链中新增：

- `collision_stability.png`

该图定位为：

- 单算法稳定性主图
- 训练过程分析图
- 碰撞相关图中的主结论图

现有碰撞相关图继续保留：

- `collision_hotspots_xy.png`
- `collision_object_breakdown.png`
- `reset_reason_rolling_ratio.png`

三者的分工明确如下：

- `collision_stability.png`
  - 回答“训练是否越来越少因碰撞失败”
- `collision_hotspots_xy.png`
  - 回答“碰撞主要集中在哪里”
- `collision_object_breakdown.png`
  - 回答“主要撞上了什么对象”
- `reset_reason_rolling_ratio.png`
  - 回答“终止结构整体如何变化”，但不再承担碰撞稳定性主图职责

### 2. 论文精简口径单算法分析

在论文精简口径中，为每个单算法分析目录各补 1 张：

- `collision_stability.png`

适用目录包括：

- `analysis_results/stage02_analysis_suite/ddpg_stage02`
- `analysis_results/stage02_analysis_suite/dqn_stage02`

这样精简口径单算法分析图从原先的 6 张扩展为 7 张，其中新增的这 1 张专门用于补充稳定性维度。

### 3. 论文精简口径横向对比分析

在横向对比口径中新增：

- `comparison_training_collision_rate.png`

它应被归类为：

- 训练过程分析图
- 稳定性补充图
- 中等偏强解释力图，但不替代最终结果指标

其作用是：

- 比较不同算法在训练过程中谁更少因碰撞终止
- 补充说明“谁的训练过程更稳”

## 图表设计

### 1. 单算法 `collision_stability.png`

建议设计为单子图，保持和现有趋势图风格一致。

#### 横轴

- `Episode`

#### 纵轴

- `Collision Termination Ratio (%)`

#### 曲线内容

- 淡色原始 episode 二值点或细线
  - `0%` 表示该轮未因碰撞终止
  - `100%` 表示该轮因碰撞终止
- 一条主曲线
  - `MA20` 滚动碰撞终止占比

#### 图表解释

该图主要用于判断：

- 训练后期是否越来越少因碰撞失败
- 是否已从“高碰撞不稳定阶段”进入“低碰撞稳定阶段”

### 2. 横向对比 `comparison_training_collision_rate.png`

建议采用与现有 comparison 图一致的带状均值曲线风格。

#### 横轴

- `Episode`

#### 纵轴

- `Collision Termination Ratio (%)`

#### 曲线内容

- 每种算法各一条滚动均值曲线
- 保持现有算法颜色和线型映射

#### 图表解释

该图主要用于判断：

- 两种算法谁在训练过程中更少出现碰撞终止
- 哪一方更早进入低碰撞区间
- 稳定性差异是否与 reward、scan efficiency 等过程图一致

## 组件边界

### `multirotor/Algorithm/visualize_scan_csv.py`

职责变化：

- 新增碰撞终止判定辅助逻辑
- 新增 `plot_collision_stability`
- 将该图加入完整口径绘图管线

它负责：

- 单算法图生成
- 与 `RunData.episode_df` 的 episode 级字段对齐

### `multirotor/Algorithm/training_analyzer.py`

职责变化：

- 新增派生训练指标 `collision_termination_rate`
- 允许该指标进入横向对比绘图逻辑

它负责：

- 统一各算法训练 CSV 口径
- 输出 `comparison_*`、`latest_comparison_*`、`recent_window_comparison_*` 系列图

### `multirotor/Algorithm/visualize_training_data.py`

职责变化：

- 在算法对比分析流程中补上碰撞稳定性对比图生成调用

它负责：

- 自动分析入口
- 把新图纳入现有比较输出体系

## 文档设计

以下文档需要同步更新：

### 1. 完整口径图表总说明

文件：

- `multirotor/DDPG_Weight/docs/ANALYSIS_CHART_GUIDE_ZH_EN.md`

更新内容：

- 在 `Chart Index / 图表总览` 中加入 `collision_stability.png`
- 新增该图的详细图解说明
- 调整推荐阅读顺序，把它放入训练稳定性分析链

### 2. Stage02 总说明

文件：

- `analysis_results/stage02_analysis_suite/README_STAGE02_ANALYSIS_ZH.md`

更新内容：

- 在训练过程分析中加入 `collision_stability.png`
- 在“当前 Stage02 的阶段性结论”中补充稳定性观察口径

### 3. 单算法说明

文件：

- `analysis_results/stage02_analysis_suite/ddpg_stage02/README_SINGLE_ANALYSIS_ZH.md`
- `analysis_results/stage02_analysis_suite/dqn_stage02/README_SINGLE_ANALYSIS_ZH.md`

更新内容：

- 将图表清单从 6 张更新为 7 张
- 为 `collision_stability.png` 增加横轴、纵轴与用途说明
- 在“每张图分别解释什么”中加入碰撞稳定性说明

### 4. 横向对比说明

文件：

- `analysis_results/stage02_analysis_suite/comparison/README_COMPARISON_EXPERIMENT_ZH.md`

更新内容：

- 将横向对比图表清单增加一张 `comparison_training_collision_rate.png`
- 明确该图主要用于稳定性比较
- 在“强可比 / 中等可比 / 弱可比”分类中给出其定位

### 5. 图注模板

文件：

- `analysis_results/stage02_analysis_suite/README_PAPER_FIGURE_CAPTIONS_ZH.md`

更新内容：

- 为单算法 `collision_stability.png` 添加图注模板
- 为横向对比 `comparison_training_collision_rate.png` 添加图注模板

## 可比性定位

建议将碰撞稳定性图定位为：

- 单算法分析中：训练稳定性主图之一
- 横向对比中：训练过程稳定性补充图

它不属于最终任务结果图，因此不应与以下结果主图同级使用：

- `comparison_scan_scan_ratio.png`
- `comparison_scan_global_avg_entropy.png`

更合适的论文表达方式是：

- 用它支撑“训练过程更稳、更少因碰撞失败”
- 不直接用它替代“最终覆盖效果更好”的结论

## 测试策略

### 单元测试

建议为以下行为补测试：

- `reset_reason` 表示碰撞时，能正确判定该轮为碰撞终止
- `reset_reason` 缺失时，能回退使用 `collision_count_final`
- 无碰撞字段或空数据时，图生成逻辑能安全跳过

### 绘图回归测试

建议验证：

- `collision_stability.png` 会在具备 episode 数据时生成
- `comparison_training_collision_rate.png` 会在对比流程中生成

### 文档回归检查

建议验证：

- 单算法说明文档中的图数量与图名一致
- 横向对比说明中的新增图名与脚本输出命名一致

## 风险

- 不同历史数据中的 `reset_reason` 命名可能不完全统一，碰撞终止判定需要做稳健匹配
- 部分旧数据可能没有可靠的 `collision_count_final` 字段，需允许回退失败后返回“无法生成”
- 若把该图误写成最终性能主图，容易与覆盖率、熵值等结果指标混淆

## 最终决策

本次按以下方式落地：

- 在完整口径单算法分析中新增 `collision_stability.png`
- 在论文精简口径中为 `DDPG+APF` 与 `纯 DQN` 各新增一张 `collision_stability.png`
- 在横向对比口径中新增 `comparison_training_collision_rate.png`
- 统一使用“碰撞终止占比”作为稳定性主指标
- 保留 `collision_hotspots_xy.png` 与 `collision_object_breakdown.png` 作为完整口径中的碰撞诊断图

这样可以在不扩大数据采集范围的前提下，为当前分析体系补上一条更直接、更适合论文表述的稳定性证据链。
