# 实验分析总说明

## 1. 文件用途

本文档用于对 [analysis_results](/D:/Work/Python%20Project/airsim-python-algorithm/analysis_results) 目录下的当前实验产出与后续实验计划做统一说明。

阅读这份文件后，应该能够快速回答以下问题：

- 当前已经完成了哪些实验分析
- 现阶段已经可以支撑哪些论文内容
- 现阶段还缺哪些实验，才能补足到更完整的论文级产出
- 后续最推荐的实验推进顺序是什么

## 2. 当前目录结构

当前目录下主要包括以下内容：

- [stage01_analysis_suite](/D:/Work/Python%20Project/airsim-python-algorithm/analysis_results/stage01_analysis_suite)
- [stage02_analysis_suite](/D:/Work/Python%20Project/airsim-python-algorithm/analysis_results/stage02_analysis_suite)
- [two_stage_analysis_suite](/D:/Work/Python%20Project/airsim-python-algorithm/analysis_results/two_stage_analysis_suite)
- [algorithm_comparison](/D:/Work/Python%20Project/airsim-python-algorithm/analysis_results/algorithm_comparison)

其中：

- `stage01_analysis_suite`  
  用于保存从头训练阶段的单算法分析与横向对比分析。
- `stage02_analysis_suite`  
  用于保存追训阶段的单算法分析与横向对比分析。
- `two_stage_analysis_suite`  
  用于将 `stage01` 与 `stage02` 串成统一的二阶段论文级分析证据链。
- `algorithm_comparison`  
  主要保留历史生成的通用对比图、阶段性比较图和若干实验报告。

## 3. 当前实验已经完成的内容

### 3.1 Stage01：从头训练阶段分析

对应目录：

- [stage01_analysis_suite](/D:/Work/Python%20Project/airsim-python-algorithm/analysis_results/stage01_analysis_suite)

这一阶段已经完成：

- `DDPG+APF stage01` 单算法分析
- `纯 DQN stage01` 单算法分析
- `DDPG+APF vs 纯 DQN` 横向对比分析
- 论文级说明页
- 图表方法学说明
- 图注模板

这一阶段主要回答：

- 从头训练时，哪种方法更稳
- 哪种方法更早进入平台
- 哪种方法在后期有更明显提升

当前较稳的阶段结论是：

- `DDPG+APF stage01` 更稳定，更早进入平台期
- `纯 DQN stage01` 后期提升更明显，但在从头训练阶段的稳定性较弱

### 3.2 Stage02：追训阶段分析

对应目录：

- [stage02_analysis_suite](/D:/Work/Python%20Project/airsim-python-algorithm/analysis_results/stage02_analysis_suite)

这一阶段已经完成：

- `DDPG+APF stage02` 单算法分析
- `纯 DQN stage02` 单算法分析
- `DDPG+APF vs 纯 DQN` 横向对比分析
- 论文级说明页
- 图表方法学说明
- 图注模板
- 强可比 / 中等可比 / 弱可比标注
- 两张补充归一化图：
  - 按时间归一化的扫描产出图
  - 按电量归一化的扫描产出图

这一阶段主要回答：

- 追训后哪种方法在结果层面更强
- 哪种方法在时间效率、电量效率上更优
- 哪些差异属于训练过程差异，哪些差异属于结果表现差异

当前较稳的阶段结论是：

- `DDPG+APF stage02` 更稳，更早进入平台
- `纯 DQN stage02` 在最终扫描率、最终全局平均熵、按时间归一化扫描产出、按电量归一化扫描产出方面更优

### 3.3 Two-Stage：二阶段总分析

对应目录：

- [two_stage_analysis_suite](/D:/Work/Python%20Project/airsim-python-algorithm/analysis_results/two_stage_analysis_suite)

这一套新分析已经完成：

- `DDPG+APF` 二阶段单算法分析
- `纯 DQN` 二阶段单算法分析
- 二阶段横向对比分析
- 二阶段方法学说明
- 二阶段论文图注模板
- 二阶段统一指标汇总表

这一套分析主要回答：

- 两种算法在 `stage01 -> stage02` 中分别发生了什么
- 哪种算法更像“平台维持”，哪种算法更像“阶段跃迁”
- 哪些结论适合写进论文主结果，哪些只能作为过程解释

当前较稳的二阶段结论是：

- `DDPG+APF` 的优势更多体现在稳定性、动作组织性与步级效率
- `纯 DQN` 的优势更多体现在最终覆盖率、更低熵值以及更高的归一化任务产出

## 4. 当前已经能支撑的论文内容

基于现有 `stage01` 与 `stage02` 结果，当前已经足够支撑以下论文内容：

### 4.1 系统实现与实验平台说明

可说明内容包括：

- `APF`
- `DDPG+APF`
- `纯 DQN`
- 阶段化训练与续训机制
- 统一分析链与论文级图表说明

### 4.2 训练过程分析

可说明内容包括：

- `DDPG+APF` 更稳、更早平台化
- `纯 DQN` 在后期具有更明显的学习提升
- `Reward`、`Episode Length`、`Cell/Step` 等过程指标的变化趋势

### 4.3 阶段性结果对比

可说明内容包括：

- `stage01` 阶段的从头训练对比
- `stage02` 阶段的追训结果对比
- 强可比、中等可比、弱可比指标的论文方法学区分

## 5. 当前还缺哪些实验，才能补足论文级产出

如果目标是“更完整、更严谨的论文级实验产出”，当前还建议补以下内容。

### 5.1 P0：冻结模型统一评测

这是当前最重要的缺项。

建议做法：

- 固定 `DDPG stage02` 最终 checkpoint
- 固定 `DQN stage02` 最终 checkpoint
- 在相同场景、相同初始条件下只做推理，不再训练
- 输出统一的评测结果表和评测对比图

这一部分补上后，论文结论会从“阶段性训练对比”提升为“最终能力对比”。

### 5.2 P0：统一评测协议

建议固定：

- 相同初始位置
- 相同障碍布局
- 相同 leader 轨迹或 leader 规则
- 相同 episode 时长上限
- 相同 reset 条件
- 相同随机种子集合

### 5.3 P0：多随机种子重复实验

建议至少：

- 每个算法 `3` 个随机种子
- 更理想为 `5` 个随机种子

最终输出建议包括：

- 均值
- 标准差
- 箱线图
- 必要时再做简单显著性检验

### 5.4 P1：验证集 / 测试集分离

如果后续要从多个 checkpoint 里挑最终模型，建议把：

- 验证集  
  用于选择 checkpoint
- 测试集  
  仅用于最终报告

分开处理。

### 5.5 P2：辅助基线

如果后续时间允许，可考虑补：

- 固定权重 APF 基线

这会帮助论文更清楚地说明：

- 学习是否确实优于固定解析策略

但这一项属于加分项，不是当前必须项。

## 6. 当前最推荐的后续实验流程

建议后续按以下顺序推进：

1. 锁定当前 `DDPG stage02` 与 `DQN stage02` 的最终 checkpoint
2. 设计并固定统一评测协议
3. 做冻结模型统一评测
4. 每个算法至少跑 `3` 个随机种子
5. 基于冻结评测结果输出最终总表和对比图
6. 若还有时间，再补固定 APF 辅助基线

## 7. 当前最合理的论文结论边界

如果当前论文只围绕 `DDPG+APF` 与 `纯 DQN` 两种方法展开，那么更稳妥的结论边界应为：

- 在当前定义的任务场景、训练预算与实验口径下，`纯 DQN stage02` 在最终扫描率、最终全局平均熵、按时间归一化扫描产出、按电量归一化扫描产出上优于 `DDPG+APF stage02`
- `DDPG+APF` 的主要优势体现在训练稳定性更高、较早进入平台期、单位步动作效率更高
- 这一结论属于“当前实验设置下的阶段性结果结论”，而不是对算法普适优劣的绝对判断

不建议直接写成：

- `DQN` 全面优于 `DDPG`
- 当前结果已经说明算法普适优劣

## 8. 推荐阅读顺序

建议阅读顺序如下：

1. [Stage01 总说明](/D:/Work/Python%20Project/airsim-python-algorithm/analysis_results/stage01_analysis_suite/README_STAGE01_ANALYSIS_ZH.md)
2. [Stage02 总说明](/D:/Work/Python%20Project/airsim-python-algorithm/analysis_results/stage02_analysis_suite/README_STAGE02_ANALYSIS_ZH.md)
3. [Stage02 方法学说明](/D:/Work/Python%20Project/airsim-python-algorithm/analysis_results/stage02_analysis_suite/README_EXPERIMENT_METHOD_ZH.md)
4. [Stage02 横向对比说明](/D:/Work/Python%20Project/airsim-python-algorithm/analysis_results/stage02_analysis_suite/comparison/README_COMPARISON_EXPERIMENT_ZH.md)

## 9. 一句话总结

当前 `analysis_results` 目录已经足以支撑一篇“阶段性结果充分、方法学说明较完整”的论文实验章节；若要补足到更严谨的最终论文级结论，下一步最关键的是：**冻结模型统一评测 + 多随机种子重复实验**。

## 四组论文实验新增目录

本仓库现已支持新的四组论文实验分析目录：

- `analysis_results/four_group_benchmark`
- `analysis_results/family_comparisons/<family_id>`
- `analysis_results/algorithm_specific/<algorithm_type>`

推荐使用顺序：

1. 先运行四组冻结评测，生成 `four_group_eval_episodes.csv`
2. 再运行四组 benchmark 主分析
3. 最后运行 family 分析

对应入口：

```bat
scripts\Run_Four_Group_Benchmark.bat
scripts\Analyze_Four_Group_Benchmark.bat
scripts\Analyze_Family_Comparisons.bat
```

四组 benchmark 主结果固定产物：

- `four_group_eval_episodes.csv`
- `four_group_eval_seed_summary.csv`
- `four_group_summary.csv`
- `scan_ratio_boxplot.png`
- `entropy_boxplot.png`
- `efficiency_bar.png`
- `safety_bar.png`
- `reset_reason_stacked_bar.png`
