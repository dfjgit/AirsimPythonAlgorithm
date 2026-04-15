# Two-Stage Analysis Suite Design

## Goal

在现有 `stage01_analysis_suite` 与 `stage02_analysis_suite` 的基础上，新建一套独立的“二阶段总分析”产物，将 `DDPG+APF` 与 `纯 DQN` 的两阶段实验结果串联为统一证据链，输出可直接服务论文写作的新分析文档。

## Why A New Suite

当前项目已经有两套成熟的阶段性分析：

- `analysis_results/stage01_analysis_suite/`
- `analysis_results/stage02_analysis_suite/`

它们分别完成了：

- `stage01`：回答“算法是否学到东西、是否值得进入下一阶段”
- `stage02`：回答“追训后是否稳定成形、当前阶段谁的结果更强”

但若直接将两套现有说明拼接到论文中，会出现三个问题：

1. 阶段叙事被拆散，`stage01 -> stage02` 的演化不连续。
2. 单算法内部跃迁与横向算法对比混在不同目录，引用成本高。
3. 论文图注与正文结果说明缺少一套统一的“二阶段总口径”。

因此，本设计选择新增独立目录，而不是覆盖既有 `stage01` / `stage02` 产物。

## Chosen Approach

采用“新建二阶段总分析套件，复用既有图与统计结果，补充新的阶段汇总表和论文级说明”的方案。

不采用以下方案：

- 不只写一份总说明，因为那会让单算法与横向对比仍然分散在旧目录中。
- 不重算全部图，因为本次任务重点是“生成新的对比分析”，不是重做整套分析管线。

## Scope

本次新增产物聚焦文档与汇总层，不重绘现有阶段图。

本次会新增：

- 二阶段总说明
- 二阶段方法学说明
- `DDPG+APF` 二阶段单算法分析
- `纯 DQN` 二阶段单算法分析
- 二阶段横向对比分析
- 二阶段论文图注模板
- 二阶段关键指标汇总表

本次不会新增：

- 全量新 PNG 图
- 重新生成 `stage01_analysis_suite` 与 `stage02_analysis_suite`
- 修改现有训练数据

## Output Structure

新增目录：

- `analysis_results/two_stage_analysis_suite/`

目录结构：

```text
analysis_results/two_stage_analysis_suite/
  README_TWO_STAGE_ANALYSIS_ZH.md
  README_EXPERIMENT_METHOD_ZH.md
  README_PAPER_FIGURE_CAPTIONS_ZH.md
  two_stage_key_metrics.csv
  ddpg_two_stage/
    README_SINGLE_ANALYSIS_ZH.md
  dqn_two_stage/
    README_SINGLE_ANALYSIS_ZH.md
  comparison/
    README_COMPARISON_EXPERIMENT_ZH.md
```

## Data Sources

### Existing Analysis Assets

直接复用：

- `analysis_results/stage01_analysis_suite/ddpg_stage01/*`
- `analysis_results/stage01_analysis_suite/dqn_stage01/*`
- `analysis_results/stage01_analysis_suite/comparison/*`
- `analysis_results/stage02_analysis_suite/ddpg_stage02/*`
- `analysis_results/stage02_analysis_suite/dqn_stage02/*`
- `analysis_results/stage02_analysis_suite/comparison/*`

### Summary Inputs

关键统计量主要来自：

- `analysis_results/stage01_analysis_suite/ddpg_stage01/summary_report.csv`
- `analysis_results/stage01_analysis_suite/dqn_stage01/summary_report.csv`
- `analysis_results/stage02_analysis_suite/ddpg_stage02/summary_report.csv`
- `analysis_results/stage02_analysis_suite/dqn_stage02/summary_report.csv`
- `analysis_results/stage02_analysis_suite/comparison/stage02_algorithm_comparison_report.csv`

必要时补充引用原始训练表：

- `multirotor/DDPG_Weight/airsim_training_logs/ddpg_training_ddpg_apf_20260326_234951_stage01_20260326_234955.csv`
- `multirotor/DDPG_Weight/airsim_training_logs/ddpg_training_ddpg_apf_20260326_234951_stage02_20260331_003640.csv`
- `multirotor/DQN_Movement/logs/dqn_scan_data/dqn_training_pure_dqn_20260330_005101_stage01_20260330_005101.csv`
- `multirotor/DQN_Movement/logs/dqn_scan_data/dqn_training_pure_dqn_20260330_005101_stage02_20260402_005952.csv`

## Analysis Questions

这套二阶段总分析需要统一回答以下问题：

1. `DDPG+APF` 与 `纯 DQN` 在 `stage01` 时各自处于什么学习阶段。
2. 两种算法从 `stage01` 到 `stage02` 分别发生了什么变化。
3. 哪些变化属于“训练稳定性改善”，哪些属于“结果能力提升”。
4. 在 `stage02` 的正式阶段结果下，哪种算法在结果指标上更强。
5. 哪些图适合支撑论文主结论，哪些图只能支撑辅助解释。

## Analysis Principles

### Evidence Chain

新套件采用以下叙事顺序：

1. 单算法看内部阶段跃迁
2. 横向对比看同阶段差异
3. 统一方法学说明约束可比性边界
4. 论文图注模板沉淀最终表述

### Metric Layers

指标分三层：

- 强可比：
  - 最终全局扫描率
  - 最终全局平均熵
- 中等可比：
  - 碰撞终止占比
  - 碰撞次数
  - 按时间归一化扫描产出
  - 按电量归一化扫描产出
- 弱可比：
  - 奖励
  - `Cell/Step`

### Stage Narrative

对每种算法都必须回答：

- `stage01` 的状态是什么
- `stage02` 是否延续、修正或放大了 `stage01`
- 当前是否进入平台期
- 当前更像“阶段上限附近”还是“仍有明显继续增长空间”

## File Responsibilities

### `README_TWO_STAGE_ANALYSIS_ZH.md`

负责：

- 解释本目录用途
- 给出推荐阅读顺序
- 说明训练过程分析与结果表现分析的区别
- 总结二阶段主结论与结论边界

### `README_EXPERIMENT_METHOD_ZH.md`

负责：

- 说明所有图对应的数据来源
- 明确 `stage01` 与 `stage02` 的引用关系
- 解释强可比 / 中等可比 / 弱可比口径
- 说明哪些指标跨阶段可直接并列，哪些只能补充性引用

### `ddpg_two_stage/README_SINGLE_ANALYSIS_ZH.md`

负责：

- 串联 `DDPG+APF stage01` 与 `stage02`
- 逐图解释 DDPG 的训练过程、阶段跃迁、结果平台化
- 给出每张图的论文级分析说明

### `dqn_two_stage/README_SINGLE_ANALYSIS_ZH.md`

负责：

- 串联 `纯 DQN stage01` 与 `stage02`
- 逐图解释 DQN 从前期高代价探索到后期高覆盖形成的过程
- 给出每张图的论文级分析说明

### `comparison/README_COMPARISON_EXPERIMENT_ZH.md`

负责：

- 比较两种算法在 `stage01` 与 `stage02` 的相对关系
- 区分“阶段内横向对比”与“跨阶段纵向提升”
- 输出适合论文主结果段落的表述

### `README_PAPER_FIGURE_CAPTIONS_ZH.md`

负责：

- 为二阶段引用到的全部关键图提供论文式图注模板
- 图注不仅说明坐标和对象，还说明图在论文中的论证功能

### `two_stage_key_metrics.csv`

负责：

- 并排整理 `DDPG+APF` 与 `纯 DQN` 在 `stage01` / `stage02` 的关键指标
- 作为正文中定量引用的统一表

建议字段：

- `algorithm`
- `stage`
- `episodes`
- `avg_reward`
- `tail_reward`
- `avg_length`
- `tail_length`
- `avg_scan_efficiency`
- `tail_scan_efficiency`
- `avg_scan_ratio_pct`
- `tail_scan_ratio_pct`
- `avg_entropy`
- `tail_entropy`
- `avg_collision_count`
- `tail_collision_count`
- `avg_out_of_range_count`
- `tail_out_of_range_count`
- `avg_scan_cells_per_second`
- `avg_scan_cells_per_volt_drop`
- `notes`

其中：

- `stage01` 中没有现成归一化效率指标的字段时，可留空
- `stage02` 中若没有 `tail_*`，则使用现有 `front20/back20` 或阶段说明替代

## Writing Style

文风要求：

- 中文
- 面向论文结果章节
- 避免口语化“谁吊打谁”式表达
- 每张图都要同时写出：
  - 图展示了什么
  - 图能支持什么结论
  - 图不能单独支持什么结论
  - 图在全文中的合适角色

## Risks And Guardrails

### Risk 1: 把现有两套分析简单拼接

规避方式：

- 所有新文档必须围绕“二阶段演化”重写
- 不允许只列旧链接不做新解释

### Risk 2: 把弱可比指标写成最终优劣结论

规避方式：

- 统一在总说明、方法学说明、横向对比说明中重复标注可比性层级

### Risk 3: 强行对 `stage01` 使用 `stage02` 才有的指标

规避方式：

- 二阶段汇总表允许部分字段为空
- 正文中对 `stage01` 重点使用两阶段共同指标

## Acceptance Criteria

完成后应满足：

1. `analysis_results/two_stage_analysis_suite/` 目录完整存在。
2. 至少包含总说明、方法学说明、两份单算法说明、一份横向对比说明、一份论文图注模板。
3. 每份说明都明确引用对应的原始阶段图或汇总表。
4. 每张关键图都配有论文级分析说明。
5. 文中明确区分强可比 / 中等可比 / 弱可比指标。
6. 二阶段主结论能够回答“算法内部如何演化、最终谁在结果层面更强”。
