# DDPG 与 DQN 实验流程说明

本文档用于规范当前项目中 `DDPG+APF` 与 `纯 DQN` 的训练、追训、过程对比、结果对比和最终报告产出流程。

适用目录：
- [DQN_Movement](/D:/Work/Python%20Project/airsim-python-algorithm/multirotor/DQN_Movement)
- [DDPG_Weight](/D:/Work/Python%20Project/airsim-python-algorithm/multirotor/DDPG_Weight)
- [Algorithm](/D:/Work/Python%20Project/airsim-python-algorithm/multirotor/Algorithm)

## 1. 实验目标

本实验最终需要回答两类不同问题：

1. 过程对比
谁学得更快，谁更稳定，谁更容易出现震荡、动作塌缩或坏策略。

2. 结果对比
最终训练好的模型，在同一任务下谁表现更好，谁覆盖率更高，谁更少出圈、更少碰撞。

这两类问题必须分开分析，不能混成一套结论。

## 2. 当前正式对比实验口径

当前这轮正式对比实验，已经统一到以下口径：

### 2.1 训练模式

- DDPG：默认从头训练，不自动加载旧模型
- DQN：默认从头训练，不自动加载旧模型

如果要续训，必须显式启用：
- DDPG：传 `--continue-model`
- DQN：设置 `USE_PRETRAINED=1`

### 2.2 训练预算

- DDPG：
  - `total_timesteps = 21600`
  - `step_duration = 2.0`
  - 约 12 小时预算
- DQN：
  - `total_timesteps = 30000`
  - `step_duration = 0.5`
  - `action_repeat = 3`
  - 约 12.5 小时预算

说明：
- 12 小时是正式对比的统一预算，不是保证收敛的硬门槛。
- 如果 12 小时后仍未完全收敛，先保留这轮作为“同预算对比”结果，后续再做追训实验。

### 2.3 终止条件

两边当前已尽量统一：

- 最大单轮时长：`300s`
- 目标扫描率：`0.25`
- 碰撞终止阈值：`max_collision_count = 6`
- 出圈终止窗口：约 `24s`
  - DDPG：连续 12 步，`2s/step`
  - DQN：`max_out_of_range_duration_sec = 24.0`
- 电量耗尽终止：`<= 3.2V`

### 2.4 当前仍然本质不同的地方

以下差异不属于实验口径错误，而是算法定义差异：

- DDPG：`DDPG + APF`
- DQN：`纯 DQN 离散动作控制`
- 动作空间不同
- 状态空间不同
- reward shaping 细节不同

因此最终报告里必须明确写成：
- `DDPG+APF`
- `纯 DQN`

而不是把它们写成两个完全同构的纯 RL 算法。

## 3. 训练输出目录

### 3.1 DQN

- 运行日志目录：[movement_dqn_airsim](/D:/Work/Python%20Project/airsim-python-algorithm/multirotor/DQN_Movement/logs/movement_dqn_airsim)
- 训练/扫描数据目录：[dqn_scan_data](/D:/Work/Python%20Project/airsim-python-algorithm/multirotor/DQN_Movement/logs/dqn_scan_data)
- 模型目录：[models](/D:/Work/Python%20Project/airsim-python-algorithm/multirotor/DQN_Movement/models)
- 分析输出目录：[analysis_results](/D:/Work/Python%20Project/airsim-python-algorithm/multirotor/DQN_Movement/logs/analysis_results)

### 3.2 DDPG

- 训练/扫描数据目录：[airsim_training_logs](/D:/Work/Python%20Project/airsim-python-algorithm/multirotor/DDPG_Weight/airsim_training_logs)
- 模型目录：[models](/D:/Work/Python%20Project/airsim-python-algorithm/multirotor/DDPG_Weight/models)

## 4. 过程对比与结果对比

## 4.1 过程对比

过程对比回答的是：
- 谁收敛更快
- 谁更稳定
- 谁更容易震荡
- 谁中后期还有提升空间

当前已归入“过程对比”的指标：
- `平均奖励`
- `最高奖励`
- `训练轮次`
- `总耗时(s)`

说明：
- 这些指标更适合看训练趋势。
- 它们不适合单独用来下最终优劣结论。

## 4.2 结果对比

结果对比回答的是：
- 最终模型谁更强
- 谁扫描更多
- 谁熵降得更多
- 谁更少出圈、更少碰撞

当前已归入“结果对比”的指标：
- `最终效率`
- `最终扫描率(%)`
- `最低熵值`

说明：
- 正式结论应优先依赖这些强可比指标。

## 5. 当前分析链已支持的对比视角

目前系统已经支持三套对比输出：

### 5.1 历史全量对比

用于回答：
- 历史上整体平均表现怎样
- 哪个算法长期更稳

### 5.2 最新一轮对比

用于回答：
- 最新一次从头训练的原始结果怎样
- 当前最新实验是否比历史更好

### 5.3 最近窗口对比

用于回答：
- 最近一段训练状态如何
- 避免“最新文件太短”导致统计不稳

输出目录示例：
- [algorithm_comparison](/D:/Work/Python%20Project/airsim-python-algorithm/analysis_results/algorithm_comparison)

## 6. 追训阶段命名和分析规则

为了让后续追训数据能自动加入分析，系统现在已经实现了统一的阶段命名与合并规则。

### 6.1 阶段元数据

每轮训练现在都会自动带上以下字段：
- `experiment_id`
- `stage_name`
- `stage_index`
- `is_resume`
- `source_model`

这些字段会写入：
- `scan_data_*.csv`
- `ddpg_training_*.csv`
- `dqn_training_*.csv`

### 6.2 自动命名规则

默认规则如下：

- 从头训练：
  - `stage01_from_scratch`
- 第一次续训：
  - `stage02_finetune`
- 第二次续训：
  - `stage03_finetune`

文件名也会自动带阶段标识，例如：
- `scan_data_ddpg_apf_20260325_101500_stage01_20260325_101501.csv`
- `dqn_training_pure_dqn_20260325_101500_stage02_20260326_090001.csv`

### 6.3 续训时如何继承阶段

保存模型时，系统会在模型旁边写一份阶段元数据 sidecar：
- `*.stage_meta.json`

后续续训时会自动读取这份 sidecar，继承：
- `experiment_id`
- 上一阶段序号
- `source_model`

因此：
- `12h 从头训练`
- `在 12h 模型基础上继续训练 12h`

会被系统识别为：
- 同一 `experiment_id`
- 不同 `stage_index`

### 6.4 分析器如何处理追训数据

分析器现在会自动把同一 `experiment_id` 的多个阶段拼接成一条连续训练记录：

- `episode` 会连续累加
- `elapsed_time` 会连续累加
- `timestep` 会连续累加
- 图中会把多阶段训练当成同一实验的连续曲线

也就是说，后续追训数据不需要手工合并 CSV。

## 7. 从头训练与追训的区别

### 7.1 从头训练

用于：
- 正式同预算对比
- 新配置首次验证

特点：
- 创建新模型
- 独立生成一组训练日志
- 通常对应 `stage01_from_scratch`

### 7.2 追训

用于：
- 观察继续给训练时间后，性能上限还能否提高
- 形成 `12h -> 24h -> 36h` 的连续成长曲线

特点：
- 加载上一阶段模型继续训练
- 会生成新的 CSV 文件
- 但分析时会自动拼接回同一实验

## 8. 推荐实验方案

### 8.1 阶段 A：正式对比实验

目标：
- 做同预算、同口径的公平比较

流程：
1. DDPG 从头训练 12h
2. DQN 从头训练 12h
3. 生成过程对比图
4. 生成结果对比图

回答的问题：
- 在同样训练预算下，谁学得更快、谁效果更好

### 8.2 阶段 B：追训实验

目标：
- 看继续给时间后，最终上限能到哪里

流程：
1. 加载阶段 A 结束时的最终模型
2. 继续训练
3. 自动生成 `stage02_finetune`
4. 继续分析同一 `experiment_id` 的连续曲线

回答的问题：
- 如果继续训练，DQN 是否能追上或逼近 DDPG+APF
- DDPG 是否还能进一步提高

## 9. 冻结模型统一评测

冻结模型统一评测的含义是：
- 把 DDPG 和 DQN 都当成“已经训练完成的模型”
- 不再更新参数
- 在同一套评测条件下只跑推理

统一评测要固定：
- leader 路径
- 初始位置
- 障碍布局
- 最大时长
- reset 条件
- 电量规则
- 评测次数

它主要回答：
- 最终模型谁更强
- 在同一场景下谁覆盖率更高
- 谁更少碰撞、更少出圈

## 10. 推荐的最终报告结构

建议按以下结构组织最终报告：

### 第一部分：训练过程对比

内容包括：
- reward 曲线
- scan efficiency 曲线
- scan ratio 曲线
- entropy 曲线
- 最新一轮 / 最近窗口 / 历史全量的对比结论

### 第二部分：最终结果对比

内容包括：
- 最终效率
- 最终扫描率
- 最低熵值
- reset 原因占比
- 冻结模型统一评测结果

### 第三部分：综合结论

内容包括：
- DDPG+APF 是否仍整体更强
- DQN 是否存在明显追赶趋势
- DQN 当前主要短板是什么
- 是否值得继续追训

## 11. 本轮正式实验建议执行顺序

1. 固定当前代码与配置版本
2. 清空旧模型与旧训练数据
3. DDPG 从头训练
4. DQN 从头训练
5. 生成过程对比图
6. 生成结果对比图
7. 如有需要，继续做 `stage02_finetune`
8. 最后补冻结模型统一评测

## 12. 常用环境变量

如需手动指定阶段元数据，可使用：

- `EXPERIMENT_ID`
- `TRAIN_STAGE_NAME`
- `TRAIN_STAGE_INDEX`
- `USE_PRETRAINED`

默认情况下，不指定也能自动完成分期。

## 13. 一句话总结

当前系统已经支持：
- 从头训练
- 追训分期
- 自动命名
- 自动把多阶段训练拼接进分析
- 过程对比与结果对比分开输出

因此后续你可以按“正式对比实验 -> 追训实验 -> 冻结模型统一评测”这条线完整产出训练图、对比图和实验报告。
