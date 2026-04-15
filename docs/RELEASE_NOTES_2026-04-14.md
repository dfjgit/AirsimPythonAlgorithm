# 2026-04-14 版本说明

## 本次新增

本次版本主要补齐了“实验工作流”这一层能力，把分散的训练、归档、分析和建议逻辑整理成了可复用入口。

新增两条核心 workflow：

1. 论文对比分析实验工作流
- 用于组织 `ddpg_apf` 与 `pure_dqn` 的 stage01 对比实验
- 支持统一实验目录归档
- 支持继续训练建议输出

2. 虚实两阶段工作流
- 用于组织 `sim_pretrain -> real_weighted_refine`
- 支持 `online` 与 `offline_logs` 两种实飞修正模式
- 支持双阶段分析摘要与继续修正建议

## 本次新增的便捷能力

- 新增统一实验状态管理
  - `workflow_state.json`
- 新增统一实验归档
  - comparison workflow
  - virtual-real two-stage workflow
- 新增 workflow 级推荐引擎
  - comparison workflow 推荐是否继续 `stage02_finetune`
  - two-stage workflow 推荐是否继续实飞修正
- 新增 orchestrator CLI
  - `--workflow comparison`
  - `--workflow virtual_real_two_stage`
  - `--refine-mode {online, offline_logs}`
- 新增批处理入口
  - `start.bat` 中 `M`：论文对比分析实验工作流
  - `start.bat` 中 `N`：虚实两阶段工作流

## 相关实现文件

- `multirotor/Algorithm/paper_workflow_state.py`
- `multirotor/Algorithm/paper_workflow_archive.py`
- `multirotor/Algorithm/paper_workflow_recommendation.py`
- `multirotor/Algorithm/paper_workflow_orchestrator.py`
- `multirotor/Algorithm/paper_two_stage_analysis.py`
- `multirotor/Algorithm/paper_two_stage_recommendation.py`
- `scripts/Run_Paper_Workflow.bat`

## 验证结果

与 workflow 相关的测试当前已通过：

- `59 passed`

补充说明：
- 当前仍会出现 `.pytest_cache` 权限 warning
- 该 warning 不影响测试通过

## 适合对外说明的一句话摘要

本版本将论文对比分析实验与虚实两阶段实验正式产品化为统一 workflow，补齐了实验状态管理、结果归档、继续训练建议和 CLI / 批处理入口，显著提升了实验流程的可重复性与可管理性。
