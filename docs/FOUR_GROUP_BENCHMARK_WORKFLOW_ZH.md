# 四组论文实验工作流

## 1. 目标

这条工作流用于同时支持：

- 四组统一冻结评测主结果
- family 横向对比分析
- 现有 DDPG/DQN 训练链兼容保留

四组算法固定为：

- `fixed_apf`
- `random_apf`
- `ddpg_apf`
- `pure_dqn`

## 2. 训练阶段

学习算法仍按原训练链进行，但统一增加 `seed` 元数据。

DDPG:

```powershell
powershell -File .\scripts\Run_Paper_Training_Seeds.ps1 -Algorithm ddpg_apf -Seeds 20260413,20260414,20260415 -StageName stage01 -StageIndex 1
```

DQN:

```powershell
powershell -File .\scripts\Run_Paper_Training_Seeds.ps1 -Algorithm pure_dqn -Seeds 20260413,20260414,20260415 -StageName stage01 -StageIndex 1
```

## 3. 冻结评测阶段

统一冻结评测入口：

```bat
scripts\Run_Four_Group_Benchmark.bat
```

这个入口会：

- 读取 `multirotor/system_config.json` 中的 `paper_benchmark`
- 按 seeds 和 episode 数运行四组评测
- 生成 `analysis_results/four_group_benchmark/four_group_eval_episodes.csv`
- 继续生成四组主结果图表和 family 比较结果

## 4. 四组主结果分析

如只想对已有 `four_group_eval_episodes.csv` 重跑主分析：

```bat
scripts\Analyze_Four_Group_Benchmark.bat
```

固定产物：

- `four_group_eval_episodes.csv`
- `four_group_eval_seed_summary.csv`
- `four_group_summary.csv`
- `scan_ratio_boxplot.png`
- `entropy_boxplot.png`
- `efficiency_bar.png`
- `safety_bar.png`
- `reset_reason_stacked_bar.png`

## 5. Family 分析

如只想重跑 family 横向分析：

```bat
scripts\Analyze_Family_Comparisons.bat
```

默认首批 family：

- `apf_family`
- `learning_family`

输出目录：

- `analysis_results/family_comparisons/apf_family`
- `analysis_results/family_comparisons/learning_family`

## 6. 新算法接入

新算法接入时的规则：

1. 必须至少进入 `global_benchmark`
2. family 归类以 `multirotor/benchmark_registry.json` 为准
3. 若未显式注册，系统只保底进入 `global_benchmark`
4. 可用注册表辅助工具生成推荐和模板

示例：

```bash
python multirotor/Algorithm/benchmark_registry_helper.py validate
python multirotor/Algorithm/benchmark_registry_helper.py recommend --algorithm-type ppo_scan --control-mode dqn --trainable
python multirotor/Algorithm/benchmark_registry_helper.py scaffold --algorithm-type ppo_scan --control-mode dqn --trainable
```

## 7. 论文结果对应关系

建议论文中这样组织：

- 主结果：`four_group_benchmark`
- 同类横向补充：`family_comparisons`
- 特定算法解释：`algorithm_specific`
- 学习过程证据：现有 `stage01/stage02/two_stage` 分析
