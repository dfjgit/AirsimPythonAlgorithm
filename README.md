# AirSim Python Algorithm

基于 AirSim 和 Unity 的多无人机协同控制与训练项目，支持 APF、DDPG+APF、DQN Movement 等算法，并提供训练日志、可视化和诊断工具。

## 项目定位

项目核心目标有三类：
- 多无人机区域扫描与协同控制
- APF 与强化学习算法训练、对比和验证
- 训练过程中的可视化、日志记录和问题诊断

核心入口：
- `multirotor/AlgorithmServer.py`：主服务，负责 AirSim、Unity、算法线程和数据采集
- `multirotor/Algorithm/scanner_algorithm.py`：APF 扫描算法
- `multirotor/DDPG_Weight/`：DDPG 权重训练
- `multirotor/DQN_Movement/`：DQN 位移控制训练
- `multirotor/Visualization/`：统一可视化模块

## 当前版本重点

本版本重点完成了配置体系整合与训练稳定性修复。

当前本地版本还包含以下系统级增强：

- 四组统一仿真对比阶段（`M`）支持先执行 APF 基线多轮仿真，再串联 `DDPG+APF` / `Pure DQN` 的 stage01 训练、冻结评测与分析
- APF baseline 默认轮次与 final benchmark 默认轮次已经拆分：
  - `paper_benchmark.apf_baseline_episodes = 100`
  - `paper_benchmark.eval_episodes_per_seed = 10`
- APF baseline / 冻结评测原始日志已经统一改为 `apf_training_*` 前缀，避免再与 `ddpg_training_*` 混淆
- 运行时可视化已经重构为以“残值”命名的双侧栏仪表板，支持：
  - `当前场景残值情况`
  - `训练状态`
  - `训练重置记录`
  - `电量状态`
  - `奖励曲线`
  - `残值采集情况`
  - `APF权重系数` / `动作分布`
- 电量面板支持按无人机数量自动切换为 1/2/3 列布局，目标覆盖最多 10 台无人机的展示
- 新增保守式旧日志清理工具，用于检测并隔离“最后一轮未完成”的遗留训练 CSV

配置整合：
- 将 15 个配置文件精简为 5 个，统一入口为 `system_config.json`
- 修正 ScannerConfigData 11 处默认值不一致问题
- 消除电池阈值、终止条件、无人机列表等参数的重复定义

已完成的训练稳定性修复：
- 修复多次重置后无人机能移动但不再采集网格熵值的问题
- 修复重置后仅无人机回原位、Leader 或网格状态残留的问题
- 统一 DDPG、DQN、HRL 环境的完整重置逻辑，保证不同算法之间对比公平
- 修复网格熵值更新异常，增加熵值边界限制，避免出现越界数据
- 优化 `start_simulation` 与重置握手机制，降低 Unity 侧状态机未恢复导致的失效概率
- 修复碰撞误判问题，不再把普通近距离编队直接当作碰撞重置
- 优化 APF 方向计算与重复访问控制，降低无人机原地徘徊和旋转现象

## 环境要求

建议环境：
- Python 3.8+
- AirSim 已正常安装并可连接
- Unity 场景已启动并能与 Python 侧通信
- Windows 环境下建议直接使用项目内批处理脚本启动

安装依赖：

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 启动仿真环境

先启动：
- Unity 场景
- AirSim 服务

### 2. 启动算法服务

进入 `multirotor` 目录后运行：

```bash
python AlgorithmServer.py
```

常用参数：

```bash
python AlgorithmServer.py --use-learned-weights
python AlgorithmServer.py --use-learned-weights --model-path DDPG_Weight/models/best_model
python AlgorithmServer.py --drones 3
python AlgorithmServer.py --no-visualization
```

### 3. 启动训练

DDPG + APF：

```bash
python multirotor/DDPG_Weight/train_with_airsim_improved.py
```

DQN Movement：

```bash
python multirotor/DQN_Movement/scripts/train_movement_with_airsim.py
```

## 关键配置文件

采用统一配置架构，所有系统级配置集中在一个文件：

| 配置文件 | 职责 |
|---|---|
| `multirotor/system_config.json` | 统一入口：无人机定义、APF 算法参数、环境规则、训练选择 |
| `multirotor/DDPG_Weight/configs/unified_train_config.json` | DDPG 训练：4 种模式（虚拟/在线/离线日志/混合） |
| `multirotor/DDPG_Weight/configs/crazyflie_reward_config.json` | Crazyflie 奖励配置 |
| `multirotor/DQN_Movement/configs/movement_dqn_config.json` | DQN Movement 专属参数 |
| `multirotor/DQN_Movement/configs/hierarchical_dqn_config.json` | 分层 DQN 专属参数 |

当前版本建议：
- 系统级参数修改只编辑 `system_config.json`
- DQN 配置中的 termination/battery 运行时自动从 `system_config.json` 继承
- 新一轮 DDPG 训练使用 `unified_train_config.json`

## 训练与日志

训练日志主要位于：
- `multirotor/DDPG_Weight/airsim_training_logs/`
- `multirotor/DQN_Movement/logs/`

其中 `scan_data_*.csv` 用于分析：
- 扫描进度
- 全局平均熵值
- 权重变化
- episode 表现

补充说明：

- comparison workflow 中的 APF baseline 产物位于：
  - `analysis_results/workflows/comparison/<experiment_id>/artifacts/apf_baseline_sim/`
- 其中：
  - `fixed_apf/`、`random_apf/` 保存阶段汇总 CSV
  - `logs/` 保存原始 `scan_data_*` 与 `apf_training_*` 日志
- APF baseline 的汇总 CSV 现在会按 episode 增量刷新，中途中断时已完成回合不会全部丢失

## 当前运行时可视化

当前系统默认的运行时可视化不再使用旧的“环境状态”面板，而是改为更强调训练与扫描过程的仪表板布局。

命名约定：

- UI 里与网格扫描有关的指标统一使用“残值”表述
- 运行时可视化中的典型面板包括：
  - `当前场景残值情况`
  - `残值采集情况`
  - `训练状态`
  - `训练重置记录`
  - `电量状态`
  - `奖励曲线`
  - `APF权重系数`

布局约定：

- 左侧栏优先展示摘要与告警类信息
- 右侧栏优先展示曲线和权重类信息
- 中间区域保留热力图、Leader 与无人机运行轨迹/位置

电量展示约定：

- 运行时电量面板支持按无人机数量自动切换为 1 / 2 / 3 列
- 设计目标是覆盖最多 10 台无人机的同时显示

## 旧日志清理工具

为避免历史训练日志中“最后一轮中断但仍残留到主 CSV”影响分析，当前版本新增了保守式清理脚本：

```bash
python multirotor/Algorithm/legacy_interrupted_log_cleanup.py --csv <training.csv> --scan-csv <scan_data.csv>
python multirotor/Algorithm/legacy_interrupted_log_cleanup.py --dir <training_dir> --apply
```

行为说明：

- 默认只分析，不改写文件
- `--apply` 时会：
  - 备份原始训练 CSV 为 `.bak`
  - 将未完成的最后一轮从主训练 CSV 中移除
  - 把被隔离的最后一轮写入 `interrupted_runs/`

## 验证与诊断工具

本版本新增或整理了多份验证脚本和说明文档，用于定位重置、扫描与熵值问题。

建议优先查看：
- `COMPLETE_FIX_SUMMARY.md`
- `VISUALIZATION_AND_SCAN_FIX_README.md`
- `ENTROPY_SCAN_DIAGNOSIS_GUIDE.md`
- `START_SIMULATION_TIMING_FIX.md`
- `multirotor/DDPG_Weight/ENTROPY_FIX_README.md`

常用脚本：
- `verify_all_fixes.py`
- `multirotor/DDPG_Weight/verify_entropy_fix.py`
- `check_entropy_scan_logs.py`
- `diagnose_entropy_scan_issue.py`
- `diagnose_scan_issue.py`

## 当前推荐的验证顺序

1. 启动 Unity 和 AirSim
2. 运行修复后的训练脚本或验证脚本
3. 连续执行多次 reset，确认：
   - 无人机、Leader、网格状态都能完整重置
   - 重置后能继续扫描并更新熵值
   - 不会无故触发碰撞重置
4. 检查最新 `scan_data_*.csv`，确认：
   - 熵值范围正常
   - 扫描比例随训练推进变化
   - 无明显卡死或平台异常

## 目录说明

```text
multirotor/
  AlgorithmServer.py                 主服务入口
  system_config.json                 统一系统配置
  AirsimServer/                      AirSim 与 Unity 通信层
  Algorithm/                         APF、网格、数据采集等算法模块
  DDPG_Weight/                       DDPG 权重训练模块
  DQN_Movement/                      DQN 位移训练模块
  Visualization/                     统一可视化模块
```

## 说明

当前仓库已完成配置整合，核心模块覆盖：
- 统一配置体系（system_config.json 单一入口）
- 重置流程
- 熵值采集
- 网格更新
- 碰撞判定
- 可视化同步
- 训练诊断

如果后续继续扩展算法或做实验对比，建议优先保证：
- 系统参数统一在 `system_config.json` 中管理
- 重置流程统一
- 统计口径统一
- 日志输出稳定
- 验证脚本可重复执行
## Four-Group Benchmark Workflow

The repository now includes a four-group benchmark workflow plus a registry-driven family comparison system.

Key files:

- `multirotor/system_config.json`
- `multirotor/benchmark_registry.json`
- `multirotor/Algorithm/four_group_benchmark_runner.py`
- `multirotor/Algorithm/four_group_benchmark_analyzer.py`
- `multirotor/Algorithm/family_analysis.py`
- `multirotor/Algorithm/benchmark_registry_helper.py`

Recommended workflow:

1. Run seeded DDPG and DQN training.
2. Run the four-group frozen benchmark.
3. Generate the four-group benchmark report.
4. Generate family comparison reports.

Seeded training examples:

```powershell
powershell -File .\scripts\Run_Paper_Training_Seeds.ps1 -Algorithm ddpg_apf -Seeds 20260413,20260414,20260415 -StageName stage01 -StageIndex 1
powershell -File .\scripts\Run_Paper_Training_Seeds.ps1 -Algorithm pure_dqn -Seeds 20260413,20260414,20260415 -StageName stage01 -StageIndex 1
```

Four-group frozen benchmark:

```bat
scripts\Run_Four_Group_Benchmark.bat
```

Generate four-group report:

```bat
scripts\Analyze_Four_Group_Benchmark.bat
```

Generate family reports:

```bat
scripts\Analyze_Family_Comparisons.bat
```

Registry helper examples:

```bash
python multirotor/Algorithm/benchmark_registry_helper.py validate
python multirotor/Algorithm/benchmark_registry_helper.py recommend --algorithm-type ppo_scan --control-mode dqn --trainable
python multirotor/Algorithm/benchmark_registry_helper.py scaffold --algorithm-type ppo_scan --control-mode dqn --trainable
```

## 新增工作流能力

当前版本新增了两条“实验工作流”主线，用于把训练、归档、分析和继续训练建议串成统一入口。

快速配置补充说明：

- `start.bat` 中支持快速配置的入口，默认值来源与仿真时间预估口径见 [docs/START_QUICK_CONFIG_ZH.md](docs/START_QUICK_CONFIG_ZH.md)
- `start.bat` / `start_en.bat` 默认以“用户模式”输出运行日志，可在主菜单按 `T` 临时切到详细模式

### 1. 四组统一仿真对比阶段

入口：
- `start.bat` 中的 `M`（四组统一仿真对比阶段）
- `python multirotor/Algorithm/paper_workflow_orchestrator.py --workflow comparison`

用途：
- 作为四组论文实验的总入口，统一衔接训练、四组仿真评测和分析
- 组织 `ddpg_apf` 与 `pure_dqn` 的 stage01 训练，并为 `fixed_apf`、`random_apf`、`ddpg_apf`、`pure_dqn` 四组评测准备输入
- 自动串联训练、归档、四组评测、分析和继续训练建议

当前流程：
1. 运行 `fixed_apf` 与 `random_apf` 的 APF 基线多轮仿真
2. 运行 `ddpg_apf` stage01 训练
3. 运行 `pure_dqn` stage01 训练
4. 在 Unity/AirSim 中运行四组最终统一仿真评测（冻结策略），覆盖 `fixed_apf`、`random_apf`、`ddpg_apf`、`pure_dqn`
5. 生成 comparison workflow 对应的分析产物
6. 给出是否继续进入 `stage02_finetune` 的建议

补充说明：
- `ddpg_apf` 与 `pure_dqn` 会先完成训练，再进入四组仿真评测
- `fixed_apf` 与 `random_apf` 不参加训练阶段，但会直接进入四组仿真评测

产物目录：
- `analysis_results/workflows/comparison/<experiment_id>/...`

### 2. 虚实两阶段工作流

入口：
- `start.bat` 中的 `N`
- `python multirotor/Algorithm/paper_workflow_orchestrator.py --workflow virtual_real_two_stage`

用途：
- 组织 `sim_pretrain -> real_weighted_refine` 的双阶段实验
- 把虚拟预训练模型和实飞修正模型归到同一个实验容器中

支持模式：
- `online`
- `offline_logs`

命令示例：

```bash
python multirotor/Algorithm/paper_workflow_orchestrator.py --workflow virtual_real_two_stage --refine-mode online --alias demo
python multirotor/Algorithm/paper_workflow_orchestrator.py --workflow virtual_real_two_stage --refine-mode offline_logs --alias demo
```

产物目录：
- `analysis_results/workflows/virtual_real_two_stage/<experiment_id>/...`

## 工作流便捷功能

本次新增的不只是菜单入口，还包括一整套实验组织能力：

- 统一实验容器
  - 每次 workflow 都会创建独立实验目录，便于归档和回看
- 状态文件
  - 使用 `workflow_state.json` 记录当前阶段、步骤状态和推荐结果
- 归档能力
  - 统一归档 comparison workflow 和 virtual-real two-stage workflow 的模型、日志和分析结果
- 两类推荐引擎
  - comparison workflow：给出是否继续 `stage02_finetune`
  - virtual-real two-stage：给出是否继续实飞修正
- CLI 统一入口
  - 通过 `scripts\Run_Paper_Workflow.bat` 或 Python CLI 直接调用
- 失败状态可追踪
  - 对 command failure、archive failure、analysis failure、recommendation failure 都有状态记录
- 回归测试补齐
  - 覆盖 state / archive / recommendation / orchestrator / start.bat 菜单入口

## 工作流文件

新增的核心实现文件：

- `multirotor/Algorithm/paper_workflow_state.py`
- `multirotor/Algorithm/paper_workflow_archive.py`
- `multirotor/Algorithm/paper_workflow_recommendation.py`
- `multirotor/Algorithm/paper_workflow_orchestrator.py`
- `multirotor/Algorithm/paper_two_stage_analysis.py`
- `multirotor/Algorithm/paper_two_stage_recommendation.py`
- `scripts/Run_Paper_Workflow.bat`

对应测试：

- `multirotor/Algorithm/tests/test_paper_workflow_state.py`
- `multirotor/Algorithm/tests/test_paper_workflow_archive.py`
- `multirotor/Algorithm/tests/test_paper_workflow_recommendation.py`
- `multirotor/Algorithm/tests/test_paper_workflow_orchestrator.py`
- `multirotor/Algorithm/tests/test_paper_two_stage_analysis.py`
- `multirotor/Algorithm/tests/test_paper_two_stage_recommendation.py`
- `test_start_bat.py`

## 快速查看 CLI 帮助

```bat
scripts\Run_Paper_Workflow.bat --help
```

当前 CLI 支持：
- `--workflow comparison`
- `--workflow virtual_real_two_stage`
- `--refine-mode {online, offline_logs}`
- `--workspace-root`
- `--alias`
