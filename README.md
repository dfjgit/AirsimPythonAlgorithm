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

## Workflow Entry Points

The batch launcher now exposes two workflow-oriented experiment entries:

- `M` / comparison workflow
  - Runs the stage01 comparison stack for `ddpg_apf` and `pure_dqn`
  - Archives outputs under `analysis_results/workflows/comparison/...`
  - Produces comparison recommendations for whether to continue to `stage02_finetune`
- `N` / virtual-real two-stage workflow
  - Runs `sim_pretrain -> real_weighted_refine`
  - Supports `online` and `offline_logs` refine modes
  - Archives outputs under `analysis_results/workflows/virtual_real_two_stage/...`

The wrapper script is:

```bat
scripts\Run_Paper_Workflow.bat --help
```

Direct CLI examples:

```bash
python multirotor/Algorithm/paper_workflow_orchestrator.py --workflow comparison --alias demo
python multirotor/Algorithm/paper_workflow_orchestrator.py --workflow virtual_real_two_stage --refine-mode online --alias demo
python multirotor/Algorithm/paper_workflow_orchestrator.py --workflow virtual_real_two_stage --refine-mode offline_logs --alias demo
```
