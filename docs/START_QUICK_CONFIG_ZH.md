# Start 快速配置说明

## 1. 这套快速配置是做什么的

`start.bat` 中的部分入口支持“快速配置”。

特点：

- 只询问当前入口真正需要的参数
- 直接回车即可沿用默认值
- 输入自定义数值只影响本次执行，不会直接改写配置文件

## 2. 默认值从哪里来

### 无人机数量

默认值来源：

- `multirotor/DDPG_Weight/configs/unified_train_config.json`
- 路径：`airsim_virtual.drone_names`
- 取值方式：读取列表长度

### DDPG 训练步数

默认值来源：

- `multirotor/DDPG_Weight/configs/unified_train_config.json`
- 路径：`airsim_virtual.total_timesteps`

### DQN 训练步数

默认值来源：

- `multirotor/DQN_Movement/configs/movement_dqn_config.json`
- 路径：`training.total_timesteps`

### DQN 续训总步数

默认值来源：

- `multirotor/DQN_Movement/configs/movement_dqn_config.json`
- 路径：`training.resume_total_timesteps`

### 分层 DQN 训练步数

默认值来源：

- `multirotor/DQN_Movement/configs/hierarchical_dqn_config.json`
- 路径：`training.total_timesteps`

### APF 基线多轮仿真轮次

默认值来源：

- `multirotor/system_config.json`
- 路径：`paper_benchmark.eval_episodes_per_seed`

说明：

- 这个值用于 `fixed APF` 与 `random APF` 的基线多轮仿真阶段
- 目的是生成足够的 episode 级过程数据，便于与学习算法一起做过程对比

### 四组 benchmark 每 seed 评测轮次

默认值来源：

- `multirotor/system_config.json`
- 路径：`paper_benchmark.eval_episodes_per_seed`

说明：

- 这个值用于四组最终统一仿真对比阶段
- 作用是控制每个 `seed` 下每组策略的评测回合数

### Seed 列表

默认值来源：

- `multirotor/system_config.json`
- 路径：`paper_benchmark.seeds`

### 仿真可视化窗口

默认值来源：

- 内置默认值：`开启`

说明：

- 这个开关用于控制 APF 基线多轮仿真、训练阶段以及四组最终统一仿真评测是否弹出可视化窗口
- 它只影响本次执行，不会直接改写配置文件

## 3. 运行时日志模式与快速配置的关系

`start.bat` / `start_en.bat` 启动后，默认会把运行时日志模式设为“用户模式”。

这意味着：

- 控制台优先保留关键阶段状态、连接状态和错误信息
- 高频调试输出仍会完整写入 `analysis_results/runtime_logs/`
- 如需临时查看更细的实时调试信息，可在主菜单按 `T` 切到“详细模式”

说明：

- 运行时日志模式不是快速配置字段
- 它是主菜单级别的当前会话开关
- 重新打开 `start.bat` 后会恢复默认的“用户模式”

## 4. 仿真时间预估是什么意思

在 `DDPG` / `DQN` 相关字段里，快速配置会显示：

- `约 X 小时仿真时间`

这不是机器上的真实墙钟耗时，而是按默认仿真步长换算出来的**仿真口径预估**。

当前口径：

- `DDPG+APF`：按 `2.0 秒 / step` 估算
- `Pure DQN`：按 `1.5 秒 / step` 估算

真实耗时会受到以下因素影响：

- 机器性能
- 是否启用可视化
- Unity / AirSim 当前状态
- 日志与分析开销

因此：

- 仿真时间预估适合用来理解实验规模
- 不应把它当作精确的实际运行时长承诺

## 5. 四组统一仿真对比阶段里的两个“轮次”有什么区别

### APF 基线多轮仿真轮次

作用：

- 只用于 `fixed APF` 和 `random APF`
- 在最终 benchmark 之前先跑一段多轮仿真
- 主要生成过程对比所需的数据

### 四组 benchmark 每 seed 评测轮次

作用：

- 用于最终四组统一仿真对比
- 覆盖：
  - `fixed APF`
  - `random APF`
  - `DDPG+APF`
  - `Pure DQN`
- 主要生成最终 boxplot / bar / family 分析所需的数据

## 6. 当前推荐理解方式

如果你在 `M` 入口中看到这两个值，可以这样理解：

- `APF 基线多轮仿真轮次`：补足 APF 系列过程数据
- `四组 benchmark 每 seed 评测轮次`：控制最终统一对比样本量

## 7. 四组统一仿真对比阶段的执行顺序

`M` 入口当前按以下顺序组织实验：

1. `fixed APF` 基线多轮仿真
2. `random APF` 基线多轮仿真
3. `DDPG+APF` stage01 训练
4. `Pure DQN` stage01 训练
5. 四组最终统一仿真对比
6. 对比分析与 `stage02` 建议

说明：

- `fixed APF` 与 `random APF` 不进入训练阶段
- 它们会先完成多轮仿真，用于生成过程对比数据
- 之后四组都会进入最终统一仿真对比阶段

## 8. 无人机数量快速配置

当前以下工作流入口支持“无人机数量”快速配置，默认值为 `3`：

- `M` 四组统一仿真对比阶段
- `N` 虚实两阶段实验工作流

同时，系统运行入口中也支持无人机数量快速配置：

- 固定权重运行
- DDPG 权重预测运行
