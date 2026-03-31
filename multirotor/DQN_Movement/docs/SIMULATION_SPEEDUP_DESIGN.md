# 仿真倍速功能设计与后续产出说明书

## 1. 目标

本说明书用于回答两个问题：

1. 当前项目是否值得接入“仿真倍速”能力。
2. 如果要做，推荐采用哪种设计、改动范围多大、最终会产出什么成果。

这里的“仿真倍速”指的是：

- 在尽量不改变训练语义的前提下
- 缩短真实墙钟训练时间
- 让 DDPG / DQN 的调参、回归验证、预训练更快完成


## 2. 设计判断结论

### 2.1 结论

**建议优先接入 `Unity timeScale + Python 等待缩放` 方案，不建议第一步就直接上 `AirSim ClockSpeed`。**

### 2.2 原因

当前项目的训练节奏，不只是由 AirSim 决定，还同时依赖：

- Unity 侧 leader 移动
- Unity 侧网格扫描/熵值更新
- Python 环境中的 `sleep(step_duration)`
- `AlgorithmServer` 控制循环中的 `updateInterval`

如果只改 AirSim 的 `ClockSpeed`，会出现一个明显问题：

- AirSim 物理可能变快了
- 但 Unity leader 和网格更新未必同步变快
- Python 侧仍然在按真实时间 `sleep`

这样反而容易造成：

- 训练节奏错位
- leader / UAV / 网格状态不同步
- reset、done、奖励统计出现偏差

而 `Unity timeScale + Python 等待缩放` 更贴合当前工程结构：

- Unity leader 和网格本来就由 Unity 驱动
- Python 训练等待本来就是显式 `sleep`
- 只要两边按同一个倍速因子一起缩放，行为更容易保持一致


## 3. 当前系统现状判断

### 3.1 当前还没有真正的仿真倍速能力

当前代码里没有已经接好的统一倍速入口：

- 没有现成的 `unity_time_scale`
- 没有现成的 `simulation_speed`
- 也没有训练链中统一使用的 `AirSim ClockSpeed`

### 3.2 当前墙钟时间主要消耗在哪里

#### DDPG

DDPG 墙钟时间主要由以下两处决定：

- [unified_train_config.json](D:/Work/Python%20Project/airsim-python-algorithm/multirotor/DDPG_Weight/configs/unified_train_config.json)
  - `step_duration = 2.0`
- [simple_weight_env.py](D:/Work/Python%20Project/airsim-python-algorithm/multirotor/DDPG_Weight/envs/simple_weight_env.py)
  - 每步真实执行 `time.sleep(self.step_duration)`

#### DQN

DQN 墙钟时间主要由以下两处决定：

- [movement_dqn_config.json](D:/Work/Python%20Project/airsim-python-algorithm/multirotor/DQN_Movement/configs/movement_dqn_config.json)
  - `step_duration = 0.5`
  - `action_repeat = 3`
- [movement_env.py](D:/Work/Python%20Project/airsim-python-algorithm/multirotor/DQN_Movement/envs/movement_env.py)
  - 每个物理子步真实执行 `time.sleep(self.step_duration)`

### 3.3 当前更适合先做“倍速模式”，不建议直接拿来做正式实验

仿真倍速对项目意义很大，但更适合作为：

- 开发调试提效工具
- 超参调试工具
- 预训练/预验证工具

不建议一接上就直接作为“正式实验默认模式”，原因是正式实验更强调：

- 物理语义稳定
- 结果可复现
- DDPG / DQN 口径一致


## 4. 推荐设计方案

## 4.1 总体方案

采用一个统一配置字段：

- `simulation_speed`

默认值：

- `1.0`

含义：

- `1.0` 表示常速
- `2.0` 表示 2 倍速
- `4.0` 表示 4 倍速

### 4.2 第一阶段实施方式

第一阶段不碰 AirSim 底层时钟，只做：

1. Unity 配置下发一个倍速因子
2. Python 侧等待时间按同一倍速缩短

也就是：

- Unity 运动 / 网格更新加快
- Python `sleep` 同步变短

### 4.3 第二阶段可选增强

如果第一阶段验证效果稳定，再考虑评估：

- `AirSim ClockSpeed`

但不建议一开始就把它作为主方案。


## 5. 需要改动的模块

### 5.1 配置层

建议在统一配置对象中新增：

- `simulation_speed: float = 1.0`

涉及文件：

- [scanner_config_data.py](D:/Work/Python%20Project/airsim-python-algorithm/multirotor/Algorithm/scanner_config_data.py)
- [apf_algorithm_config.json](D:/Work/Python%20Project/airsim-python-algorithm/multirotor/apf_algorithm_config.json)
- [movement_dqn_config.json](D:/Work/Python%20Project/airsim-python-algorithm/multirotor/DQN_Movement/configs/movement_dqn_config.json)
- [unified_train_config.json](D:/Work/Python%20Project/airsim-python-algorithm/multirotor/DDPG_Weight/configs/unified_train_config.json)

### 5.2 Unity 配置下发链

当前 [unity_socket_server.py](D:/Work/Python%20Project/airsim-python-algorithm/multirotor/AirsimServer/unity_socket_server.py) 已有 `send_config(...)`，因此推荐：

- 不新增独立命令
- 直接把 `simulation_speed` 放进现有 config 包

涉及文件：

- [unity_socket_server.py](D:/Work/Python%20Project/airsim-python-algorithm/multirotor/AirsimServer/unity_socket_server.py)
- [scanner_config_data.py](D:/Work/Python%20Project/airsim-python-algorithm/multirotor/Algorithm/scanner_config_data.py)

### 5.3 AlgorithmServer 控制节奏

当前 [AlgorithmServer.py](D:/Work/Python%20Project/airsim-python-algorithm/multirotor/AlgorithmServer.py) 会按 `updateInterval` 真实 `sleep`。

建议改成：

- `effective_update_interval = updateInterval / simulation_speed`

但以下等待建议保守处理：

- reset 后固定等待
- 起飞确认等待
- home restore 验证等待

这些建议使用：

- 逆比例缩放
- 再加一个最小保护下限

例如：

- `effective_wait = max(base_wait / simulation_speed, min_wait)`

### 5.4 DDPG 环境

当前 [simple_weight_env.py](D:/Work/Python%20Project/airsim-python-algorithm/multirotor/DDPG_Weight/envs/simple_weight_env.py) 中：

- `time.sleep(self.step_duration)`

建议改成：

- `logical_step_duration = self.step_duration`
- `effective_sleep = logical_step_duration / simulation_speed`

并且训练日志里的“逻辑时长”仍保留 `logical_step_duration`，不要直接用墙钟时间替代。

### 5.5 DQN 环境

当前 [movement_env.py](D:/Work/Python%20Project/airsim-python-algorithm/multirotor/DQN_Movement/envs/movement_env.py) 中：

- 单机、多机都按 `step_duration` 和 `action_repeat` 做真实 sleep

建议同样改成：

- 逻辑步长不变
- 真实等待按倍速缩短


## 6. 建议增加的配置项

建议新增以下字段：

### 6.1 核心字段

- `simulation_speed`
  - 类型：`float`
  - 默认：`1.0`
  - 说明：训练/仿真倍速因子

### 6.2 安全保护字段

- `min_effective_sleep_sec`
  - 默认：`0.05`
  - 说明：避免倍速后 sleep 过短导致 CPU 空转或控制抖动

- `reset_wait_scale_cap`
  - 默认：`2.0`
  - 说明：reset 相关等待最多只缩到一半，避免过度激进

### 6.3 模式字段

- `simulation_speed_mode`
  - 可选：`formal` / `fast_train`
  - 默认：`formal`
  - 说明：
    - `formal`：只允许 `1.0`
    - `fast_train`：允许 `>1.0`


## 7. 风险与注意事项

### 7.1 reset 风险

倍速后最容易先出问题的是：

- reset 完成判定
- 起飞确认
- home restore
- Unity runtime 首帧同步

这些都依赖“等一小段时间再确认”的逻辑。

### 7.2 训练语义风险

如果只是单纯把等待缩短，而 Unity / AirSim 实际没有同步变快，会导致：

- leader 行为未变
- UAV 控制频率变快
- 奖励与物理时间意义改变

因此必须确保：

- Unity 运动和网格逻辑确实吃到了倍速
- Python 等待缩放和 Unity 倍速使用同一个因子

### 7.3 正式实验风险

在没有完成验证前，不建议把倍速模式直接用于：

- DDPG vs DQN 正式对比实验
- 最终报告主结果


## 8. 验证方案

建议按以下顺序验证：

### 8.1 功能验证

1. `simulation_speed = 1.0`
   - 行为与当前版本一致
2. `simulation_speed = 2.0`
   - 训练墙钟时间明显下降
   - 轨迹、reward、reset 不出现明显异常

### 8.2 关键检查项

验证以下内容是否仍然正常：

- leader 轨迹是否连续
- 无人机起飞/重置是否稳定
- reset 原因分布是否异常恶化
- 碰撞率是否突然飙升
- battery 电压下降曲线是否合理
- global_scan_ratio 是否仍有可解释性

### 8.3 回归对比

同一份配置下分别跑：

- `1.0x`
- `2.0x`

对比：

- reset 原因占比
- 平均 episode 长度
- global_scan_ratio
- global_avg_entropy
- collision_count

如果两边趋势接近，则说明倍速可用。


## 9. 推荐实施节奏

### 阶段 A：设计验证

只实现最小可用版：

- config 增加 `simulation_speed`
- Unity 吃到该参数
- Python 训练等待按倍速缩放

目标：

- 跑通 `2.0x`

### 阶段 B：开发提效

把倍速模式用于：

- 调参
- reward 调试
- reset 链回归测试

### 阶段 C：正式实验前验证

决定是否允许倍速进入正式实验：

- 若验证通过，可单独列出“倍速实验结果”
- 若验证未完全通过，则正式实验仍保持 `1.0x`


## 10. 对项目的意义评估

### 10.1 意义大的地方

1. 明显减少开发等待时间
2. 更快做 DDPG / DQN 参数扫描
3. 更快做 reset / reward / done 条件回归测试

### 10.2 不宜夸大的地方

1. 它不保证算法更容易收敛
2. 它主要减少的是墙钟时间，不是样本复杂度
3. 如果验证不充分，可能会损伤正式实验可信度


## 11. 建议是否现在就做

### 建议

**可以做，但建议作为“开发提效功能”先接入，不建议现在直接替代正式实验常速模式。**

也就是说，推荐策略是：

- 现在先完成设计与接入
- 先用来做调试和预训练
- 正式对比实验仍保留 `1.0x`
- 等倍速模式验证稳定后，再决定是否将其纳入正式实验体系


## 12. 本功能完成后的预期产出

如果现在开始实现，建议最终产出这些成果：

### 12.1 代码成果

1. 统一配置字段 `simulation_speed`
2. Unity 配置下发链支持倍速
3. DDPG / DQN 环境等待缩放
4. AlgorithmServer 控制循环等待缩放
5. reset 安全等待保护

### 12.2 文档成果

1. 倍速功能使用说明
2. 正式实验模式 vs 快速训练模式说明
3. 推荐倍速范围说明（例如 `1.0x / 2.0x / 4.0x`）

### 12.3 分析成果

1. `1.0x` 与 `2.0x` 的对比表
2. 关键稳定性指标对比图
3. 是否建议纳入正式实验的结论


## 13. 最终建议一句话版

**推荐做，而且优先采用“Unity timeScale + Python 等待缩放”的最小可用设计；它对开发提效意义很大，但在完成验证前，不建议直接替代正式实验常速模式。**
