# 分层训练可视化功能使用指南

## 功能概述

为分层强化学习(Hierarchical DQN)训练添加了实时可视化功能，可以直观展示：

1. **5×5任务区域划分** - 高层DQN的动作空间（25个离散区域）
2. **无人机任务目标** - 每架无人机当前的高层目标位置
3. **熵值热力图** - 环境网格的熵值分布（扫描进度）
4. **高层决策历史** - 最近的高层动作决策记录
5. **训练统计信息** - Episode、步数、奖励等
6. **奖励曲线** - 实时奖励变化趋势

## 使用方法

### 1. 离线训练模式（Mock环境）

```bash
# 启用可视化（默认）
python train_hierarchical_dqn.py

# 禁用可视化
python train_hierarchical_dqn.py --no-visualization
```

### 2. AirSim集成训练模式

```bash
# 启用可视化（默认）
python train_hierarchical_with_airsim.py

# 禁用可视化
python train_hierarchical_with_airsim.py --no-visualization
```

### 3. 测试可视化功能

```bash
# 运行测试脚本（模拟训练数据）
python test_hierarchical_visualization.py
```

## 可视化窗口说明

### 主视图（左侧）
- **黑色背景**: 环境空间
- **彩色点**: 网格单元（绿色=已扫描，红色=未扫描）
- **灰色网格**: 5×5任务区域划分
- **浅蓝色圆圈**: Leader位置和扫描范围
- **彩色标记**: 无人机位置（每个无人机不同颜色）
- **十字标记**: 高层目标位置
- **连线**: 无人机到目标的路径

### 右侧面板
1. **Training Statistics** - 训练统计信息
   - Episode数
   - 总步数
   - Episode奖励
   - 扫描进度

2. **High-Level Actions History** - 高层决策历史
   - 最近10个决策
   - 格式: [步数] 无人机: 动作ID (区域位置)

3. **Reward History** - 奖励曲线
   - 最近200步的奖励变化

4. **Instructions** - 操作说明

## 区域编号规则

5×5网格的动作空间编号（0-24）：
```
 0   1   2   3   4
 5   6   7   8   9
10  11  12  13  14
15  16  17  18  19
20  21  22  23  24
```

每个动作对应一个区域中心位置，以Leader为中心展开。

## 技术细节

### HierarchicalVisualizer类

位置: `multirotor/DQN_Movement/hierarchical_visualizer.py`

主要方法：
- `start_visualization()` - 启动可视化线程
- `stop_visualization()` - 停止可视化
- `update_training_data(step, action, reward, drone_name)` - 更新训练数据
- `on_episode_end(episode)` - Episode结束回调

### VisualizationCallback类

在训练脚本中定义，用于在训练过程中自动更新可视化数据。

每个训练步骤自动调用：
1. 获取当前动作和奖励
2. 更新可视化数据
3. 检测Episode结束

## 性能优化

- 窗口刷新率: 30 FPS
- 熵值点绘制: 限制500个以内
- 历史记录: 
  - 高层动作历史: 100条
  - 奖励历史: 200条

## 依赖要求

```bash
pip install pygame
```

如果pygame未安装，训练仍可正常进行，但不显示可视化。

## 快捷键

- **ESC** - 关闭可视化窗口

## 常见问题

### Q: 可视化窗口卡顿？
A: 这是正常的，因为训练和可视化在不同线程运行。可以使用`--no-visualization`参数禁用可视化以提升训练速度。

### Q: 看不到无人机？
A: 确保AirSim服务器正常运行，或在离线模式下环境已正确初始化。

### Q: 区域划分显示不清晰？
A: 可以调整`HierarchicalVisualizer`类中的`scale`参数来改变显示比例。

## 与现有可视化的区别

| 功能 | SimpleVisualizer | HierarchicalVisualizer |
|------|------------------|------------------------|
| 适用场景 | 常规APF/DDPG训练 | 分层DQN训练 |
| 区域划分 | 无 | 5×5任务区域 |
| 高层目标 | 无 | 显示HL目标 |
| 决策历史 | 无 | 高层动作历史 |
| 奖励曲线 | 无 | 实时奖励曲线 |

## 扩展建议

如需添加更多可视化内容，可在`HierarchicalVisualizer`类中添加方法：

1. 在`draw_*`方法中实现新的绘制逻辑
2. 在`run()`主循环中调用新方法
3. 通过`update_training_data()`或新方法更新数据

## 参考文件

- `hierarchical_visualizer.py` - 可视化器实现
- `train_hierarchical_dqn.py` - 离线训练脚本
- `train_hierarchical_with_airsim.py` - AirSim集成训练脚本
- `test_hierarchical_visualization.py` - 测试脚本
- `hierarchical_movement_env.py` - 分层环境实现
