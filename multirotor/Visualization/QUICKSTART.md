# 统一可视化架构 - 5分钟快速上手

## 🎯 核心概念

新的可视化架构基于**三个核心组件**:

1. **BaseVisualizer** - 基类,提供公共功能
2. **BasePanel** - 面板基类,显示特定信息
3. **PanelManager** - 面板管理器,组织和布局

## 🚀 场景一: 使用现有可视化器

### 运行时监控

```python
from multirotor.Visualization.runtime_visualizer import RuntimeVisualizer

# 创建并启动
visualizer = RuntimeVisualizer(server=your_algorithm_server)
visualizer.start_visualization()

# 停止
visualizer.stop_visualization()
```

**就这么简单!** 窗口会自动显示:
- 环境网格和无人机
- 环境状态面板
- APF权重面板

## 🛠️ 场景二: 创建自己的可视化器

### Step 1: 继承BaseVisualizer

```python
from multirotor.Visualization.base_visualizer import BaseVisualizer

class MyVisualizer(BaseVisualizer):
    def __init__(self, server, env):
        super().__init__(
            server=server,
            env=env,
            window_title="我的算法可视化"
        )
```

### Step 2: 注册面板

```python
from multirotor.Visualization.panels import (
    EnvironmentPanel,
    TrainingStatsPanel,
    RewardCurvePanel
)

class MyVisualizer(BaseVisualizer):
    # ... __init__ ...
    
    def setup_panels(self):
        """选择你需要的面板"""
        self.panel_manager.register_panel(EnvironmentPanel())
        self.panel_manager.register_panel(TrainingStatsPanel())
        self.panel_manager.register_panel(RewardCurvePanel())
```

### Step 3: 提供数据

```python
class MyVisualizer(BaseVisualizer):
    # ... __init__ ...
    # ... setup_panels ...
    
    def get_visualization_data(self):
        """告诉面板显示什么"""
        return {
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'current_episode_reward': self.reward,
            'reward_history': self.rewards,
            'weights': self.get_weights()
        }
```

### Step 4: 使用

```python
visualizer = MyVisualizer(server, env)
visualizer.start_visualization()
```

**完成!** 只需4步,你就有了一个功能完整的可视化器。

## 🎨 场景三: 创建自定义面板

### Step 1: 继承BasePanel

```python
from multirotor.Visualization.panel_system import BasePanel
import pygame

class MyCustomPanel(BasePanel):
    def __init__(self):
        super().__init__(
            name="my_panel",  # 唯一标识
            width=350,        # 面板宽度
            height=200        # 面板高度
        )
```

### Step 2: 实现draw方法

```python
class MyCustomPanel(BasePanel):
    # ... __init__ ...
    
    def draw(self, screen: pygame.Surface, data: dict):
        """绘制面板内容"""
        # 1. 初始化字体
        self._init_fonts()
        
        # 2. 绘制背景和边框
        self.draw_panel_background(screen, border_color=self.CYAN)
        
        # 3. 绘制标题
        y_offset = self.draw_title(screen, "我的面板", self.CYAN)
        
        # 4. 绘制内容
        text_x = self.x + 15
        y = self.y + y_offset
        
        my_value = data.get('my_metric', 0)
        text = self._font.render(f"指标: {my_value:.2f}", True, self.WHITE)
        screen.blit(text, (text_x, y))
```

### Step 3: 使用自定义面板

```python
class MyVisualizer(BaseVisualizer):
    def setup_panels(self):
        self.panel_manager.register_panel(MyCustomPanel())
    
    def get_visualization_data(self):
        return {
            'my_metric': self.calculate_my_metric()
        }
```

**就是这样!** 你的自定义面板会自动:
- 布局在右侧
- 有统一的样式
- 接收可视化数据

## 📦 可用的预置面板

| 面板类 | 用途 | 需要的数据键 |
|-------|------|-------------|
| `EnvironmentPanel` | 环境状态 | `grid_data`, `runtime_data` |
| `TrainingStatsPanel` | 训练统计 | `episode_count`, `total_steps`, `current_episode_reward` |
| `RewardCurvePanel` | 奖励曲线 | `reward_history` |
| `WeightPanel` | APF权重 | `weights`, `use_dqn` |

## 🎯 常用数据键

在`get_visualization_data()`中返回以下键,面板会自动显示:

```python
{
    # 环境数据(BaseVisualizer自动提供)
    'grid_data': grid_data,
    'runtime_data': runtime_data_dict,
    
    # 训练数据
    'episode_count': 100,
    'total_steps': 5000,
    'current_episode_steps': 50,
    'current_episode_reward': 123.45,
    'steps_per_sec': 10.5,
    
    # 奖励数据
    'reward_history': [100, 120, 115, ...],
    'avg_reward': 115.0,
    'max_reward': 150.0,
    'min_reward': 80.0,
    
    # 权重数据
    'weights': {
        'repulsionCoefficient': 1.5,
        'entropyCoefficient': 2.0,
        ...
    },
    'use_dqn': True,
    
    # 自定义数据
    'my_metric': 42.0
}
```

## 💡 最佳实践

### 1. 面板组合

```python
def setup_panels(self):
    # 通用面板
    self.panel_manager.register_panel(EnvironmentPanel())
    
    # 根据模式添加
    if self.training_mode:
        self.panel_manager.register_panel(TrainingStatsPanel())
        self.panel_manager.register_panel(RewardCurvePanel())
    
    # 算法特定面板
    self.panel_manager.register_panel(MyAlgorithmPanel())
```

### 2. 数据更新

```python
def update_training_stats(self, reward, done):
    """在训练循环中调用"""
    self.current_episode_reward += reward
    if done:
        self.reward_history.append(self.current_episode_reward)
        self.episode_count += 1
        self.current_episode_reward = 0
```

### 3. 运行时控制

```python
# 启动
visualizer.start_visualization()

# 隐藏某个面板
visualizer.panel_manager.set_panel_visibility('training_stats', False)

# 动态添加面板
visualizer.panel_manager.register_panel(NewPanel())

# 停止
visualizer.stop_visualization()
```

## 🐛 常见问题

### Q: 窗口不显示?

**A**: 检查pygame是否安装:
```bash
pip install pygame
```

### Q: 面板显示不全?

**A**: 调整面板高度:
```python
class MyPanel(BasePanel):
    def __init__(self):
        super().__init__("my_panel", width=350, height=250)  # 增加高度
```

### Q: 数据不更新?

**A**: 确保在`get_visualization_data()`中返回最新数据:
```python
def get_visualization_data(self):
    return {
        'my_value': self.my_value  # 确保是实例变量,会实时更新
    }
```

### Q: 如何调试?

**A**: 在面板的`draw`方法中添加print:
```python
def draw(self, screen, data):
    print(f"Panel data: {data}")  # 查看接收到的数据
    ...
```

## 📚 更多资源

- [详细架构设计](./ARCHITECTURE.md)
- [完整文档](./README.md)
- [测试示例](./test_visualizer.py)

## 🎉 开始使用吧!

选择一个场景:
- ✅ **场景一**: 使用现有可视化器 → 1分钟上手
- ✅ **场景二**: 创建自己的可视化器 → 5分钟完成
- ✅ **场景三**: 创建自定义面板 → 10分钟搞定

**快速开始,逐步深入!**
