# 统一可视化模块

> 可扩展的可视化底座,支持多种算法和训练场景

## ✨ 特性

- 🎨 **统一风格**: 所有可视化器共享一致的UI风格
- 🔌 **可插拔面板**: 按需注册面板,灵活组合
- 🚀 **易于扩展**: 添加新算法只需继承基类
- 💾 **代码复用**: 公共功能不再重复实现
- 🧵 **线程安全**: 内置线程管理和数据缓存

## 📦 快速开始

### 1. 运行时监控

```python
from multirotor.Visualization.runtime_visualizer import RuntimeVisualizer

# 创建可视化器
visualizer = RuntimeVisualizer(server=your_algorithm_server)

# 启动(独立线程)
visualizer.start_visualization()

# 停止
visualizer.stop_visualization()
```

### 2. 创建自定义可视化器

```python
from multirotor.Visualization.base_visualizer import BaseVisualizer
from multirotor.Visualization.panels import EnvironmentPanel, TrainingStatsPanel

class MyVisualizer(BaseVisualizer):
    def __init__(self, server, env):
        super().__init__(server, env, window_title="我的算法")
    
    def setup_panels(self):
        # 注册面板
        self.panel_manager.register_panel(EnvironmentPanel())
        self.panel_manager.register_panel(TrainingStatsPanel())
    
    def get_visualization_data(self):
        # 提供数据
        return {
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
        }

# 使用
visualizer = MyVisualizer(server, env)
visualizer.start_visualization()
```

### 3. 创建自定义面板

```python
from multirotor.Visualization.panel_system import BasePanel
import pygame

class MyPanel(BasePanel):
    def __init__(self):
        super().__init__("my_panel", width=350, height=200)
    
    def draw(self, screen: pygame.Surface, data: dict):
        self._init_fonts()
        self.draw_panel_background(screen, self.CYAN)
        y = self.draw_title(screen, "我的面板", self.CYAN)
        
        # 绘制内容
        text = self._font.render("Hello World", True, self.WHITE)
        screen.blit(text, (self.x + 15, self.y + y))
```

## 📚 文档

- [架构设计](./ARCHITECTURE.md) - 详细架构说明
- [面板开发指南](./PANEL_DEVELOPMENT.md) - 如何开发自定义面板
- [迁移指南](./MIGRATION_GUIDE.md) - 从旧可视化器迁移

## 🎯 预置面板

| 面板 | 说明 | 导入路径 |
|-----|------|---------|
| `EnvironmentPanel` | 环境状态 | `panels.environment_panel` |
| `TrainingStatsPanel` | 训练统计 | `panels.training_stats_panel` |
| `RewardCurvePanel` | 奖励曲线 | `panels.reward_curve_panel` |
| `WeightPanel` | APF权重 | `panels.weight_panel` |

## 🔧 依赖

```bash
pip install pygame
```

## 🤝 兼容性

### 向后兼容

新架构提供别名支持,旧代码无需修改:

```python
# 旧代码仍可正常运行
from multirotor.Algorithm.simple_visualizer import SimpleVisualizer
visualizer = SimpleVisualizer(server)
visualizer.start_visualization()

# 推荐使用新路径
from multirotor.Visualization.runtime_visualizer import RuntimeVisualizer
visualizer = RuntimeVisualizer(server)
visualizer.start_visualization()
```

## 📈 示例

### 示例1: 最小化可视化器

```python
from multirotor.Visualization.base_visualizer import BaseVisualizer

class MinimalVisualizer(BaseVisualizer):
    def setup_panels(self):
        pass  # 不注册面板,只显示环境
    
    def get_visualization_data(self):
        return {}

visualizer = MinimalVisualizer(server)
visualizer.start_visualization()
```

### 示例2: 组合多个面板

```python
from multirotor.Visualization.runtime_visualizer import RuntimeVisualizer
from multirotor.Visualization.panels import *

visualizer = RuntimeVisualizer(server)
visualizer.start_visualization()

# 动态添加面板
from multirotor.Visualization.panels.reward_curve_panel import RewardCurvePanel
visualizer.panel_manager.register_panel(RewardCurvePanel())
```

## 🐛 调试

设置环境变量启用调试输出:

```bash
export VISUALIZER_DEBUG=1
python your_script.py
```

## 📝 版本

- v1.0.0 (2026-02-03)
  - 初始发布
  - 基础框架和预置面板
  - 运行时可视化器

## 🙏 致谢

本模块重构自以下可视化器:
- `Algorithm/simple_visualizer.py`
- `DDPG_Weight/training_visualizer.py`
- `DQN_Movement/visualizers/hierarchical_visualizer.py`

## 📧 反馈

如有问题或建议,请提Issue或联系项目维护者。
