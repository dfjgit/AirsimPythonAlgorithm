# 统一可视化架构说明

## 🎯 设计目标

将原有的三个独立可视化器(`SimpleVisualizer`、`TrainingVisualizer`、`HierarchicalVisualizer`)重构为一个**统一的、可扩展的可视化底座**,解决以下问题:

1. **代码冗余**: 三个可视化器有大量重复代码(网格绘制、无人机绘制、面板布局等)
2. **维护困难**: 修改公共功能需要同步三份代码
3. **扩展性差**: 添加新算法需要从头编写可视化器
4. **风格不统一**: 三个界面风格各异,缺乏统一规范

## 📐 架构设计

### 核心组件

```
multirotor/Visualization/
├── __init__.py                    # 模块入口
├── base_visualizer.py             # 基类: 公共绘制方法、线程管理
├── panel_system.py                # 面板系统: BasePanel + PanelManager
├── panels/                        # 预置面板库
│   ├── environment_panel.py       # 环境信息
│   ├── training_stats_panel.py    # 训练统计
│   ├── reward_curve_panel.py      # 奖励曲线
│   └── weight_panel.py            # APF权重
├── runtime_visualizer.py          # 运行监控可视化器
├── ddpg_training_visualizer.py    # DDPG训练可视化器(待实现)
└── hierarchical_training_visualizer.py  # 分层DQN训练可视化器(待实现)
```

### 设计模式

#### 1. **模板方法模式** (Template Method)

`BaseVisualizer`定义可视化流程骨架:

```python
class BaseVisualizer(ABC):
    def run(self):
        # 初始化
        self.pygame_init()
        self.setup_panels()  # 子类实现
        
        # 主循环
        while self.running:
            self.update_data()
            self.draw_environment()  # 公共方法
            vis_data = self.get_visualization_data()  # 子类实现
            self.panel_manager.draw_all_panels(vis_data)
    
    @abstractmethod
    def setup_panels(self):
        """子类注册所需面板"""
        pass
    
    @abstractmethod
    def get_visualization_data(self) -> Dict:
        """子类提供数据"""
        pass
```

#### 2. **策略模式** (Strategy)

面板系统允许动态注册不同的显示策略:

```python
# 运行时监控: 只需环境+权重面板
visualizer = RuntimeVisualizer(server)
visualizer.panel_manager.register_panel(EnvironmentPanel())
visualizer.panel_manager.register_panel(WeightPanel())

# DDPG训练: 需要训练统计+奖励曲线+权重历史
visualizer = DDPGTrainingVisualizer(server, env)
visualizer.panel_manager.register_panel(TrainingStatsPanel())
visualizer.panel_manager.register_panel(RewardCurvePanel())
visualizer.panel_manager.register_panel(WeightHistoryPanel())
```

#### 3. **组合模式** (Composite)

面板管理器统一管理多个面板:

```python
class PanelManager:
    def __init__(self):
        self.panels: Dict[str, BasePanel] = {}
        self.panel_order: List[str] = []
    
    def register_panel(self, panel: BasePanel):
        """注册面板"""
        self.panels[panel.name] = panel
    
    def draw_all_panels(self, data: Dict):
        """统一绘制"""
        for name in self.panel_order:
            self.panels[name].draw(screen, data)
```

## 🔧 使用指南

### 1. 创建新的可视化器

只需继承`BaseVisualizer`并实现两个方法:

```python
from multirotor.Visualization.base_visualizer import BaseVisualizer
from multirotor.Visualization.panels import EnvironmentPanel, TrainingStatsPanel

class MyCustomVisualizer(BaseVisualizer):
    def __init__(self, server, env):
        super().__init__(
            server=server,
            env=env,
            window_title="我的算法可视化"
        )
    
    def setup_panels(self):
        """注册所需面板"""
        self.panel_manager.register_panel(EnvironmentPanel())
        self.panel_manager.register_panel(TrainingStatsPanel())
        # 可以添加自定义面板
        self.panel_manager.register_panel(MyCustomPanel())
    
    def get_visualization_data(self) -> Dict:
        """提供可视化数据"""
        return {
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'weights': self.get_current_weights(),
            # 自定义数据
            'my_metric': self.calculate_metric()
        }
```

### 2. 创建自定义面板

继承`BasePanel`并实现`draw`方法:

```python
from multirotor.Visualization.panel_system import BasePanel
import pygame

class MyCustomPanel(BasePanel):
    def __init__(self):
        super().__init__(name="my_panel", width=350, height=200)
    
    def draw(self, screen: pygame.Surface, data: Dict):
        """绘制面板内容"""
        self._init_fonts()
        
        # 绘制背景
        self.draw_panel_background(screen, border_color=self.CYAN)
        
        # 绘制标题
        y_offset = self.draw_title(screen, "我的面板", self.CYAN)
        
        # 绘制内容
        text_x = self.x + 15
        y = self.y + y_offset
        
        my_value = data.get('my_metric', 0)
        text = self._font.render(f"指标: {my_value:.2f}", True, self.WHITE)
        screen.blit(text, (text_x, y))
```

### 3. 替换旧的可视化器

```python
# 旧代码
from multirotor.Algorithm.simple_visualizer import SimpleVisualizer
visualizer = SimpleVisualizer(server)

# 新代码(完全兼容)
from multirotor.Visualization.runtime_visualizer import RuntimeVisualizer
visualizer = RuntimeVisualizer(server)

# 启动方式相同
visualizer.start_visualization()
```

## 🎨 预置面板库

| 面板名称 | 用途 | 适用场景 |
|---------|------|---------|
| `EnvironmentPanel` | 环境状态(网格、扫描进度) | 所有场景 |
| `TrainingStatsPanel` | 训练统计(Episode、步数、奖励) | 训练场景 |
| `RewardCurvePanel` | 奖励曲线(实时趋势) | 训练场景 |
| `WeightPanel` | APF权重显示 | 需要权重的场景 |
| `WeightHistoryPanel` | 权重变化历史 | DDPG权重训练 |
| `HierarchicalGridPanel` | 5x5高层网格 | 分层DQN训练 |
| `ActionHistoryPanel` | 高层决策历史 | 分层DQN训练 |

## 🚀 扩展性

### 添加新算法可视化

只需3步:

1. **创建可视化器类** (继承`BaseVisualizer`)
2. **复用现有面板** + **添加算法特定面板**(如有)
3. **提供数据** (实现`get_visualization_data`)

不需要:
- ❌ 重写环境渲染逻辑
- ❌ 重写面板布局系统
- ❌ 重写线程管理
- ❌ 重写事件处理

### 添加新面板

只需1步:

1. **创建面板类** (继承`BasePanel`,实现`draw`方法)

面板自动享有:
- ✅ 统一的背景和边框样式
- ✅ 自动布局管理
- ✅ 颜色和字体常量
- ✅ 工具方法(`draw_title`等)

## 📊 性能优化

1. **数据缓存**: 基类内置缓存机制,减少锁竞争
2. **延迟初始化**: 字体等资源按需加载
3. **可见性控制**: 面板可动态显示/隐藏
4. **帧率控制**: 统一30 FPS渲染

## 🔗 兼容性

### 向后兼容

旧代码无需修改即可使用新架构:

```python
# 别名支持
from multirotor.Visualization.runtime_visualizer import SimpleVisualizer
```

### 迁移路径

1. **第一阶段**: 新代码使用新架构,旧代码保持不变
2. **第二阶段**: 逐步迁移旧代码到新架构
3. **第三阶段**: 弃用旧可视化器(可选)

## 📝 最佳实践

### 1. 面板命名

使用描述性名称,避免冲突:

```python
# 好
panel = EnvironmentPanel()  # name="environment"
panel = TrainingStatsPanel()  # name="training_stats"

# 不好
panel = BasePanel("panel1")
```

### 2. 数据传递

使用明确的键名:

```python
def get_visualization_data(self):
    return {
        'episode_count': 100,      # 明确
        'total_steps': 5000,       # 明确
        # 避免: 'data': {...}    # 模糊
    }
```

### 3. 面板尺寸

遵循标准尺寸,保持一致性:

```python
# 标准宽度
panel_width = 350  # 或 370

# 高度根据内容
panel_height = 180  # 小面板
panel_height = 250  # 中面板
panel_height = 350  # 大面板
```

### 4. 错误处理

面板绘制失败不应影响其他面板:

```python
def draw(self, screen, data):
    try:
        # 绘制逻辑
        value = data['my_key']
        self._draw_value(value)
    except KeyError:
        # 优雅降级
        self._draw_placeholder(screen)
    except Exception as e:
        print(f"绘制错误: {e}")
```

## 🎯 下一步

- [ ] 完成DDPG训练可视化器迁移
- [ ] 完成分层DQN训练可视化器迁移
- [ ] 添加更多预置面板(电量、通信、路径规划等)
- [ ] 支持自定义主题/配色
- [ ] 支持实时配置调整(不重启窗口)
- [ ] 支持录制和回放功能

## 📚 参考

- 旧可视化器:
  - `multirotor/Algorithm/simple_visualizer.py`
  - `multirotor/DDPG_Weight/training_visualizer.py`
  - `multirotor/DQN_Movement/visualizers/hierarchical_visualizer.py`
  
- 设计模式参考:
  - [Template Method Pattern](https://refactoring.guru/design-patterns/template-method)
  - [Strategy Pattern](https://refactoring.guru/design-patterns/strategy)
  - [Composite Pattern](https://refactoring.guru/design-patterns/composite)
