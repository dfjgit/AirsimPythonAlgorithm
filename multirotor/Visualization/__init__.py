"""
统一可视化模块

提供可扩展的可视化框架,支持:
- 系统运行监控
- 训练过程可视化
- 算法特定可视化

架构:
- BaseVisualizer: 基类,提供公共功能
- PanelManager: 面板管理系统
- BasePanel: 面板基类
- 预置面板库: panels/

可视化器:
- RuntimeVisualizer: 运行时监控可视化器
- DDPGTrainingVisualizer: DDPG训练可视化器
- HierarchicalTrainingVisualizer: 分层DQN训练可视化器
- DQNMovementTrainingVisualizer: DQN移动训练可视化器
"""
from .base_visualizer import BaseVisualizer
from .panel_system import BasePanel, PanelManager
from .runtime_visualizer import RuntimeVisualizer
from .ddpg_training_visualizer import DDPGTrainingVisualizer
from .hierarchical_training_visualizer import HierarchicalTrainingVisualizer
from .dqn_movement_visualizer import DQNMovementTrainingVisualizer

__version__ = "1.0.0"
__all__ = [
    'BaseVisualizer',
    'BasePanel',
    'PanelManager',
    'RuntimeVisualizer',
    'DDPGTrainingVisualizer',
    'HierarchicalTrainingVisualizer',
    'DQNMovementTrainingVisualizer'
]
