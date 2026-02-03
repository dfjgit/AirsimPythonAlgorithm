"""
分层DQN训练可视化器 - 用于分层强化学习训练

替代原有的HierarchicalVisualizer,使用新的可视化底座架构
支持:
- 5x5高层任务区域显示
- 高层决策历史
- 训练统计
- 奖励曲线
- 熵值统计
"""
import sys
import os
from typing import Dict, Any
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from multirotor.Visualization.base_visualizer import BaseVisualizer
from multirotor.Visualization.panels.environment_panel import EnvironmentPanel
from multirotor.Visualization.panels.training_stats_panel import TrainingStatsPanel
from multirotor.Visualization.panels.reward_curve_panel import RewardCurvePanel
from multirotor.Visualization.panels.hierarchical_grid_panel import HierarchicalGridPanel
from multirotor.Visualization.panels.battery_panel import BatteryPanel


class HierarchicalTrainingVisualizer(BaseVisualizer):
    """
    分层DQN训练可视化器
    
    用于分层强化学习训练,显示:
    - 环境状态(熵值热力图)
    - 5x5高层任务区域
    - 高层决策历史
    - 训练统计
    - 奖励曲线
    """
    
    def __init__(self, env, server=None):
        super().__init__(
            server=server,
            env=env,
            window_title="🎯 Hierarchical DQN Training Visualization"
        )
        
        # 数据缓存
        self.hl_action_history = deque(maxlen=100)  # 高层动作历史
        self.hl_goal_history = deque(maxlen=100)  # 高层目标历史
        self.reward_history = deque(maxlen=200)  # 奖励历史
        self.drone_colors = {}  # 无人机颜色映射
        
        # 统计信息
        self.episode_count = 0
        self.total_steps = 0
        self.current_episode_reward = 0
        self._entropy_stats = {}  # 熵值统计
    
    def setup_panels(self):
        """注册分层DQN专用面板"""
        # 环境状态面板(包含熵值统计)
        env_panel = EnvironmentPanel(width=350, height=200)
        self.panel_manager.register_panel(env_panel, position='auto')
        
        # 训练统计面板
        training_panel = TrainingStatsPanel(width=370, height=280)
        self.panel_manager.register_panel(training_panel, position='auto')
        
        # 5x5高层网格面板
        grid_panel = HierarchicalGridPanel(width=370, height=300)
        self.panel_manager.register_panel(grid_panel, position='auto')
        
        # 奖励曲线面板
        reward_panel = RewardCurvePanel(width=370, height=200)
        self.panel_manager.register_panel(reward_panel, position='auto')
        
        # 电量面板
        battery_panel = BatteryPanel(width=370, height=260)
        self.panel_manager.register_panel(battery_panel, position='auto')
    
    def get_visualization_data(self) -> Dict[str, Any]:
        """收集分层DQN训练可视化数据"""
        data = {}
        
        # 训练统计数据
        data['episode_count'] = self.episode_count
        data['total_steps'] = self.total_steps
        data['current_episode_steps'] = getattr(self.env, 'step_count', 0) if self.env else 0
        data['current_episode_reward'] = self.current_episode_reward
        
        # 奖励历史
        data['reward_history'] = list(self.reward_history)
        
        # 统计数据
        if self.reward_history:
            data['avg_reward'] = sum(self.reward_history) / len(self.reward_history)
            data['max_reward'] = max(self.reward_history)
            data['min_reward'] = min(self.reward_history)
        
        # 高层动作历史
        data['hl_action_history'] = list(self.hl_action_history)
        
        # 熵值统计(从grid_data计算)
        if data.get('grid_data'):
            self._update_entropy_stats(data['grid_data'])
            data['entropy_stats'] = self._entropy_stats
        
        return data
    
    def _update_entropy_stats(self, grid_data):
        """更新熵值统计"""
        try:
            if not grid_data or not hasattr(grid_data, 'cells') or not grid_data.cells:
                return
            
            total_cells = len(grid_data.cells)
            scanned_cells = sum(1 for cell in grid_data.cells if cell.entropy < 30)
            high_entropy_cells = sum(1 for cell in grid_data.cells if cell.entropy > 70)
            total_entropy = sum(cell.entropy for cell in grid_data.cells)
            avg_entropy = total_entropy / total_cells if total_cells > 0 else 0
            
            self._entropy_stats = {
                'total': total_cells,
                'scanned': scanned_cells,
                'high_entropy': high_entropy_cells,
                'avg_entropy': avg_entropy,
                'scan_ratio': (scanned_cells / total_cells * 100) if total_cells > 0 else 0
            }
        except Exception as e:
            pass
    
    def update_training_data(self, step: int, action: int, reward: float, drone_name: str = "UAV1"):
        """
        更新训练数据(由训练脚本调用)
        
        Args:
            step: 当前步数
            action: 高层动作ID
            reward: 奖励值
            drone_name: 无人机名称
        """
        self.total_steps = step
        self.current_episode_reward += reward
        self.hl_action_history.append((step, action, drone_name))
        self.reward_history.append(reward)
    
    def on_episode_end(self, episode: int):
        """
        Episode结束时调用
        
        Args:
            episode: Episode编号
        """
        self.episode_count = episode
        self.current_episode_reward = 0


# 兼容性别名
HierarchicalVisualizer = HierarchicalTrainingVisualizer


if __name__ == "__main__":
    # 测试代码
    print("分层DQN训练可视化器 - 独立测试模式")
    print("提示: 此模式需要HierarchicalMovementEnv实例才能完整显示")
    
    visualizer = HierarchicalTrainingVisualizer(env=None, server=None)
    visualizer.start_visualization()
    
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n正在关闭...")
        visualizer.stop_visualization()
