"""
DQN移动训练可视化器 - 用于纯DQN移动控制训练

支持:
- 环境状态(熵值热力图)
- 训练统计(Episode、步数、奖励)
- 奖励曲线
- Q值分布(可选)
- 动作选择统计
"""
import sys
import os
from typing import Dict, Any
from collections import deque, Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from multirotor.Visualization.base_visualizer import BaseVisualizer
from multirotor.Visualization.panels.environment_panel import EnvironmentPanel
from multirotor.Visualization.panels.training_stats_panel import TrainingStatsPanel
from multirotor.Visualization.panels.reward_curve_panel import RewardCurvePanel
from multirotor.Visualization.panels.battery_panel import BatteryPanel
from multirotor.Visualization.panels.action_output_panel import ActionOutputPanel
from multirotor.Visualization.panel_system import BasePanel
from multirotor.training_stats_schema import normalize_training_stats
import pygame


class ActionDistributionPanel(BasePanel):
    """动作选择分布面板"""
    
    def __init__(self, width: int = 370, height: int = 250):
        super().__init__("action_distribution", width, height)
        self.action_names = {
            0: "上升",
            1: "下降",
            2: "左移",
            3: "右移",
            4: "前进",
            5: "后退",
        }
    
    def draw(self, screen: pygame.Surface, data: Dict[str, Any]):
        """绘制动作分布"""
        self._init_fonts()
        
        # 背景和边框
        self.draw_panel_background(screen, border_color=self.ORANGE)
        
        # 标题
        y_offset = self.draw_title(screen, "🎮 动作选择分布", self.ORANGE)
        
        # 获取动作统计
        action_counts = data.get('action_counts', {})
        if not action_counts:
            hint_text = self._font.render("等待动作数据...", True, self.GRAY)
            hint_rect = hint_text.get_rect(center=(self.x + self.width // 2,
                                                   self.y + self.height // 2))
            screen.blit(hint_text, hint_rect)
            return
        
        text_x = self.x + 15
        y = self.y + y_offset
        
        # 计算总数和百分比
        total = sum(action_counts.values())
        if total == 0:
            return
        
        # 颜色映射
        action_colors = [
            self.GREEN,    # 前进
            self.RED,      # 后退
            self.BLUE,     # 左移
            self.CYAN,     # 右移
            self.YELLOW,   # 上升
            self.MAGENTA   # 下降
        ]
        
        # 绘制每个动作的统计
        for action_id in range(6):
            count = action_counts.get(action_id, 0)
            percentage = (count / total * 100) if total > 0 else 0
            
            # 动作名称
            action_name = self.action_names.get(action_id, f"动作{action_id}")
            text = self._font.render(f"{action_name}:", True, self.WHITE)
            screen.blit(text, (text_x, y))
            
            # 统计数字
            count_text = self._small_font.render(f"{count} ({percentage:.1f}%)", 
                                                 True, self.LIGHT_GRAY)
            screen.blit(count_text, (text_x + 80, y + 2))
            
            # 进度条
            bar_x = text_x + 180
            bar_y = y + 3
            bar_width = 150
            bar_height = 10
            
            # 背景条
            pygame.draw.rect(screen, self.DARK_GRAY, (bar_x, bar_y, bar_width, bar_height))
            
            # 填充条
            fill_width = int(bar_width * (percentage / 100))
            if fill_width > 0:
                color = action_colors[action_id]
                pygame.draw.rect(screen, color, (bar_x, bar_y, fill_width, bar_height))
            
            # 边框
            pygame.draw.rect(screen, self.WHITE, (bar_x, bar_y, bar_width, bar_height), 1)
            
            y += 22
        
        # 总计
        y += 5
        pygame.draw.line(screen, self.GRAY, (text_x, y), 
                        (self.x + self.width - 15, y), 1)
        y += 8
        total_text = self._font.render(f"总动作数: {total}", True, self.YELLOW)
        screen.blit(total_text, (text_x, y))


class DQNMovementTrainingVisualizer(BaseVisualizer):
    """
    DQN移动训练可视化器
    
    用于纯DQN移动控制训练,显示:
    - 环境状态(熵值热力图)
    - 训练统计
    - 奖励曲线
    - 动作选择分布
    """
    
    def __init__(self, env, server=None):
        super().__init__(
            server=server,
            env=env,
            window_title="DQN移动训练可视化"
        )
        
        # 训练统计
        self.episode_count = 0
        self.total_steps = 0
        self.current_episode_reward = 0
        self.current_episode_steps = 0
        
        # Episode 时间统计
        self.episode_start_time = None
        self.last_episode_duration = 0.0
        self.total_training_time = 0.0
        
        # 奖励历史
        self.reward_history = deque(maxlen=200)
        
        # 动作统计
        self.action_counts = Counter()
        self.recent_actions = deque(maxlen=1000)  # 最近1000个动作
    
    def setup_panels(self):
        """注册DQN移动训练专用面板"""
        # 环境状态面板
        env_panel = EnvironmentPanel(width=350, height=180)
        self.panel_manager.register_panel(env_panel, position='auto')
        
        # 训练统计面板
        training_panel = TrainingStatsPanel(width=370, height=280)
        self.panel_manager.register_panel(training_panel, position='auto')

        # 当前动作输出面板 (新增)
        action_out_panel = ActionOutputPanel(width=370, height=260)
        self.panel_manager.register_panel(action_out_panel, position='auto')
        
        # 奖励曲线面板
        reward_panel = RewardCurvePanel(width=370, height=200)
        self.panel_manager.register_panel(reward_panel, position='auto')
        
        # 动作分布面板
        action_panel = ActionDistributionPanel(width=370, height=250)
        self.panel_manager.register_panel(action_panel, position='auto')
        
        # 电量面板
        battery_panel = BatteryPanel(width=370, height=260)
        self.panel_manager.register_panel(battery_panel, position='auto')
    
    def get_visualization_data(self) -> Dict[str, Any]:
        """收集DQN移动训练可视化数据"""
        data = {}

        fallback_stats = {
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'current_episode_steps': self.current_episode_steps,
            'current_episode_reward': self.current_episode_reward,
            'reward_history': list(self.reward_history),
        }
        if self.episode_start_time is not None:
            import time as _time
            fallback_stats['current_episode_time'] = _time.time() - self.episode_start_time
        elif self.last_episode_duration > 0:
            fallback_stats['last_episode_duration'] = self.last_episode_duration
        if self.total_training_time > 0:
            fallback_stats['total_training_time'] = self.total_training_time
        if self.reward_history:
            fallback_stats['avg_reward'] = sum(self.reward_history) / len(self.reward_history)
            fallback_stats['max_reward'] = max(self.reward_history)
            fallback_stats['min_reward'] = min(self.reward_history)

        cts = None
        try:
            if self.server and hasattr(self.server, 'current_training_stats'):
                cts = self.server.current_training_stats
        except Exception:
            cts = None

        normalized_stats = normalize_training_stats(
            stats=cts if isinstance(cts, dict) else None,
            fallback=fallback_stats,
        )
        data.update(normalized_stats)

        raw_counts = normalized_stats.get('action_counts') or {}
        normalized_counts = {}
        for action_id in range(6):
            val = raw_counts.get(action_id, raw_counts.get(str(action_id), 0))
            normalized_counts[action_id] = int(val)
        if any(normalized_counts.values()):
            data['action_counts'] = normalized_counts
        else:
            data['action_counts'] = dict(self.action_counts)

        data['current_training_stats'] = normalized_stats

        # 获取电量数据 (通过父类方法获取 server 中的数据)
        battery_data = self.get_battery_data()
        if battery_data:
            data['battery_data'] = battery_data

        return data
    
    def update_training_stats(self, episode_reward: float = None, 
                             current_step_reward: float = None,
                             action: int = None,
                             is_episode_done: bool = False):
        """
        更新训练统计信息
        
        Args:
            episode_reward: 当前episode的总奖励
            current_step_reward: 当前步的奖励
            action: 当前选择的动作
            is_episode_done: 是否episode结束
        """
        if current_step_reward is not None:
            # 若当前没有记录episode开始时间，则认为是新的一轮episode开始
            if self.episode_start_time is None:
                import time as _time
                self.episode_start_time = _time.time()
            
            self.current_episode_reward += current_step_reward
            self.current_episode_steps += 1
            self.total_steps += 1
        
        if action is not None:
            self.action_counts[action] += 1
            self.recent_actions.append(action)
        
        if is_episode_done:
            # 计算本轮episode耗时并累加到总训练时间
            if self.episode_start_time is not None:
                import time as _time
                self.last_episode_duration = _time.time() - self.episode_start_time
                self.total_training_time += self.last_episode_duration
                self.episode_start_time = None

            if episode_reward is not None:
                self.reward_history.append(episode_reward)
            else:
                self.reward_history.append(self.current_episode_reward)
            
            self.episode_count += 1
            
            # 重置当前episode统计
            self.current_episode_reward = 0.0
            self.current_episode_steps = 0
    
    def on_episode_end(self, episode: int):
        """Episode结束时调用"""
        self.episode_count = episode


# 兼容性别名
MovementVisualizer = DQNMovementTrainingVisualizer


if __name__ == "__main__":
    # 测试代码
    print("DQN移动训练可视化器 - 独立测试模式")
    print("提示: 此模式需要MovementEnv实例才能完整显示")
    
    visualizer = DQNMovementTrainingVisualizer(env=None, server=None)
    visualizer.start_visualization()
    
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n正在关闭...")
        visualizer.stop_visualization()
