"""奖励曲线面板 - 显示实时奖励变化趋势"""
import sys
import os
import pygame
from typing import Dict, Any
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from multirotor.Visualization.panel_system import BasePanel


class RewardCurvePanel(BasePanel):
    """奖励曲线面板"""
    
    def __init__(self, width: int = 370, height: int = 200, max_points: int = 200):
        super().__init__("reward_curve", width, height)
        self.reward_history = deque(maxlen=max_points)
    
    def update_data(self, data: Dict[str, Any]):
        """更新奖励历史"""
        reward = data.get('current_episode_reward', 0.0)
        if reward != 0 or len(self.reward_history) == 0:
            # 避免大量0值填充
            pass
    
    def draw(self, screen: pygame.Surface, data: Dict[str, Any]):
        """绘制奖励曲线"""
        self._init_fonts()
        
        # 背景和边框
        self.draw_panel_background(screen, border_color=self.CYAN)
        
        # 标题
        y_offset = self.draw_title(screen, f"📈 奖励曲线 (最近{len(self.reward_history)})", self.CYAN)
        
        # 获取奖励历史
        reward_history = data.get('reward_history', [])
        if not reward_history or len(reward_history) < 2:
            hint_text = self._font.render("等待训练数据...", True, self.GRAY)
            hint_rect = hint_text.get_rect(center=(self.x + self.width // 2, 
                                                   self.y + self.height // 2))
            screen.blit(hint_text, hint_rect)
            return
        
        # 图表区域
        chart_margin_x = 40
        chart_margin_y = 35
        chart_width = self.width - chart_margin_x - 20
        chart_height = self.height - y_offset - chart_margin_y - 20
        chart_origin_x = self.x + chart_margin_x
        chart_origin_y = self.y + y_offset + chart_height
        
        # 绘制坐标轴
        pygame.draw.line(screen, self.LIGHT_GRAY,
                        (chart_origin_x, chart_origin_y),
                        (chart_origin_x + chart_width, chart_origin_y), 2)
        pygame.draw.line(screen, self.LIGHT_GRAY,
                        (chart_origin_x, chart_origin_y - chart_height),
                        (chart_origin_x, chart_origin_y), 2)
        
        # 计算缩放
        rewards = list(reward_history)
        min_reward = min(rewards)
        max_reward = max(rewards)
        reward_range = max(max_reward - min_reward, 1.0)
        
        # 绘制曲线
        points = []
        for i, reward in enumerate(rewards):
            x = chart_origin_x + (i / (len(rewards) - 1)) * chart_width
            y = chart_origin_y - ((reward - min_reward) / reward_range) * chart_height
            points.append((int(x), int(y)))
        
        if len(points) > 1:
            pygame.draw.lines(screen, self.CYAN, False, points, 2)
        
        # 绘制平均线
        avg_reward = sum(rewards) / len(rewards)
        avg_y = chart_origin_y - ((avg_reward - min_reward) / reward_range) * chart_height
        pygame.draw.line(screen, self.ORANGE,
                        (chart_origin_x, int(avg_y)),
                        (chart_origin_x + chart_width, int(avg_y)), 1)
        
        # Y轴标签
        label_max = self._small_font.render(f"{max_reward:.1f}", True, self.WHITE)
        screen.blit(label_max, (self.x + 5, chart_origin_y - chart_height - 5))
        
        label_min = self._small_font.render(f"{min_reward:.1f}", True, self.WHITE)
        screen.blit(label_min, (self.x + 5, chart_origin_y - 10))
        
        # 最新值
        latest_text = self._small_font.render(f"最新: {rewards[-1]:.2f} | 平均: {avg_reward:.1f}",
                                              True, self.CYAN)
        screen.blit(latest_text, (chart_origin_x, self.y + self.height - 18))
