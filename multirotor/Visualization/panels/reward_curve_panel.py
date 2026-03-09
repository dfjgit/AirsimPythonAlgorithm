import os
import sys
from collections import deque
from typing import Any, Dict

import pygame

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from multirotor.Visualization.panel_system import BasePanel


class RewardCurvePanel(BasePanel):
    """奖励曲线面板，默认展示每个 episode 的最终奖励。"""

    def __init__(self, width: int = 370, height: int = 200, max_points: int = 200):
        super().__init__("reward_curve", width, height)
        self.reward_history = deque(maxlen=max_points)

    def update_data(self, data: Dict[str, Any]):
        """当前版本直接使用上层传入的奖励序列。"""
        return None

    def draw(self, screen: pygame.Surface, data: Dict[str, Any]):
        self._init_fonts()
        self.draw_panel_background(screen, border_color=self.CYAN)

        episode_reward_history = data.get('episode_reward_history', [])
        fallback_history = data.get('reward_history', [])
        reward_history = episode_reward_history if episode_reward_history else fallback_history
        displayed_count = len(reward_history)

        y_offset = self.draw_title(screen, f"奖励曲线 (Episode {displayed_count})", self.CYAN)

        if not reward_history or len(reward_history) < 2:
            hint_text = self._font.render("等待训练数据...", True, self.GRAY)
            hint_rect = hint_text.get_rect(center=(self.x + self.width // 2, self.y + self.height // 2))
            screen.blit(hint_text, hint_rect)
            return

        chart_margin_x = 40
        chart_margin_y = 35
        chart_bottom_margin = 34
        chart_width = self.width - chart_margin_x - 20
        chart_height = self.height - y_offset - chart_margin_y - chart_bottom_margin
        chart_origin_x = self.x + chart_margin_x
        chart_origin_y = self.y + y_offset + chart_height

        pygame.draw.line(
            screen,
            self.LIGHT_GRAY,
            (chart_origin_x, chart_origin_y),
            (chart_origin_x + chart_width, chart_origin_y),
            2,
        )
        pygame.draw.line(
            screen,
            self.LIGHT_GRAY,
            (chart_origin_x, chart_origin_y - chart_height),
            (chart_origin_x, chart_origin_y),
            2,
        )

        rewards = list(reward_history)
        min_reward = min(rewards)
        max_reward = max(rewards)
        reward_range = max(max_reward - min_reward, 1.0)

        points = []
        for i, reward in enumerate(rewards):
            x = chart_origin_x + (i / max(len(rewards) - 1, 1)) * chart_width
            y = chart_origin_y - ((reward - min_reward) / reward_range) * chart_height
            points.append((int(x), int(y)))

        if len(points) > 1:
            pygame.draw.lines(screen, self.CYAN, False, points, 2)
            for point in points[-min(4, len(points)):]:
                pygame.draw.circle(screen, self.CYAN, point, 3)

        avg_reward = sum(rewards) / len(rewards)
        avg_y = chart_origin_y - ((avg_reward - min_reward) / reward_range) * chart_height
        pygame.draw.line(
            screen,
            self.ORANGE,
            (chart_origin_x, int(avg_y)),
            (chart_origin_x + chart_width, int(avg_y)),
            1,
        )

        label_max = self._small_font.render(f"{max_reward:.1f}", True, self.WHITE)
        screen.blit(label_max, (self.x + 5, chart_origin_y - chart_height - 5))

        label_min = self._small_font.render(f"{min_reward:.1f}", True, self.WHITE)
        screen.blit(label_min, (self.x + 5, chart_origin_y - 10))

        latest_text = self._small_font.render(
            f"最终奖励: {rewards[-1]:.2f} | 平均: {avg_reward:.1f}",
            True,
            self.CYAN,
        )
        screen.blit(latest_text, (chart_origin_x, self.y + self.height - 30))

        current_episode_reward = float(data.get('current_episode_reward', 0.0) or 0.0)
        current_text = self._small_font.render(
            f"当前轮累计: {current_episode_reward:.2f}",
            True,
            self.ORANGE,
        )
        screen.blit(current_text, (chart_origin_x, self.y + self.height - 16))
