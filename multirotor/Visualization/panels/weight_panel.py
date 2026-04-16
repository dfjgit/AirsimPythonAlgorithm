
from __future__ import annotations

import os
import sys
from typing import Any, Dict

import pygame

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from multirotor.Visualization.panel_system import BasePanel


class WeightPanel(BasePanel):
    def __init__(self, width: int = 370, height: int = 280):
        super().__init__('weight', width, height)

    def draw(self, screen: pygame.Surface, data: Dict[str, Any]):
        self._init_fonts()
        weights = data.get('weights', {})
        if not weights:
            return

        use_dqn = data.get('use_dqn', False)
        accent = self.MINT if use_dqn else self.INFO
        self.draw_panel_background(screen, border_color=accent)
        title = 'DQN预测权重' if use_dqn else 'APF权重系数'
        y_offset = self.draw_title(screen, title, accent)
        text_x = self.x + 12
        y = self.y + y_offset + 2

        core_weights = [
            ('α1 排斥', weights.get('repulsionCoefficient', 0.0)),
            ('α2 熵值', weights.get('entropyCoefficient', 0.0)),
            ('α3 距离', weights.get('distanceCoefficient', 0.0)),
            ('α4 Leader', weights.get('leaderRangeCoefficient', 0.0)),
            ('α5 方向', weights.get('directionRetentionCoefficient', 0.0)),
        ]
        obstacle_weights = [
            ('避障距离', weights.get('obstacleRepulsionDistance', 15.0)),
            ('避障系数', weights.get('obstacleRepulsionCoefficient', 5.0)),
            ('禁飞区距离', weights.get('restrictedZoneDistance', 15.0)),
            ('禁飞区系数', weights.get('restrictedZoneCoefficient', 5.0)),
        ]

        row_gap = 18
        for name, value in core_weights:
            self._draw_weight_row(screen, text_x, y, name, value, min_val=0.5, max_val=5.0)
            y += row_gap
        y += 2
        self.draw_divider(screen, y)
        y += 8
        for name, value in obstacle_weights:
            self._draw_weight_row(screen, text_x, y, name, value, min_val=0.0, max_val=30.0, is_obstacle=True)
            y += row_gap

    def _bar_color(self, value: float, min_val: float, max_val: float, is_obstacle: bool) -> tuple[int, int, int]:
        if is_obstacle:
            if value >= 15:
                return self.ORANGE
            if value >= 7:
                return self.WARNING
            return self.SUCCESS
        if value < 1.5:
            return self.SUCCESS
        if value < 3.0:
            return self.WARNING
        return self.DANGER

    def _draw_weight_row(self, screen: pygame.Surface, x: int, y: int, name: str, value: float, min_val: float = 0.5, max_val: float = 5.0, is_obstacle: bool = False):
        label = self._strong_small_font.render(f'{name}:', True, self.TEXT_SECONDARY)
        screen.blit(label, (x, y))
        bar_x = x + label.get_width() + 14
        bar_y = y + 4
        value_text_width = 34
        horizontal_padding = 18
        bar_width = max(80, self.x + self.width - bar_x - value_text_width - horizontal_padding)
        bar_height = 8
        pygame.draw.rect(screen, self.PANEL_BACKGROUND_SOFT, (bar_x, bar_y, bar_width, bar_height), border_radius=5)
        pygame.draw.rect(screen, self.PANEL_BORDER, (bar_x, bar_y, bar_width, bar_height), 1, border_radius=5)
        ratio = 0.0 if max_val <= min_val else max(0.0, min((value - min_val) / (max_val - min_val), 1.0))
        fill_width = int(bar_width * ratio)
        if fill_width > 0:
            color = self._bar_color(value, min_val, max_val, is_obstacle)
            pygame.draw.rect(screen, color, (bar_x, bar_y, fill_width, bar_height), border_radius=5)
        value_text = self._strong_small_font.render(f'{value:.1f}', True, self.TEXT_PRIMARY)
        screen.blit(value_text, (bar_x + bar_width + 6, y))
