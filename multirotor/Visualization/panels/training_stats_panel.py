
from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict

import pygame

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from multirotor.Visualization.panel_system import BasePanel


class TrainingStatsPanel(BasePanel):
    def __init__(self, width: int = 370, height: int = 280):
        super().__init__('training_stats', width, height)
        self.training_start_time = time.time()
        self._big_font = None

    def draw(self, screen: pygame.Surface, data: Dict[str, Any]):
        self._init_fonts()
        if self._big_font is None:
            try:
                self._big_font = pygame.font.SysFont(['SimHei', 'Microsoft YaHei', 'Arial'], 22, bold=True)
            except Exception:
                self._big_font = pygame.font.Font(None, 22)

        accent = self.WARNING
        self.draw_panel_background(screen, border_color=accent)
        y_offset = self.draw_title(screen, '训练状态', accent)

        text_x = self.x + 15
        y = self.y + y_offset + 6
        episode_count = int(data.get('episode_count', 0) or 0)
        total_steps = int(data.get('total_steps', 0) or 0)
        current_episode_steps = int(data.get('current_episode_steps', 0) or 0)
        current_episode_reward = float(data.get('current_episode_reward', 0.0) or 0.0)
        steps_per_sec = float(data.get('steps_per_sec', 0.0) or 0.0)

        big_step = self._big_font.render(f'步数: {total_steps}', True, self.INFO)
        screen.blit(big_step, (text_x, y))
        y += 34

        lines = [
            (f'Episode: {episode_count}', self.TEXT_PRIMARY),
            (f'当前Episode步数: {current_episode_steps}', self.INFO),
            (f'当前Episode奖励: {current_episode_reward:.2f}', self.INFO),
            (f'速率: {steps_per_sec:.2f} steps/s', self.SUCCESS if steps_per_sec > 0 else self.TEXT_SECONDARY),
        ]
        for content, color in lines:
            text = self._font.render(content, True, color)
            screen.blit(text, (text_x, y))
            y += 22

        elapsed_time = time.time() - self.training_start_time
        hh = int(elapsed_time // 3600)
        mm = int((elapsed_time % 3600) // 60)
        ss = int(elapsed_time % 60)
        time_text = self._font.render(f'已用时间(可视化): {hh:02d}:{mm:02d}:{ss:02d}', True, self.TEXT_PRIMARY)
        screen.blit(time_text, (text_x, y))
        y += 24

        current_ep_time = data.get('current_episode_time', data.get('episode_elapsed_time'))
        last_ep_duration = data.get('last_episode_duration')
        total_training_time = data.get('total_training_time')
        if current_ep_time is not None:
            text = self._font.render(f'当前Episode耗时: {float(current_ep_time):.1f} s', True, self.TEXT_PRIMARY)
            screen.blit(text, (text_x, y))
            y += 20
        elif last_ep_duration is not None:
            text = self._font.render(f'上一Episode耗时: {float(last_ep_duration):.1f} s', True, self.TEXT_PRIMARY)
            screen.blit(text, (text_x, y))
            y += 20

        if total_training_time is not None and float(total_training_time) > 0:
            text = self._font.render(f'总训练耗时: {float(total_training_time):.1f} s', True, self.TEXT_PRIMARY)
            screen.blit(text, (text_x, y))
            y += 24

        avg_reward = data.get('avg_reward', 0.0)
        max_reward = data.get('max_reward', 0.0)
        min_reward = data.get('min_reward', 0.0)
        if any(v not in (None, 0, 0.0) for v in [avg_reward, max_reward, min_reward]):
            self.draw_divider(screen, y)
            y += 12
            stats = [
                (f'平均奖励: {float(avg_reward):.2f}', self.SUCCESS),
                (f'最佳奖励: {float(max_reward):.2f}', self.INFO),
                (f'最低奖励: {float(min_reward):.2f}', self.DANGER),
            ]
            for content, color in stats:
                text = self._font.render(content, True, color)
                screen.blit(text, (text_x, y))
                y += 19
