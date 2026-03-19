
from __future__ import annotations

import os
import sys
from typing import Any, Dict

import pygame

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from multirotor.Visualization.panel_system import BasePanel


class BatteryPanel(BasePanel):
    def __init__(self, width: int = 370, height: int = 260):
        super().__init__('battery', width, height)

    def draw(self, screen: pygame.Surface, data: Dict[str, Any]):
        self._init_fonts()
        accent = self.WARNING
        self.draw_panel_background(screen, border_color=accent)
        y_offset = self.draw_title(screen, '电量状态', accent)
        battery_data = data.get('battery_data', {})
        if not battery_data:
            hint_text = self._font.render('等待电量数据...', True, self.TEXT_MUTED)
            hint_rect = hint_text.get_rect(center=(self.x + self.width // 2, self.y + self.height // 2))
            screen.blit(hint_text, hint_rect)
            return

        text_x = self.x + 15
        y = self.y + y_offset + 2
        status_colors = {
            'normal': self.SUCCESS,
            'warning': self.WARNING,
            'low': self.ORANGE,
            'critical': self.DANGER,
            'empty': self.TEXT_MUTED,
        }
        status_names = {
            'normal': '正常',
            'warning': '警告',
            'low': '低电',
            'critical': '危险',
            'empty': '耗尽',
        }

        for i, (drone_name, battery_info) in enumerate(battery_data.items()):
            if y + 50 > self.y + self.height - 10:
                remaining = len(battery_data) - i
                if remaining > 0:
                    more_text = self._small_font.render(f'... 还有{remaining}架无人机', True, self.TEXT_MUTED)
                    screen.blit(more_text, (text_x, y))
                break

            voltage = float(battery_info.get('voltage', 4.2) or 4.2)
            percentage = float(battery_info.get('remaining_percentage', 100.0) or 100.0)
            status = str(battery_info.get('status', 'normal') or 'normal')
            is_crazyflie = bool(battery_info.get('crazyflieMirror', False))
            status_color = status_colors.get(status, self.TEXT_PRIMARY)
            drone_type = ' CF' if is_crazyflie else ''
            name_text = self._font.render(f'{drone_name}{drone_type}', True, self.TEXT_PRIMARY)
            screen.blit(name_text, (text_x, y))
            voltage_text = self._small_font.render(f'{voltage:.2f}V ({percentage:.0f}%)', True, status_color)
            screen.blit(voltage_text, (text_x + 90, y + 2))
            status_text = self._small_font.render(status_names.get(status, status), True, status_color)
            screen.blit(status_text, (text_x + 200, y + 2))
            y += 18

            bar_x = text_x
            bar_y = y
            bar_width = self.width - 30
            bar_height = 8
            pygame.draw.rect(screen, self.PANEL_BACKGROUND_SOFT, (bar_x, bar_y, bar_width, bar_height), border_radius=4)
            pygame.draw.rect(screen, self.PANEL_BORDER, (bar_x, bar_y, bar_width, bar_height), 1, border_radius=4)
            fill_width = int(bar_width * max(0.0, min(percentage / 100.0, 1.0)))
            if fill_width > 0:
                pygame.draw.rect(screen, status_color, (bar_x, bar_y, fill_width, bar_height), border_radius=4)
            y += 16

        if battery_data:
            y = self.y + self.height - 35
            self.draw_divider(screen, y)
            y += 8
            avg_percentage = sum(float(b.get('remaining_percentage', 0) or 0) for b in battery_data.values()) / len(battery_data)
            avg_color = self.SUCCESS if avg_percentage > 50 else self.WARNING if avg_percentage > 30 else self.DANGER
            summary_text = self._font.render(f'平均电量: {avg_percentage:.1f}%  |  {len(battery_data)}架无人机', True, avg_color)
            screen.blit(summary_text, (text_x, y))
