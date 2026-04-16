
from __future__ import annotations

import os
import sys
from typing import Any, Dict

import pygame

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from multirotor.Visualization.panel_system import BasePanel


class BatteryPanel(BasePanel):
    def __init__(self, width: int = 370, height: int = 220):
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

        items = list(battery_data.items())
        content_x = self.x + 12
        content_y = self.y + y_offset + 4
        content_width = self.width - 24
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

        column_gap = 10
        if len(items) <= 4:
            max_columns = 1
        elif len(items) <= 8:
            max_columns = 2
        else:
            max_columns = 3
        card_width = (
            (content_width - column_gap * (max_columns - 1)) // max_columns
            if max_columns > 1
            else content_width
        )
        row_height = 54 if max_columns == 1 else 46 if max_columns == 2 else 38
        available_height = self.height - y_offset - 34
        rows_per_column = max(1, available_height // row_height)
        if len(items) > rows_per_column * max_columns and max_columns < 3:
            max_columns = min(3, len(items))
            card_width = (
                (content_width - column_gap * (max_columns - 1)) // max_columns
                if max_columns > 1
                else content_width
            )
            row_height = 54 if max_columns == 1 else 46 if max_columns == 2 else 38
            rows_per_column = max(1, available_height // row_height)

        for index, (drone_name, battery_info) in enumerate(items):
            column_index = index // rows_per_column
            row_index = index % rows_per_column
            if column_index >= max_columns:
                break

            card_x = content_x + column_index * (card_width + column_gap)
            y = content_y + row_index * row_height

            voltage = float(battery_info.get('voltage', 4.2) or 4.2)
            raw_percentage = battery_info.get('remaining_percentage', 100.0)
            percentage = 100.0 if raw_percentage is None else float(raw_percentage)
            status = str(battery_info.get('status', 'normal') or 'normal')
            is_crazyflie = bool(battery_info.get('crazyflieMirror', False))
            status_color = status_colors.get(status, self.TEXT_PRIMARY)
            drone_type = ' CF' if is_crazyflie else ''
            compact_mode = max_columns >= 3
            name_font = self._strong_small_font if compact_mode else self._strong_font
            name_text = name_font.render(f'{drone_name}{drone_type}', True, self.TEXT_PRIMARY)
            screen.blit(name_text, (card_x, y))
            if compact_mode:
                y += 12
                voltage_text = self._strong_small_font.render(f'{voltage:.2f}V {percentage:.0f}%', True, status_color)
                screen.blit(voltage_text, (card_x, y))
                y += 12
            else:
                status_text = self._strong_small_font.render(status_names.get(status, status), True, status_color)
                status_rect = status_text.get_rect(topright=(card_x + card_width, y))
                screen.blit(status_text, status_rect)
                y += 18

                voltage_text = self._strong_small_font.render(f'{voltage:.2f}V  ({percentage:.0f}%)', True, status_color)
                screen.blit(voltage_text, (card_x, y))
                y += 16

            bar_x = card_x
            bar_y = y
            width_ratio = 0.74 if max_columns == 1 else 0.84 if max_columns == 2 else 0.92
            bar_width = max(72, int(card_width * width_ratio))
            bar_height = 6 if compact_mode else 7
            pygame.draw.rect(screen, self.PANEL_BACKGROUND_SOFT, (bar_x, bar_y, bar_width, bar_height), border_radius=4)
            pygame.draw.rect(screen, self.PANEL_BORDER, (bar_x, bar_y, bar_width, bar_height), 1, border_radius=4)
            fill_width = int(bar_width * max(0.0, min(percentage / 100.0, 1.0)))
            if fill_width > 0:
                pygame.draw.rect(screen, status_color, (bar_x, bar_y, fill_width, bar_height), border_radius=4)

        if battery_data:
            y = self.y + self.height - 28
            self.draw_divider(screen, y)
            y += 6
            avg_percentage = sum(
                float(0.0 if b.get('remaining_percentage', 0) is None else b.get('remaining_percentage', 0))
                for b in battery_data.values()
            ) / len(battery_data)
            avg_color = self.SUCCESS if avg_percentage > 50 else self.WARNING if avg_percentage > 30 else self.DANGER
            summary_text = self._strong_small_font.render(f'平均电量: {avg_percentage:.1f}%  |  {len(battery_data)}架无人机', True, avg_color)
            screen.blit(summary_text, (content_x, y))
