
from __future__ import annotations

import os
import sys
from typing import Any, Dict

import pygame

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from multirotor.Visualization.panel_system import BasePanel


class EnvironmentPanel(BasePanel):
    def __init__(self, width: int = 350, height: int = 180):
        super().__init__('environment', width, height)

    def draw(self, screen: pygame.Surface, data: Dict[str, Any]):
        self._init_fonts()
        accent = self.TEAL
        self.draw_panel_background(screen, border_color=accent)
        y_offset = self.draw_title(screen, '环境状态', accent)

        grid_stats = self._calculate_grid_stats(data.get('grid_data'))
        if (
            data.get('csv_global_total_count', 0)
            and data.get('csv_global_scanned_count', 0) >= 0
        ):
            total = int(data.get('csv_global_total_count', 0) or 0)
            scanned = int(data.get('csv_global_scanned_count', 0) or 0)
            if total > 0:
                csv_ratio = (scanned / total) * 100.0
                if not grid_stats:
                    grid_stats = {'total': total, 'avg': 0.0, 'scanned': scanned, 'scan_ratio': csv_ratio}
                else:
                    grid_stats['scanned'] = scanned
                    grid_stats['total'] = total
                    grid_stats['scan_ratio'] = csv_ratio
        if not grid_stats:
            hint = self._font.render('等待网格数据...', True, self.TEXT_SECONDARY)
            screen.blit(hint, (self.x + 15, self.y + y_offset + 8))
            return

        text_x = self.x + 15
        y = self.y + y_offset + 6
        rows = [
            (f"网格单元: {grid_stats['total']}", self.TEXT_PRIMARY),
            (f"平均熵值: {grid_stats['avg']:.1f}", self.TEXT_PRIMARY),
        ]
        for content, color in rows:
            text = self._font.render(content, True, color)
            screen.blit(text, (text_x, y))
            y += 22

        scanned = grid_stats['scanned']
        total = grid_stats['total']
        ratio = grid_stats['scan_ratio']
        progress_color = self.SUCCESS if ratio >= 15 else self.WARNING if ratio >= 5 else self.ORANGE
        progress_text = self._font.render(f'已扫描: {scanned}/{total} ({ratio:.1f}%)', True, progress_color)
        screen.blit(progress_text, (text_x, y))
        y += 26

        bar_x = text_x
        bar_y = y
        bar_width = self.width - 30
        bar_height = 12
        pygame.draw.rect(screen, self.PANEL_BACKGROUND_SOFT, (bar_x, bar_y, bar_width, bar_height), border_radius=6)
        pygame.draw.rect(screen, self.PANEL_BORDER, (bar_x, bar_y, bar_width, bar_height), 1, border_radius=6)
        fill_width = int(bar_width * max(0.0, min(ratio / 100.0, 1.0)))
        if fill_width > 0:
            pygame.draw.rect(screen, progress_color, (bar_x, bar_y, fill_width, bar_height), border_radius=6)
        y += 22

        runtime_data = data.get('runtime_data', {})
        drone_count = len(runtime_data) if runtime_data else 0
        drone_text = self._font.render(f'无人机数量: {drone_count}', True, self.INFO)
        screen.blit(drone_text, (text_x, y))

    def _calculate_grid_stats(self, grid_data):
        if not grid_data or not hasattr(grid_data, 'cells') or not grid_data.cells:
            return None
        total = len(grid_data.cells)
        total_entropy = sum(cell.entropy for cell in grid_data.cells)
        avg_entropy = total_entropy / total if total > 0 else 0.0
        scanned = sum(1 for cell in grid_data.cells if cell.entropy < 30)
        scan_ratio = (scanned / total * 100.0) if total > 0 else 0.0
        return {'total': total, 'avg': avg_entropy, 'scanned': scanned, 'scan_ratio': scan_ratio}
