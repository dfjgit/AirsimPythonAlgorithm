from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import pygame

from multirotor.Visualization.panel_system import BasePanel


class EntropyTrendPanel(BasePanel):
    def __init__(self, width: int = 370, height: int = 280):
        super().__init__("entropy_trend", width, height)

    def draw(self, screen: pygame.Surface, data: Dict[str, Any]):
        self._init_fonts()
        accent = self.SKY
        self.draw_panel_background(screen, border_color=accent)
        y_offset = self.draw_title(screen, "残值采集情况", accent)

        entropy_history = self._normalize_history(data.get("entropy_history"), expected_size=2)
        scan_progress_history = self._normalize_history(
            data.get("scan_progress_history"), expected_size=4
        )
        if len(entropy_history) < 2 or len(scan_progress_history) < 2:
            hint = self._font.render("等待残值采集趋势数据...", True, self.TEXT_SECONDARY)
            screen.blit(hint, (self.x + 15, self.y + y_offset + 8))
            return

        text_x = self.x + 15
        y = self.y + y_offset + 12
        chart_gap = 18
        summary_height = self._small_font.get_height() + 8
        chart_height = max(
            86,
            int((self.height - y_offset - summary_height * 2 - chart_gap - 18) / 2),
        )

        entropy_chart_rect = pygame.Rect(text_x, y, self.width - 30, chart_height)
        self._draw_series_chart(
            screen,
            entropy_chart_rect,
            entropy_history,
            value_index=1,
            line_color=self.ROSE,
            label="平均残值",
            min_value=0.0,
            max_value=100.0,
        )

        latest_entropy = float(entropy_history[-1][1])
        latest_scan_ratio = float(scan_progress_history[-1][1])
        latest_scanned = int(scan_progress_history[-1][2])
        latest_total = int(scan_progress_history[-1][3])
        entropy_text = self._strong_small_font.render(
            f"最新平均残值: {latest_entropy:.1f}",
            True,
            self.TEXT_PRIMARY,
        )
        screen.blit(entropy_text, (text_x, y + entropy_chart_rect.height + 4))

        y = entropy_chart_rect.bottom + chart_gap
        progress_chart_rect = pygame.Rect(text_x, y, self.width - 30, chart_height)
        self._draw_series_chart(
            screen,
            progress_chart_rect,
            scan_progress_history,
            value_index=1,
            line_color=self.MINT,
            label="扫描率",
            min_value=0.0,
            max_value=100.0,
        )

        progress_text = self._strong_small_font.render(
            f"最新扫描率: {latest_scan_ratio:.1f}% | 已扫描: {latest_scanned}/{latest_total}",
            True,
            self.TEXT_PRIMARY,
        )
        screen.blit(progress_text, (text_x, y + progress_chart_rect.height + 4))

    def _draw_series_chart(
        self,
        screen: pygame.Surface,
        rect: pygame.Rect,
        series: List[Tuple[float, ...]],
        value_index: int,
        line_color: Tuple[int, int, int],
        label: str,
        min_value: float,
        max_value: float,
    ) -> None:
        pygame.draw.rect(screen, self.PANEL_BACKGROUND_SOFT, rect, border_radius=8)
        pygame.draw.rect(screen, self.PANEL_BORDER, rect, 1, border_radius=8)

        label_text = self._strong_small_font.render(label, True, self.TEXT_SECONDARY)
        screen.blit(label_text, (rect.x + 8, rect.y + 6))
        latest_value = float(series[-1][value_index])
        latest_text = self._strong_small_font.render(
            f"{latest_value:.1f}",
            True,
            line_color,
        )
        latest_rect = latest_text.get_rect(topright=(rect.right - 8, rect.y + 6))
        screen.blit(latest_text, latest_rect)

        chart_margin_left = 8
        chart_margin_right = 8
        chart_margin_top = 24
        chart_margin_bottom = 12
        chart_width = max(10, rect.width - chart_margin_left - chart_margin_right)
        chart_height = max(10, rect.height - chart_margin_top - chart_margin_bottom)
        chart_left = rect.x + chart_margin_left
        chart_bottom = rect.y + rect.height - chart_margin_bottom

        points: List[Tuple[int, int]] = []
        series_count = len(series)
        value_range = max(max_value - min_value, 1.0)
        for index, entry in enumerate(series):
            value = float(entry[value_index])
            x = chart_left + int((index / max(series_count - 1, 1)) * chart_width)
            y = chart_bottom - int(((value - min_value) / value_range) * chart_height)
            points.append((x, y))

        if len(points) >= 2:
            pygame.draw.lines(screen, line_color, False, points, 2)
        for point in points[-4:]:
            pygame.draw.circle(screen, line_color, point, 3)

        baseline_y = chart_bottom - int(((0.0 - min_value) / value_range) * chart_height)
        if rect.y + chart_margin_top <= baseline_y <= chart_bottom:
            pygame.draw.line(
                screen,
                self.DIVIDER,
                (chart_left, baseline_y),
                (chart_left + chart_width, baseline_y),
                1,
            )

    def _normalize_history(self, raw_history: Any, expected_size: int) -> List[Tuple[float, ...]]:
        history: List[Tuple[float, ...]] = []
        if not isinstance(raw_history, (list, tuple)):
            return history
        for entry in raw_history:
            if not isinstance(entry, (list, tuple)) or len(entry) < expected_size:
                continue
            try:
                history.append(tuple(float(entry[i]) for i in range(expected_size)))
            except Exception:
                continue
        return history
