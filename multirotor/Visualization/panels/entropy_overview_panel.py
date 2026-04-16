from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import pygame

from multirotor.Visualization.panel_system import BasePanel


class EntropyOverviewPanel(BasePanel):
    def __init__(self, width: int = 350, height: int = 190):
        super().__init__("entropy_overview", width, height)

    def draw(self, screen: pygame.Surface, data: Dict[str, Any]):
        self._init_fonts()
        accent = self.TEAL
        self.draw_panel_background(screen, border_color=accent)
        y_offset = self.draw_title(screen, "当前场景残值情况", accent)

        overview = self._build_overview(data)
        if overview is None:
            hint = self._font.render("等待残值数据...", True, self.TEXT_SECONDARY)
            screen.blit(hint, (self.x + 15, self.y + y_offset + 8))
            return

        layout = self._compute_layout(overview)
        text_x = self.x + 15

        for item in layout["stat_lines"]:
            text = item["font"].render(item["text"], True, item["color"])
            screen.blit(text, (text_x, item["y"]))

        ratio = overview["scan_ratio"]
        scanned = overview["scanned"]
        total = overview["total"]
        progress_color = (
            self.SUCCESS if ratio >= 25 else self.WARNING if ratio >= 10 else self.ORANGE
        )
        progress_text = self._strong_font.render(
            f"已扫描: {scanned}/{total} ({ratio:.1f}%)", True, progress_color
        )
        screen.blit(progress_text, (text_x, layout["progress_text_y"]))

        bar_x = text_x
        bar_y = layout["progress_bar_y"]
        bar_width = self.width - 30
        bar_height = 10
        pygame.draw.rect(
            screen,
            self.PANEL_BACKGROUND_SOFT,
            (bar_x, bar_y, bar_width, bar_height),
            border_radius=6,
        )
        pygame.draw.rect(
            screen,
            self.PANEL_BORDER,
            (bar_x, bar_y, bar_width, bar_height),
            1,
            border_radius=6,
        )
        fill_width = int(bar_width * max(0.0, min(ratio / 100.0, 1.0)))
        if fill_width > 0:
            pygame.draw.rect(
                screen,
                progress_color,
                (bar_x, bar_y, fill_width, bar_height),
                border_radius=6,
            )

        for item in layout["distribution_lines"]:
            text = item["font"].render(item["text"], True, item["color"])
            screen.blit(text, (text_x, item["y"]))

    def _compute_layout(self, overview: Dict[str, float]) -> Dict[str, Any]:
        self._init_fonts()
        title_block = 34
        stat_gap = 17
        progress_text_gap = 16
        progress_bar_gap = 14
        distribution_gap = 14

        top_y = self.y + title_block + 6
        stat_lines = [
            {
                "text": f"格子总数: {overview['total']}  |  平均残值: {overview['avg']:.1f}",
                "color": self.TEXT_PRIMARY,
                "font": self._strong_font,
                "y": top_y,
            },
            {
                "text": f"最小/最大残值: {overview['min']:.1f} / {overview['max']:.1f}",
                "color": self.TEXT_PRIMARY,
                "font": self._strong_small_font,
                "y": top_y + stat_gap,
            },
        ]

        progress_text_y = stat_lines[-1]["y"] + stat_gap
        progress_bar_y = progress_text_y + progress_text_gap
        distribution_start_y = progress_bar_y + progress_bar_gap

        distribution_lines = [
            {
                "text": f"低残值(0-30): {overview['low']}   中残值(30-70): {overview['medium']}",
                "color": self.TEXT_SECONDARY,
                "font": self._strong_small_font,
                "y": distribution_start_y,
            },
            {
                "text": f"高残值(70-100): {overview['high']}",
                "color": self.DANGER,
                "font": self._strong_small_font,
                "y": distribution_start_y + distribution_gap,
            },
        ]

        last_line_bottom = distribution_lines[-1]["y"] + self._strong_small_font.get_height()
        bottom_padding = self.y + self.height - last_line_bottom

        return {
            "stat_lines": stat_lines,
            "progress_text_y": progress_text_y,
            "progress_bar_y": progress_bar_y,
            "distribution_lines": distribution_lines,
            "bottom_padding": bottom_padding,
        }

    def _build_overview(self, data: Dict[str, Any]) -> Dict[str, float] | None:
        grid_stats = self._stats_from_grid(data.get("grid_data"))
        if grid_stats is not None:
            return grid_stats
        return self._stats_from_distribution(data)

    def _stats_from_grid(self, grid_data: Any) -> Dict[str, float] | None:
        cells = getattr(grid_data, "cells", None)
        if not cells:
            return None

        entropies = [float(getattr(cell, "entropy", 0.0) or 0.0) for cell in cells]
        total = len(entropies)
        low = sum(1 for value in entropies if value < 30.0)
        medium = sum(1 for value in entropies if 30.0 <= value < 70.0)
        high = total - low - medium
        scan_ratio = (low / total * 100.0) if total > 0 else 0.0

        return {
            "total": total,
            "avg": sum(entropies) / total if total > 0 else 0.0,
            "min": min(entropies) if entropies else 0.0,
            "max": max(entropies) if entropies else 0.0,
            "scanned": low,
            "scan_ratio": scan_ratio,
            "low": low,
            "medium": medium,
            "high": high,
        }

    def _stats_from_distribution(self, data: Dict[str, Any]) -> Dict[str, float] | None:
        entropy_distribution = list(data.get("entropy_distribution") or [])
        entropy_bins = list(data.get("entropy_bins") or [])
        scan_progress_history = list(data.get("scan_progress_history") or [])
        if not entropy_distribution or not entropy_bins:
            return None

        latest = entropy_distribution[-1]
        hist = self._extract_histogram(latest)
        if not hist:
            return None

        total = sum(hist)
        low = self._sum_bucket_range(entropy_bins, hist, 0.0, 30.0)
        medium = self._sum_bucket_range(entropy_bins, hist, 30.0, 70.0)
        high = total - low - medium

        weighted_sum = 0.0
        weighted_min = None
        weighted_max = None
        for index, count in enumerate(hist):
            if count <= 0:
                continue
            lower = float(entropy_bins[index])
            upper = float(entropy_bins[index + 1]) if index + 1 < len(entropy_bins) else lower + 5.0
            midpoint = (lower + upper) / 2.0
            weighted_sum += midpoint * count
            weighted_min = lower if weighted_min is None else min(weighted_min, lower)
            weighted_max = upper if weighted_max is None else max(weighted_max, upper)

        latest_progress = scan_progress_history[-1] if scan_progress_history else None
        scan_ratio = float(latest_progress[1]) if latest_progress else (low / total * 100.0 if total > 0 else 0.0)
        scanned = int(latest_progress[2]) if latest_progress else low

        return {
            "total": total,
            "avg": weighted_sum / total if total > 0 else 0.0,
            "min": weighted_min if weighted_min is not None else 0.0,
            "max": weighted_max if weighted_max is not None else 0.0,
            "scanned": scanned,
            "scan_ratio": scan_ratio,
            "low": low,
            "medium": medium,
            "high": high,
        }

    def _extract_histogram(self, entry: Any) -> List[int]:
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            hist = entry[1]
            if isinstance(hist, (list, tuple)):
                return [int(value) for value in hist]
        return []

    def _sum_bucket_range(
        self,
        bins: Iterable[float],
        hist: Iterable[int],
        lower_bound: float,
        upper_bound: float,
    ) -> int:
        bins_list = list(bins)
        hist_list = list(hist)
        total = 0
        for index, count in enumerate(hist_list):
            bucket_lower = float(bins_list[index])
            bucket_upper = float(bins_list[index + 1]) if index + 1 < len(bins_list) else bucket_lower + 5.0
            if bucket_lower >= upper_bound or bucket_upper <= lower_bound:
                continue
            total += int(count)
        return total
