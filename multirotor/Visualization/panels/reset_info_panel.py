"""Reset info panel."""

import os
import sys
import time as pytime
from typing import Any, Dict, Tuple

import pygame

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)
from multirotor.Visualization.panel_system import BasePanel


class ResetInfoPanel(BasePanel):
    """Panel showing recent reset reasons and reset history."""

    def __init__(self, width: int = 370, height: int = 200):
        super().__init__("reset_info", width, height)
        self._last_reason = ""
        self._reason_show_time = 0.0
        self._flash_duration = 5.0

    def _is_time_limit_reason(self, reason: str) -> bool:
        reason_lower = reason.lower()
        return (
            "达到时长上限" in reason
            or "超时" in reason
            or "时间" in reason
            or "时长" in reason
            or "time_limit" in reason_lower
            or "max_elapsed" in reason_lower
            or "timeout" in reason_lower
        )

    def _is_collision_reason(self, reason: str) -> bool:
        reason_lower = reason.lower()
        return "碰撞" in reason or "collision" in reason_lower

    def _is_complete_reason(self, reason: str) -> bool:
        reason_lower = reason.lower()
        return "扫描完成" in reason or "完成" in reason or "scan" in reason_lower

    def _is_battery_reason(self, reason: str) -> bool:
        reason_lower = reason.lower()
        return "电量" in reason or "battery" in reason_lower

    def _is_out_of_range_reason(self, reason: str) -> bool:
        reason_lower = reason.lower()
        return "出圈" in reason or "range" in reason_lower or "distance" in reason_lower

    def _normalize_display_reason(self, reason: str) -> str:
        if self._is_time_limit_reason(reason):
            return "达到时长上限"
        return reason

    def _normalize_reason_category(self, reason: str) -> str:
        if self._is_time_limit_reason(reason):
            return "时长结束"
        if self._is_collision_reason(reason):
            return "碰撞"
        if self._is_complete_reason(reason):
            return "完成"
        if self._is_out_of_range_reason(reason):
            return "出圈"
        if self._is_battery_reason(reason):
            return "电量"
        return "其他"

    def _get_reason_icon(self, reason: str) -> str:
        if self._is_time_limit_reason(reason):
            return "⏱"
        if self._is_collision_reason(reason):
            return "💥"
        if self._is_complete_reason(reason):
            return "✅"
        if self._is_battery_reason(reason):
            return "🔋"
        if self._is_out_of_range_reason(reason):
            return "📍"
        return "📌"

    def _get_reason_color(self, reason: str) -> Tuple[int, int, int]:
        if self._is_collision_reason(reason):
            return self.RED
        if self._is_complete_reason(reason):
            return self.GREEN
        if self._is_battery_reason(reason):
            return self.ORANGE
        if self._is_time_limit_reason(reason):
            return self.CYAN
        if self._is_out_of_range_reason(reason):
            return self.YELLOW
        return self.GRAY

    def _format_time_ago(self, timestamp: float) -> str:
        elapsed = pytime.time() - timestamp
        if elapsed < 60:
            return f"{int(elapsed)}秒前"
        if elapsed < 3600:
            return f"{int(elapsed / 60)}分钟前"
        return f"{int(elapsed / 3600)}小时前"

    def _parse_history_entry(self, entry) -> Tuple[float, str, str, float]:
        if isinstance(entry, dict):
            return (
                float(entry.get("time", 0.0) or 0.0),
                str(entry.get("reason", "") or ""),
                str(entry.get("collision_object_name", "") or ""),
                float(entry.get("collision_penetration_depth", 0.0) or 0.0),
            )
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            return float(entry[0] or 0.0), str(entry[1] or ""), "", 0.0
        return 0.0, "", "", 0.0

    def _format_collision_detail(
        self, reason: str, object_name: str, penetration_depth: float
    ) -> str:
        if not self._is_collision_reason(reason):
            return ""
        if object_name:
            if penetration_depth > 0:
                return f"对象: {object_name}  深度: {penetration_depth:.3f}m"
            return f"对象: {object_name}"
        if penetration_depth > 0:
            return f"穿透深度: {penetration_depth:.3f}m"
        return ""

    def draw(self, screen: pygame.Surface, data: Dict[str, Any]):
        self._init_fonts()
        self.draw_panel_background(screen, border_color=self.PURPLE)

        y_offset = self.draw_title(screen, "训练重置记录", self.PURPLE)
        text_x = self.x + 15
        y = self.y + y_offset + 5

        last_reason = str(data.get("last_reset_reason", "") or "")
        last_reset_time = float(data.get("last_reset_time", 0.0) or 0.0)
        last_collision_object_name = str(
            data.get("last_collision_object_name", "") or ""
        )
        last_collision_penetration_depth = float(
            data.get("last_collision_penetration_depth", 0.0) or 0.0
        )
        reset_history = data.get("reset_history", []) or []

        if last_reason and last_reason != self._last_reason:
            self._last_reason = last_reason
            self._reason_show_time = pytime.time()

        reason_counts: Dict[str, int] = {}
        for entry in reset_history:
            _, reason, _, _ = self._parse_history_entry(entry)
            category = self._normalize_reason_category(reason)
            reason_counts[category] = reason_counts.get(category, 0) + 1

        if last_reason:
            icon = self._get_reason_icon(last_reason)
            color = self._get_reason_color(last_reason)
            display_reason = self._normalize_display_reason(last_reason)

            time_since_reset = pytime.time() - self._reason_show_time
            if time_since_reset < self._flash_duration:
                flash_alpha = int((1 - time_since_reset / self._flash_duration) * 128)
                flash_surface = pygame.Surface((self.width - 20, 30))
                flash_surface.fill(color)
                flash_surface.set_alpha(flash_alpha)
                screen.blit(flash_surface, (self.x + 10, y - 5))

            text = self._font.render(f"{icon} 最新: {display_reason}", True, color)
            screen.blit(text, (text_x, y))

            if last_reset_time > 0:
                time_text = self._small_font.render(
                    f"({self._format_time_ago(last_reset_time)})", True, self.GRAY
                )
                screen.blit(time_text, (text_x + 230, y + 2))

            collision_detail = self._format_collision_detail(
                last_reason,
                last_collision_object_name,
                last_collision_penetration_depth,
            )
            if collision_detail:
                y += 18
                detail_text = self._small_font.render(
                    f"    {collision_detail}", True, self.LIGHT_BLUE
                )
                screen.blit(detail_text, (text_x, y))

        y += 35
        pygame.draw.line(
            screen, self.GRAY, (self.x + 10, y), (self.x + self.width - 10, y), 1
        )
        y += 10

        if reason_counts:
            stats_title = self._small_font.render(
                "重置原因统计:", True, self.LIGHT_BLUE
            )
            screen.blit(stats_title, (text_x, y))
            y += 22

            icon_map = {
                "时长结束": "⏱",
                "碰撞": "💥",
                "完成": "✅",
                "出圈": "📍",
                "电量": "🔋",
                "其他": "📌",
            }
            color_map = {
                "时长结束": self.CYAN,
                "碰撞": self.RED,
                "完成": self.GREEN,
                "出圈": self.YELLOW,
                "电量": self.ORANGE,
                "其他": self.GRAY,
            }
            order = ["时长结束", "碰撞", "完成", "出圈", "电量", "其他"]
            stat_x = text_x + 10
            ordered_counts = [
                (name, reason_counts[name]) for name in order if name in reason_counts
            ]

            for i, (category, count) in enumerate(ordered_counts):
                stat_text = f"{icon_map.get(category, '📌')}{category}:{count}"
                text = self._small_font.render(
                    stat_text, True, color_map.get(category, self.GRAY)
                )
                screen.blit(text, (stat_x + (i % 3) * 110, y + (i // 3) * 18))
        else:
            no_data = self._small_font.render("暂无重置记录", True, self.GRAY)
            screen.blit(no_data, (text_x, y))
            y += 18

        y += 45

        if reset_history:
            history_title = self._small_font.render("最近重置:", True, self.LIGHT_BLUE)
            screen.blit(history_title, (text_x, y))
            y += 18

            recent = reset_history[-3:] if len(reset_history) >= 3 else reset_history
            for entry in reversed(recent):
                ts, reason, collision_object_name, collision_penetration_depth = (
                    self._parse_history_entry(entry)
                )
                display_reason = self._normalize_display_reason(reason)
                if len(display_reason) > 25:
                    display_reason = display_reason[:25] + "..."

                history_text = f"  {self._get_reason_icon(reason)} {display_reason}"
                text = self._small_font.render(history_text, True, self.GRAY)
                screen.blit(text, (text_x, y))

                time_text = self._small_font.render(
                    self._format_time_ago(ts), True, self.DARK_GRAY
                )
                screen.blit(time_text, (self.x + self.width - 70, y))
                y += 16

                collision_detail = self._format_collision_detail(
                    reason, collision_object_name, collision_penetration_depth
                )
                if collision_detail:
                    detail_text = self._small_font.render(
                        f"    {collision_detail[:34]}", True, self.DARK_GRAY
                    )
                    screen.blit(detail_text, (text_x, y))
                    y += 14
