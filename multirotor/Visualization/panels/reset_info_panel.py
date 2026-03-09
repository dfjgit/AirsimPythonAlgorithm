"""重置信息面板 - 显示训练重置原因和历史"""

import sys
import os
import pygame
import time as pytime
from typing import Dict, Any, List, Tuple

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)
from multirotor.Visualization.panel_system import BasePanel


class ResetInfoPanel(BasePanel):
    """训练重置信息面板"""

    def __init__(self, width: int = 370, height: int = 200):
        super().__init__("reset_info", width, height)
        self._last_reason = ""
        self._reason_show_time = 0
        self._flash_duration = 5.0  # 重置原因闪烁显示5秒

    def _get_reason_icon(self, reason: str) -> str:
        """根据重置原因返回图标"""
        reason_lower = reason.lower()
        if "时间" in reason or "time" in reason_lower or "max_elapsed" in reason_lower:
            return "⏱️"
        elif "碰撞" in reason or "collision" in reason_lower:
            return "💥"
        elif "覆盖" in reason or "scan" in reason_lower or "coverage" in reason_lower:
            return "✅"
        elif "电量" in reason or "battery" in reason_lower:
            return "🔋"
        elif "距离" in reason or "distance" in reason_lower or "range" in reason_lower:
            return "📏"
        elif "leader" in reason_lower:
            return "👑"
        else:
            return "🔄"

    def _get_reason_color(self, reason: str) -> Tuple[int, int, int]:
        """根据重置原因返回颜色"""
        reason_lower = reason.lower()
        if "碰撞" in reason or "collision" in reason_lower:
            return self.RED
        elif "覆盖" in reason or "scan" in reason_lower or "coverage" in reason_lower:
            return self.GREEN
        elif "电量" in reason or "battery" in reason_lower:
            return self.ORANGE
        elif (
            "时间" in reason or "time" in reason_lower or "max_elapsed" in reason_lower
        ):
            return self.CYAN
        else:
            return self.YELLOW

    def _format_time_ago(self, timestamp: float) -> str:
        """格式化为相对时间"""
        elapsed = pytime.time() - timestamp
        if elapsed < 60:
            return f"{int(elapsed)}秒前"
        elif elapsed < 3600:
            return f"{int(elapsed / 60)}分钟前"
        else:
            return f"{int(elapsed / 3600)}小时前"

    def _parse_history_entry(self, entry) -> Tuple[float, str, str, float]:
        """兼容旧版 tuple 和新版 dict 的重置历史结构。"""
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
        if not reason:
            return ""
        reason_lower = reason.lower()
        if "碰撞" not in reason and "collision" not in reason_lower:
            return ""
        if object_name:
            if penetration_depth > 0:
                return f"对象: {object_name}  深度: {penetration_depth:.3f}m"
            return f"对象: {object_name}"
        if penetration_depth > 0:
            return f"穿透深度: {penetration_depth:.3f}m"
        return ""

    def draw(self, screen: pygame.Surface, data: Dict[str, Any]):
        """绘制重置信息"""
        self._init_fonts()

        # 背景和边框
        self.draw_panel_background(screen, border_color=self.PURPLE)

        # 标题
        y_offset = self.draw_title(screen, "🔄 训练重置记录", self.PURPLE)

        text_x = self.x + 15
        y = self.y + y_offset + 5

        # 获取重置信息
        last_reason = data.get("last_reset_reason", "")
        last_reset_time = data.get("last_reset_time", 0)
        last_collision_object_name = data.get("last_collision_object_name", "")
        last_collision_penetration_depth = float(
            data.get("last_collision_penetration_depth", 0.0) or 0.0
        )
        reset_history = data.get("reset_history", [])

        # 如果有新的重置原因，更新显示时间
        if last_reason and last_reason != self._last_reason:
            self._last_reason = last_reason
            self._reason_show_time = pytime.time()

        # 统计重置原因
        reason_counts = {}
        if reset_history:
            for entry in reset_history:
                ts, reason, _, _ = self._parse_history_entry(entry)
                # 简化原因分类
                reason_lower = reason.lower()
                if "时间" in reason or "max_elapsed" in reason_lower:
                    category = "超时"
                elif "碰撞" in reason or "collision" in reason_lower:
                    category = "碰撞"
                elif (
                    "覆盖" in reason
                    or "scan" in reason_lower
                    or "coverage" in reason_lower
                ):
                    category = "完成"
                elif "电量" in reason or "battery" in reason_lower:
                    category = "电量"
                else:
                    category = "其他"
                reason_counts[category] = reason_counts.get(category, 0) + 1

        # 显示当前重置原因（带闪烁效果）
        if last_reason:
            icon = self._get_reason_icon(last_reason)
            color = self._get_reason_color(last_reason)

            # 计算闪烁效果（新重置时闪烁）
            time_since_reset = pytime.time() - self._reason_show_time
            if time_since_reset < self._flash_duration:
                # 闪烁效果：背景色交替
                flash_alpha = int((1 - time_since_reset / self._flash_duration) * 128)
                flash_surface = pygame.Surface((self.width - 20, 30))
                flash_surface.fill(color)
                flash_surface.set_alpha(flash_alpha)
                screen.blit(flash_surface, (self.x + 10, y - 5))

            # 显示最新原因
            reason_text = f"{icon} 最新: {last_reason}"
            text = self._font.render(reason_text, True, color)
            screen.blit(text, (text_x, y))

            # 显示时间
            if last_reset_time > 0:
                time_ago = self._format_time_ago(last_reset_time)
                time_text = self._small_font.render(f"({time_ago})", True, self.GRAY)
                screen.blit(time_text, (text_x + 250, y + 2))

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

        # 分隔线
        pygame.draw.line(
            screen, self.GRAY, (self.x + 10, y), (self.x + self.width - 10, y), 1
        )
        y += 10

        # 显示重置统计
        if reason_counts:
            stats_title = self._small_font.render(
                "📊 重置原因统计:", True, self.LIGHT_BLUE
            )
            screen.blit(stats_title, (text_x, y))
            y += 22

            # 显示各原因的数量
            stat_x = text_x + 10
            for i, (category, count) in enumerate(reason_counts.items()):
                icon_map = {
                    "超时": "⏱️",
                    "碰撞": "💥",
                    "完成": "✅",
                    "电量": "🔋",
                    "其他": "📌",
                }
                color_map = {
                    "超时": self.CYAN,
                    "碰撞": self.RED,
                    "完成": self.GREEN,
                    "电量": self.ORANGE,
                    "其他": self.GRAY,
                }

                icon = icon_map.get(category, "📌")
                color = color_map.get(category, self.GRAY)

                stat_text = f"{icon}{category}:{count}"
                text = self._small_font.render(stat_text, True, color)
                screen.blit(text, (stat_x + (i % 3) * 110, y + (i // 3) * 18))
        else:
            no_data = self._small_font.render("暂无重置记录", True, self.GRAY)
            screen.blit(no_data, (text_x, y))
            y += 18

        y += 45

        # 显示最近几次重置历史
        if reset_history and len(reset_history) > 0:
            history_title = self._small_font.render(
                "📜 最近重置:", True, self.LIGHT_BLUE
            )
            screen.blit(history_title, (text_x, y))
            y += 18

            # 显示最近3条
            recent = reset_history[-3:] if len(reset_history) >= 3 else reset_history
            for entry in reversed(recent):
                ts, reason, collision_object_name, collision_penetration_depth = (
                    self._parse_history_entry(entry)
                )
                icon = self._get_reason_icon(reason)
                time_ago = self._format_time_ago(ts)
                # 截断过长的原因
                display_reason = reason[:25] + "..." if len(reason) > 25 else reason
                history_text = f"  {icon} {display_reason}"
                text = self._small_font.render(history_text, True, self.GRAY)
                screen.blit(text, (text_x, y))

                # 时间
                time_text = self._small_font.render(time_ago, True, self.DARK_GRAY)
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
