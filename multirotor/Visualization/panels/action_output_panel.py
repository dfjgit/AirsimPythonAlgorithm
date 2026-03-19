import os
import re
import sys
from typing import Any, Dict, Iterable, List, Tuple

import pygame

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)
from multirotor.Visualization.panel_system import BasePanel


class ActionOutputPanel(BasePanel):
    def __init__(self, width: int = 370, height: int = 260):
        super().__init__("action_output", width, height)
        self.action_names = {
            0: "上升",
            1: "下降",
            2: "左移",
            3: "右移",
            4: "前进",
            5: "后退",
        }

    def _format_float(self, value: Any, suffix: str = "", digits: int = 2) -> str:
        if value is None:
            return "--"
        try:
            return f"{float(value):.{digits}f}{suffix}"
        except Exception:
            return str(value)

    def _translate_done_reason(self, reason: Any) -> str:
        if reason in (None, "", "-"):
            return "-"

        text = str(reason)
        patterns = [
            (r"^Timeout \((.+)\)$", r"超时结束 (\1)"),
            (r"^Target Scan Ratio Reached \((.+)\)$", r"达到目标扫描率 (\1)"),
            (r"^Collision Limit Reached \((.+)\)$", r"达到碰撞上限 (\1)"),
            (r"^Drone (.+) Out of Range Reset \((.+)\)$", r"无人机 \1 越界重置 (\2)"),
            (r"^Drone (.+) Out of Range Too Long \((.+)\)$", r"无人机 \1 越界过久 (\2)"),
            (r"^Drone (.+) Severe Out of Range \((.+)\)$", r"无人机 \1 严重越界 (\2)"),
            (r"^Drone (.+) Battery Empty$", r"无人机 \1 电量耗尽"),
            (r"^Drone (.+) Landed \(Physics\)$", r"无人机 \1 已落地(物理)"),
        ]
        for pattern, replacement in patterns:
            translated = re.sub(pattern, replacement, text)
            if translated != text:
                return translated
        return text

    def _format_action(self, action: Any) -> str:
        if action is None:
            return "--"
        try:
            action_id = int(action)
        except Exception:
            return str(action)
        return f"{action_id}: {self.action_names.get(action_id, '未知动作')}"

    def _iter_drone_rows(self, stats: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
        drone_actions = stats.get("per_drone_actions") or {}
        if isinstance(drone_actions, dict) and drone_actions:
            for drone_name in sorted(drone_actions.keys()):
                row = drone_actions.get(drone_name) or {}
                if isinstance(row, dict):
                    yield drone_name, row
            return

        drone_name = stats.get("drone_name")
        if drone_name:
            yield str(drone_name), stats

    def draw(self, screen: pygame.Surface, data: Dict[str, Any]):
        self._init_fonts()

        self.draw_panel_background(screen, border_color=self.CYAN)
        y_offset = self.draw_title(screen, "动作输出", self.CYAN)

        stats = data.get("current_training_stats", {}) or {}
        if not stats:
            hint_text = self._font.render("等待动作数据...", True, self.GRAY)
            hint_rect = hint_text.get_rect(
                center=(self.x + self.width // 2, self.y + self.height // 2)
            )
            screen.blit(hint_text, hint_rect)
            return

        text_x = self.x + 15
        y = self.y + y_offset

        timestep = int(stats.get("timestep", 0) or 0)
        episode_steps = int(stats.get("current_episode_steps", 0) or 0)
        step_reward = self._format_float(stats.get("current_step_reward"))
        episode_reward = self._format_float(stats.get("current_episode_reward"))
        done_reason = self._translate_done_reason(stats.get("last_done_reason"))

        summary_lines = [
            (f"总步数: {timestep} | 当前轮步数: {episode_steps}", self.TEXT_SECONDARY, self._small_font),
            (f"步奖励: {step_reward} | 当前轮奖励: {episode_reward}", self.TEXT_PRIMARY, self._small_font),
        ]
        for content, color, font in summary_lines:
            rendered = font.render(content, True, color)
            screen.blit(rendered, (text_x, y))
            y += 18

        y += 2
        drone_rows: List[Tuple[str, Dict[str, Any]]] = list(self._iter_drone_rows(stats))
        if not drone_rows:
            rendered = self._font.render("暂无无人机动作信息", True, self.GRAY)
            screen.blit(rendered, (text_x, y))
            return

        for drone_name, row in drone_rows:
            action_text = self._format_action(row.get("last_action"))
            leader_distance = self._format_float(row.get("leader_distance"), suffix="m")
            drone_reward = self._format_float(row.get("current_drone_reward"))
            out_of_range = bool(row.get("is_out_of_range", False))
            oob_steps = int(row.get("out_of_range_steps", 0) or 0)
            status_label = "越界" if out_of_range else "正常"
            status_color = self.DANGER if out_of_range else self.SUCCESS

            header = self._font.render(f"{drone_name}: {action_text}", True, self.WARNING)
            screen.blit(header, (text_x, y))
            y += 22

            detail = self._small_font.render(
                f"距Leader: {leader_distance} | 状态: {status_label} | 连续越界: {oob_steps}",
                True,
                status_color,
            )
            screen.blit(detail, (text_x + 8, y))
            y += 17

            reward_line = self._small_font.render(
                f"当前无人机奖励: {drone_reward}",
                True,
                self.TEXT_PRIMARY,
            )
            screen.blit(reward_line, (text_x + 8, y))
            y += 20

        if y < self.y + self.height - 20:
            footer = self._small_font.render(
                f"结束原因: {done_reason}",
                True,
                self.LIGHT_GRAY,
            )
            screen.blit(footer, (text_x, y))
