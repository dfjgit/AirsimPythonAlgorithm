import sys
import os
import pygame
from typing import Dict, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from multirotor.Visualization.panel_system import BasePanel


class ActionOutputPanel(BasePanel):
    def __init__(self, width: int = 370, height: int = 160):
        super().__init__("action_output", width, height)
        self.action_names = {
            0: "前进",
            1: "后退",
            2: "左移",
            3: "右移",
            4: "上升",
            5: "下降",
        }

    def draw(self, screen: pygame.Surface, data: Dict[str, Any]):
        self._init_fonts()

        self.draw_panel_background(screen, border_color=self.CYAN)
        y_offset = self.draw_title(screen, "🧭 当前动作输出", self.CYAN)

        stats = data.get('current_training_stats', {})
        if not stats:
            hint_text = self._font.render("等待动作输出...", True, self.GRAY)
            hint_rect = hint_text.get_rect(center=(self.x + self.width // 2,
                                                   self.y + self.height // 2))
            screen.blit(hint_text, hint_rect)
            return

        last_action = stats.get('last_action', None)
        if last_action is None:
            action_str = "暂无"
            action_color = self.GRAY
        else:
            action_str = f"{last_action} - {self.action_names.get(int(last_action), '未知')}"
            action_color = self.YELLOW

        timestep = stats.get('timestep', None)
        ep_reward = stats.get('episode_reward', None)

        text_x = self.x + 15
        y = self.y + y_offset

        line1 = self._font.render(f"当前动作: {action_str}", True, action_color)
        screen.blit(line1, (text_x, y))
        y += 26

        if timestep is not None:
            line2 = self._small_font.render(f"timestep: {timestep}", True, self.LIGHT_GRAY)
            screen.blit(line2, (text_x, y))
            y += 18

        if ep_reward is not None:
            line3 = self._small_font.render(f"当前Episode累计奖励: {float(ep_reward):.2f}", True, self.LIGHT_GRAY)
            screen.blit(line3, (text_x, y))
            y += 18
