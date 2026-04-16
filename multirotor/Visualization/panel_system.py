
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import pygame


class BasePanel(ABC):
    def __init__(self, name: str, width: int = 350, height: int = 200):
        self.name = name
        self.width = width
        self.height = height
        self.x = 0
        self.y = 0
        self.visible = True

        self.SCREEN_BACKGROUND = (220, 225, 232)
        self.PANEL_BACKGROUND = (232, 236, 241)
        self.PANEL_BACKGROUND_SOFT = (225, 230, 236)
        self.PANEL_BORDER = (155, 170, 188)
        self.DIVIDER = (188, 198, 210)

        self.TEXT_PRIMARY = (44, 52, 64)
        self.TEXT_SECONDARY = (95, 106, 123)
        self.TEXT_MUTED = (137, 147, 162)

        self.SUCCESS = (87, 137, 114)
        self.WARNING = (176, 140, 82)
        self.DANGER = (178, 108, 101)
        self.INFO = (86, 133, 166)
        self.VIOLET = (126, 112, 170)
        self.ORANGE = (189, 133, 94)
        self.TEAL = (84, 145, 135)
        self.SKY = (102, 145, 180)
        self.MINT = (112, 168, 150)
        self.ROSE = (165, 111, 126)

        self.BLACK = self.SCREEN_BACKGROUND
        self.WHITE = self.TEXT_PRIMARY
        self.RED = self.DANGER
        self.GREEN = self.SUCCESS
        self.BLUE = self.SKY
        self.YELLOW = self.WARNING
        self.CYAN = self.INFO
        self.MAGENTA = self.ROSE
        self.PURPLE = self.VIOLET
        self.GRAY = self.DIVIDER
        self.LIGHT_GRAY = self.TEXT_SECONDARY
        self.DARK_GRAY = self.TEXT_MUTED
        self.LIGHT_BLUE = (201, 219, 232)

        self._font = None
        self._small_font = None
        self._title_font = None
        self._strong_font = None
        self._strong_small_font = None

    def _init_fonts(self):
        if self._font is None:
            try:
                font_names = ["SimHei", "Microsoft YaHei", "Arial"]
                self._font = pygame.font.SysFont(font_names, 14)
                self._small_font = pygame.font.SysFont(font_names, 12)
                self._title_font = pygame.font.SysFont(font_names, 18, bold=True)
                self._strong_font = pygame.font.SysFont(font_names, 14, bold=True)
                self._strong_small_font = pygame.font.SysFont(font_names, 12, bold=True)
            except Exception:
                self._font = pygame.font.Font(None, 14)
                self._small_font = pygame.font.Font(None, 12)
                self._title_font = pygame.font.Font(None, 18)
                self._strong_font = pygame.font.Font(None, 14)
                self._strong_small_font = pygame.font.Font(None, 12)

    @abstractmethod
    def draw(self, screen: pygame.Surface, data: Dict[str, Any]):
        pass

    def draw_panel_background(
        self,
        screen: pygame.Surface,
        border_color: Tuple[int, int, int] | None = None,
        alpha: int = 250,
    ):
        if border_color is None:
            border_color = self.PANEL_BORDER

        panel_rect = pygame.Rect(self.x, self.y, self.width, self.height)
        shadow = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        shadow.fill((26, 32, 44, 10))
        screen.blit(shadow, (self.x + 2, self.y + 3))

        surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        surface.fill((*self.PANEL_BACKGROUND, alpha))
        screen.blit(surface, (self.x, self.y))
        pygame.draw.rect(screen, border_color, panel_rect, 2, border_radius=10)

    def draw_title(
        self,
        screen: pygame.Surface,
        title: str,
        color: Tuple[int, int, int] | None = None,
    ) -> int:
        self._init_fonts()
        color = color or self.TEXT_PRIMARY
        text = self._title_font.render(title, True, color)
        screen.blit(text, (self.x + 14, self.y + 12))
        return 34

    def draw_divider(self, screen: pygame.Surface, y: int):
        pygame.draw.line(screen, self.DIVIDER, (self.x + 12, y), (self.x + self.width - 12, y), 1)

    def update_data(self, data: Dict[str, Any]):
        return None


class PanelManager:
    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        left_panel_width: int = 380,
        right_panel_width: int = 380,
    ):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.left_panel_width = left_panel_width
        self.right_panel_width = right_panel_width
        self.panels: Dict[str, BasePanel] = {}
        self.panel_order: List[str] = []
        self.margin = 10
        self.row_gap = 10
        self.layout_areas = {
            "top_left": (10, 10),
            "top_right": (screen_width - right_panel_width + 10, 10),
            "bottom_left": (10, screen_height - 250),
            "bottom_right": (screen_width - right_panel_width + 10, screen_height - 250),
        }

    def register_panel(self, panel: BasePanel, position: str = "top_right"):
        if panel.name in self.panels:
            print(f"Warning: panel '{panel.name}' already exists and will be replaced")
        self.panels[panel.name] = panel
        if panel.name not in self.panel_order:
            self.panel_order.append(panel.name)
        if position == "auto":
            self._auto_layout()
        elif position in self.layout_areas:
            panel.x, panel.y = self.layout_areas[position]

    def unregister_panel(self, panel_name: str):
        if panel_name in self.panels:
            del self.panels[panel_name]
        if panel_name in self.panel_order:
            self.panel_order.remove(panel_name)

    def _auto_layout(self):
        visible_panels = [
            self.panels[name]
            for name in self.panel_order
            if name in self.panels and self.panels[name].visible
        ]
        if not visible_panels:
            return

        left_panels: List[BasePanel] = []
        right_panels: List[BasePanel] = []
        left_height = 0
        right_height = 0
        for panel in sorted(visible_panels, key=lambda p: p.height, reverse=True):
            if left_height <= right_height:
                left_panels.append(panel)
                left_height += panel.height + self.row_gap
            else:
                right_panels.append(panel)
                right_height += panel.height + self.row_gap

        current_y = self.margin
        for panel in left_panels:
            panel.width = min(panel.width, self.left_panel_width - 2 * self.margin)
            panel.x = self.margin
            panel.y = current_y
            current_y += panel.height + self.row_gap

        current_y = self.margin
        right_start_x = self.screen_width - self.right_panel_width + self.margin
        for panel in right_panels:
            panel.width = min(panel.width, self.right_panel_width - 2 * self.margin)
            panel.x = right_start_x
            panel.y = current_y
            current_y += panel.height + self.row_gap

    def draw_all_panels(self, screen: pygame.Surface, data: Dict[str, Any]):
        for name in self.panel_order:
            panel = self.panels.get(name)
            if panel and panel.visible:
                previous_clip = screen.get_clip()
                try:
                    screen.set_clip(pygame.Rect(panel.x, panel.y, panel.width, panel.height))
                    panel.draw(screen, data)
                except Exception as exc:
                    print(f"Failed to draw panel '{name}': {exc}")
                finally:
                    try:
                        screen.set_clip(previous_clip)
                    except Exception:
                        pass

    def update_all_panels(self, data: Dict[str, Any]):
        for panel in self.panels.values():
            try:
                panel.update_data(data)
            except Exception as exc:
                print(f"Failed to update panel '{panel.name}': {exc}")

    def set_panel_visibility(self, panel_name: str, visible: bool):
        if panel_name in self.panels:
            self.panels[panel_name].visible = visible
            self._auto_layout()

    def get_panel(self, panel_name: str) -> Optional[BasePanel]:
        return self.panels.get(panel_name)
