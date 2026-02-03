"""高层网格面板 - 显示5x5任务区域划分"""
import sys
import os
import pygame
from typing import Dict, Any, List, Tuple
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from multirotor.Visualization.panel_system import BasePanel

try:
    from multirotor.Algorithm.Vector3 import Vector3
except:
    Vector3 = None


class HierarchicalGridPanel(BasePanel):
    """5x5高层任务区域面板"""
    
    def __init__(self, width: int = 370, height: int = 300):
        super().__init__("hierarchical_grid", width, height)
    
    def draw(self, screen: pygame.Surface, data: Dict[str, Any]):
        """绘制5x5网格信息"""
        self._init_fonts()
        
        # 背景和边框
        self.draw_panel_background(screen, border_color=self.YELLOW)
        
        # 标题
        y_offset = self.draw_title(screen, "🎯 高层动作空间 (5x5)", self.YELLOW)
        
        # 获取高层动作历史
        hl_action_history = data.get('hl_action_history', [])
        
        text_x = self.x + 15
        y = self.y + y_offset
        
        # 显示最近的高层动作
        if hl_action_history:
            text = self._font.render("最近决策:", True, self.WHITE)
            screen.blit(text, (text_x, y))
            y += 25
            
            # 显示最近10个动作
            for i, action_info in enumerate(list(hl_action_history)[-10:]):
                if len(action_info) >= 3:
                    step, action, drone = action_info[:3]
                    region = f"R{action // 5}C{action % 5}"
                    
                    # 根据无人机名称选择颜色
                    color = self._get_drone_color(drone)
                    
                    text = self._small_font.render(
                        f"[{step:4d}] {drone}: {action:2d} ({region})",
                        True, color
                    )
                    screen.blit(text, (text_x, y))
                    y += 18
        else:
            text = self._font.render("等待高层决策...", True, self.GRAY)
            screen.blit(text, (text_x, y))
            y += 20
        
        # 绘制网格示意图(简化版)
        y += 10
        grid_size = 40
        grid_start_x = text_x + 10
        grid_start_y = y
        
        # 绘制5x5小格子
        for row in range(5):
            for col in range(5):
                rect_x = grid_start_x + col * grid_size
                rect_y = grid_start_y + row * grid_size
                
                # 绘制格子
                pygame.draw.rect(screen, self.DARK_GRAY, 
                               (rect_x, rect_y, grid_size-2, grid_size-2))
                pygame.draw.rect(screen, self.YELLOW,
                               (rect_x, rect_y, grid_size-2, grid_size-2), 1)
                
                # 绘制格子编号
                action_id = row * 5 + col
                text = self._small_font.render(str(action_id), True, self.LIGHT_GRAY)
                text_rect = text.get_rect(center=(rect_x + grid_size//2 - 1, 
                                                  rect_y + grid_size//2 - 1))
                screen.blit(text, text_rect)
    
    def _get_drone_color(self, drone_name: str) -> Tuple[int, int, int]:
        """为无人机分配颜色"""
        colors = [
            self.GREEN, self.CYAN, self.MAGENTA,
            self.ORANGE, self.PURPLE, (255, 192, 203),
            (0, 255, 127), (255, 215, 0)
        ]
        
        # 简单哈希
        hash_val = sum(ord(c) for c in drone_name)
        return colors[hash_val % len(colors)]
