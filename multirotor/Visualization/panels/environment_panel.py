"""环境信息面板 - 显示网格统计、扫描进度等"""
import sys
import os
import pygame
from typing import Dict, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from multirotor.Visualization.panel_system import BasePanel


class EnvironmentPanel(BasePanel):
    """环境状态面板"""
    
    def __init__(self, width: int = 350, height: int = 180):
        super().__init__("environment", width, height)
    
    def draw(self, screen: pygame.Surface, data: Dict[str, Any]):
        """绘制环境信息"""
        self._init_fonts()
        
        # 背景和边框
        self.draw_panel_background(screen, border_color=self.GREEN)
        
        # 标题
        y_offset = self.draw_title(screen, "🌍 环境状态", self.GREEN)
        
        # 网格统计
        grid_stats = self._calculate_grid_stats(data.get('grid_data'))
        if grid_stats:
            text_x = self.x + 15
            y = self.y + y_offset
            
            # 网格单元数
            text1 = self._font.render(f"网格单元: {grid_stats['total']}", True, self.WHITE)
            screen.blit(text1, (text_x, y))
            y += 20
            
            # 平均熵值
            text2 = self._font.render(f"平均熵值: {grid_stats['avg']:.1f}", True, self.WHITE)
            screen.blit(text2, (text_x, y))
            y += 20
            
            # 扫描进度
            scanned = grid_stats['scanned']
            total = grid_stats['total']
            ratio = grid_stats['scan_ratio']
            color = self.GREEN if ratio > 50 else self.YELLOW
            text3 = self._font.render(f"已扫描: {scanned}/{total} ({ratio:.1f}%)", True, color)
            screen.blit(text3, (text_x, y))
            y += 25
            
            # 进度条
            bar_x = text_x
            bar_y = y
            bar_width = self.width - 30
            bar_height = 12
            
            pygame.draw.rect(screen, self.DARK_GRAY, (bar_x, bar_y, bar_width, bar_height))
            fill_width = int(bar_width * (ratio / 100))
            if fill_width > 0:
                pygame.draw.rect(screen, self.GREEN, (bar_x, bar_y, fill_width, bar_height))
            pygame.draw.rect(screen, self.WHITE, (bar_x, bar_y, bar_width, bar_height), 1)
            y += 20
            
            # 无人机数量
            runtime_data = data.get('runtime_data', {})
            if runtime_data:
                drone_count = len(runtime_data)
                text4 = self._font.render(f"无人机数量: {drone_count}", True, self.CYAN)
                screen.blit(text4, (text_x, y))
    
    def _calculate_grid_stats(self, grid_data) -> Dict:
        """计算网格统计信息"""
        if not grid_data or not hasattr(grid_data, 'cells') or not grid_data.cells:
            return None
        
        total = len(grid_data.cells)
        total_entropy = sum(cell.entropy for cell in grid_data.cells)
        avg_entropy = total_entropy / total if total > 0 else 0
        
        scanned = sum(1 for cell in grid_data.cells if cell.entropy < 30)
        scan_ratio = (scanned / total * 100) if total > 0 else 0
        
        return {
            'total': total,
            'avg': avg_entropy,
            'scanned': scanned,
            'scan_ratio': scan_ratio
        }
