"""电量信息面板 - 显示所有无人机的电池状态"""
import sys
import os
import pygame
from typing import Dict, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from multirotor.Visualization.panel_system import BasePanel


class BatteryPanel(BasePanel):
    """电量状态面板 - 显示所有无人机的电量信息"""
    
    def __init__(self, width: int = 370, height: int = 260):
        super().__init__("battery", width, height)
    
    def draw(self, screen: pygame.Surface, data: Dict[str, Any]):
        """绘制电量信息"""
        self._init_fonts()
        
        # 背景和边框
        self.draw_panel_background(screen, border_color=self.YELLOW)
        
        # 标题
        y_offset = self.draw_title(screen, "🔋 电量状态", self.YELLOW)
        
        # 获取电量数据
        battery_data = data.get('battery_data', {})
        if not battery_data:
            hint_text = self._font.render("等待电量数据...", True, self.GRAY)
            hint_rect = hint_text.get_rect(center=(self.x + self.width // 2,
                                                   self.y + self.height // 2))
            screen.blit(hint_text, hint_rect)
            return
        
        text_x = self.x + 15
        y = self.y + y_offset
        
        # 状态颜色映射
        status_colors = {
            'normal': self.GREEN,      # 正常 >= 4.0V
            'warning': self.YELLOW,    # 警告 3.7-4.0V
            'low': self.ORANGE,        # 低电量 3.5-3.7V
            'critical': self.RED,      # 严重 3.0-3.5V
            'empty': self.DARK_GRAY    # 耗尽 < 3.0V
        }
        
        # 状态中文名
        status_names = {
            'normal': '正常',
            'warning': '警告',
            'low': '低电',
            'critical': '危险',
            'empty': '耗尽'
        }
        
        # 遍历每个无人机的电量
        for i, (drone_name, battery_info) in enumerate(battery_data.items()):
            if y + 50 > self.y + self.height - 10:  # 防止溢出
                remaining = len(battery_data) - i
                if remaining > 0:
                    more_text = self._small_font.render(f"... 还有{remaining}架无人机", 
                                                       True, self.GRAY)
                    screen.blit(more_text, (text_x, y))
                break
            
            # 获取电量信息
            voltage = battery_info.get('voltage', 4.2)
            percentage = battery_info.get('remaining_percentage', 100.0)
            status = battery_info.get('status', 'normal')
            is_crazyflie = battery_info.get('crazyflieMirror', False)
            
            status_color = status_colors.get(status, self.WHITE)
            
            # 无人机名称
            drone_type = " 🚁" if is_crazyflie else ""
            name_text = self._font.render(f"{drone_name}{drone_type}", True, self.WHITE)
            screen.blit(name_text, (text_x, y))
            
            # 电压和百分比
            voltage_text = self._small_font.render(
                f"{voltage:.2f}V ({percentage:.0f}%)", 
                True, status_color
            )
            screen.blit(voltage_text, (text_x + 90, y + 2))
            
            # 状态标签
            status_name = status_names.get(status, status)
            status_text = self._small_font.render(status_name, True, status_color)
            screen.blit(status_text, (text_x + 200, y + 2))
            
            y += 18
            
            # 电量条
            bar_x = text_x
            bar_y = y
            bar_width = self.width - 30
            bar_height = 8
            
            # 背景条
            pygame.draw.rect(screen, self.DARK_GRAY, (bar_x, bar_y, bar_width, bar_height))
            
            # 电量填充条
            fill_width = int(bar_width * (percentage / 100))
            if fill_width > 0:
                pygame.draw.rect(screen, status_color, (bar_x, bar_y, fill_width, bar_height))
            
            # 边框
            pygame.draw.rect(screen, self.WHITE, (bar_x, bar_y, bar_width, bar_height), 1)
            
            y += 16
        
        # 底部统计信息
        if len(battery_data) > 0:
            y = self.y + self.height - 35
            pygame.draw.line(screen, self.GRAY, 
                           (text_x, y), 
                           (self.x + self.width - 15, y), 1)
            y += 8
            
            # 计算平均电量
            avg_percentage = sum(b.get('remaining_percentage', 0) 
                               for b in battery_data.values()) / len(battery_data)
            avg_color = self.GREEN if avg_percentage > 50 else (
                        self.YELLOW if avg_percentage > 30 else self.RED)
            
            summary_text = self._font.render(
                f"平均电量: {avg_percentage:.1f}%  |  {len(battery_data)}架无人机",
                True, avg_color
            )
            screen.blit(summary_text, (text_x, y))
