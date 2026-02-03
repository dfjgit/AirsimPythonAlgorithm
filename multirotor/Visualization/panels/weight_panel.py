"""权重显示面板 - 显示APF权重系数"""
import sys
import os
import pygame
from typing import Dict, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from multirotor.Visualization.panel_system import BasePanel


class WeightPanel(BasePanel):
    """APF权重显示面板"""
    
    def __init__(self, width: int = 370, height: int = 180):
        super().__init__("weight", width, height)
    
    def draw(self, screen: pygame.Surface, data: Dict[str, Any]):
        """绘制权重信息"""
        self._init_fonts()
        
        # 获取权重数据
        weights = data.get('weights', {})
        if not weights:
            return
        
        # 判断是否使用DQN
        use_dqn = data.get('use_dqn', False)
        
        # 背景和边框
        border_color = self.GREEN if use_dqn else self.CYAN
        self.draw_panel_background(screen, border_color=border_color)
        
        # 标题
        title_text = "⚙️ DQN预测权重" if use_dqn else "⚙️ APF权重系数"
        title_color = self.GREEN if use_dqn else self.CYAN
        y_offset = self.draw_title(screen, title_text, title_color)
        
        text_x = self.x + 15
        y = self.y + y_offset
        
        # 权重信息
        weight_info = [
            ("α1 排斥", weights.get('repulsionCoefficient', 0)),
            ("α2 熵值", weights.get('entropyCoefficient', 0)),
            ("α3 距离", weights.get('distanceCoefficient', 0)),
            ("α4 Leader", weights.get('leaderRangeCoefficient', 0)),
            ("α5 方向", weights.get('directionRetentionCoefficient', 0))
        ]
        
        for name, value in weight_info:
            # 权重名称和值
            text = self._font.render(f"{name}: {value:.2f}", True, self.LIGHT_BLUE)
            screen.blit(text, (text_x, y))
            
            # 权重条
            bar_x = self.x + 130
            bar_y = y + 3
            bar_width = 120
            bar_height = 10
            
            # 背景条
            pygame.draw.rect(screen, self.GRAY, (bar_x, bar_y, bar_width, bar_height))
            
            # 填充条(范围0.5-5.0)
            fill_ratio = min((value - 0.5) / 4.5, 1.0)
            fill_width = int(bar_width * max(0, fill_ratio))
            if fill_width > 0:
                # 颜色根据值变化
                if value < 1.5:
                    color = self.GREEN
                elif value < 3.0:
                    color = self.YELLOW
                else:
                    color = self.RED
                pygame.draw.rect(screen, color, (bar_x, bar_y, fill_width, bar_height))
            
            # 边框
            pygame.draw.rect(screen, self.WHITE, (bar_x, bar_y, bar_width, bar_height), 1)
            
            # 数值
            value_text = self._small_font.render(f"{value:.2f}", True, self.WHITE)
            screen.blit(value_text, (bar_x + bar_width + 5, y))
            
            y += 20
