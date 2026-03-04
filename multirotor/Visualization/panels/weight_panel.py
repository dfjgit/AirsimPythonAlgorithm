"""权重显示面板 - 显示APF权重系数"""
import sys
import os
import pygame
from typing import Dict, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from multirotor.Visualization.panel_system import BasePanel


class WeightPanel(BasePanel):
    """APF权重显示面板"""
    
    def __init__(self, width: int = 370, height: int = 280):
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
        
        # APF核心权重
        core_weights = [
            ("α1 排斥", weights.get('repulsionCoefficient', 0)),
            ("α2 熵值", weights.get('entropyCoefficient', 0)),
            ("α3 距离", weights.get('distanceCoefficient', 0)),
            ("α4 Leader", weights.get('leaderRangeCoefficient', 0)),
            ("α5 方向", weights.get('directionRetentionCoefficient', 0))
        ]
        
        # 避障参数
        obstacle_weights = [
            ("避障距离", weights.get('obstacleRepulsionDistance', 15.0)),
            ("避障系数", weights.get('obstacleRepulsionCoefficient', 5.0)),
            ("禁飞区距离", weights.get('restrictedZoneDistance', 15.0)),
            ("禁飞区系数", weights.get('restrictedZoneCoefficient', 5.0))
        ]
        
        # 绘制APF核心权重
        for name, value in core_weights:
            self._draw_weight_row(screen, text_x, y, name, value, min_val=0.5, max_val=5.0)
            y += 20
        
        # 分隔线
        y += 5
        pygame.draw.line(screen, self.GRAY, (self.x + 10, y), (self.x + self.width - 10, y), 1)
        y += 10
        
        # 绘制避障参数（使用不同的颜色范围）
        for name, value in obstacle_weights:
            self._draw_weight_row(screen, text_x, y, name, value, min_val=0, max_val=30, is_obstacle=True)
            y += 20
    
    def _draw_weight_row(self, screen: pygame.Surface, x: int, y: int, 
                         name: str, value: float, min_val: float = 0.5, 
                         max_val: float = 5.0, is_obstacle: bool = False):
        """绘制单行权重信息"""
        # 权重名称
        text = self._font.render(f"{name}:", True, self.LIGHT_BLUE)
        screen.blit(text, (x, y))
        
        # 权重条
        bar_x = self.x + 100
        bar_y = y + 3
        bar_width = 150
        bar_height = 10
        
        # 背景条
        pygame.draw.rect(screen, self.GRAY, (bar_x, bar_y, bar_width, bar_height))
        
        # 填充条
        range_val = max_val - min_val
        fill_ratio = min((value - min_val) / range_val, 1.0) if range_val > 0 else 0
        fill_width = int(bar_width * max(0, fill_ratio))
        
        if fill_width > 0:
            if is_obstacle:
                color = self.ORANGE if value > 10 else self.YELLOW if value > 5 else self.GREEN
            else:
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
        value_text = self._small_font.render(f"{value:.1f}", True, self.WHITE)
        screen.blit(value_text, (bar_x + bar_width + 5, y))
