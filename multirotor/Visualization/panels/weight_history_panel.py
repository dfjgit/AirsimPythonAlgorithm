"""权重历史面板 - 显示APF权重随训练的变化趋势"""
import sys
import os
import pygame
from typing import Dict, Any, List
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from multirotor.Visualization.panel_system import BasePanel


class WeightHistoryPanel(BasePanel):
    """权重变化历史面板"""
    
    def __init__(self, width: int = 370, height: int = 250):
        super().__init__("weight_history", width, height)
        # 降采样参数
        self.max_display_points = 100  # 最多显示100个点
    
    def _adaptive_downsample(self, data: List[float], max_points: int = None) -> List[float]:
        """
        自适应降采样：近期高密度 + 远期低密度
        
        策略：
        - 最近50步：每步都保留（高清晰度）
        - 50-200步：每2步取1个
        - 200步以上：每5步取1个
        
        Args:
            data: 原始数据列表
            max_points: 最大点数（默认使用self.max_display_points）
            
        Returns:
            降采样后的数据
        """
        if max_points is None:
            max_points = self.max_display_points
        
        if len(data) <= max_points:
            return data
        
        result = []
        
        # 最近50步：全部保留
        recent_count = min(50, len(data))
        recent = list(data)[-recent_count:]
        
        # 中期数据：50-200步，每2步取1个
        if len(data) > 50:
            mid_start = max(0, len(data) - 200)
            mid_end = len(data) - 50
            if mid_end > mid_start:
                mid = list(data)[mid_start:mid_end:2]
                result.extend(mid)
        
        # 远期数据：200步以上，每5步取1个
        if len(data) > 200:
            far_end = max(0, len(data) - 200)
            if far_end > 0:
                far = list(data)[:far_end:5]
                result = far + result
        
        # 添加最近的数据
        result.extend(recent)
        
        # 确保不超过最大点数
        if len(result) > max_points:
            # 如果还是太多，均匀采样
            step = len(result) / max_points
            result = [result[int(i * step)] for i in range(max_points)]
        
        return result
    
    def draw(self, screen: pygame.Surface, data: Dict[str, Any]):
        """绘制权重历史曲线"""
        self._init_fonts()
        
        # 背景和边框
        self.draw_panel_background(screen, border_color=self.PURPLE)
        
        # 标题
        y_offset = self.draw_title(screen, "📊 权重变化历史", self.PURPLE)
        
        # 获取权重历史数据
        weight_history = data.get('weight_history', {})
        
        if not weight_history or not any(len(v) > 0 for v in weight_history.values()):
            hint_text = self._font.render("等待权重更新数据...", True, self.GRAY)
            hint_rect = hint_text.get_rect(center=(self.x + self.width // 2, 
                                                   self.y + self.height // 2))
            screen.blit(hint_text, hint_rect)
            return
        
        # 图表区域
        chart_margin_x = 50
        chart_margin_y = 35
        chart_width = self.width - chart_margin_x - 20
        chart_height = self.height - y_offset - chart_margin_y - 30
        chart_origin_x = self.x + chart_margin_x
        chart_origin_y = self.y + y_offset + chart_height
        
        # 绘制坐标轴
        pygame.draw.line(screen, self.LIGHT_GRAY,
                        (chart_origin_x, chart_origin_y),
                        (chart_origin_x + chart_width, chart_origin_y), 2)
        pygame.draw.line(screen, self.LIGHT_GRAY,
                        (chart_origin_x, chart_origin_y - chart_height),
                        (chart_origin_x, chart_origin_y), 2)
        
        # 权重名称和颜色
        weight_colors = {
            'repulsionCoefficient': (255, 107, 107),      # 红色
            'entropyCoefficient': (78, 205, 196),          # 青色
            'distanceCoefficient': (69, 183, 209),         # 蓝色
            'leaderRangeCoefficient': (255, 160, 122),     # 橙色
            'directionRetentionCoefficient': (152, 216, 200)  # 绿色
        }
        
        # 找到最大步数
        max_steps = max(len(v) for v in weight_history.values() if len(v) > 0)
        if max_steps == 0:
            return
        
        # 找到权重范围
        all_values = []
        for values in weight_history.values():
            all_values.extend(values)
        
        if not all_values:
            return
        
        min_weight = min(all_values)
        max_weight = max(all_values)
        weight_range = max(max_weight - min_weight, 0.1)
        
        # 绘制每个权重的曲线
        curves_drawn = 0
        points_drawn = 0
        for key, values in weight_history.items():
            if len(values) == 0:
                continue
            
            # 应用降采样：从完整数据中智能采样
            sampled_values = self._adaptive_downsample(list(values))
                        
            color = weight_colors.get(key, self.WHITE)
            points = []
            
            # 使用降采样后的数据绘制
            for i, value in enumerate(sampled_values):
                # 按照采样后的索引计算位置
                x = chart_origin_x + (i / max(len(sampled_values) - 1, 1)) * chart_width
                y = chart_origin_y - ((value - min_weight) / weight_range) * chart_height
                points.append((int(x), int(y)))
            
            try:
                if len(points) == 1:
                    # 只有1个数据点，绘制圆点
                    pygame.draw.circle(screen, color, points[0], 4)
                    points_drawn += 1
                elif len(points) > 1:
                    # 多个数据点，绘制曲线
                    pygame.draw.lines(screen, color, False, points, 2)
                    # 在每个数据点上也画一个小圆点
                    for point in points:
                        pygame.draw.circle(screen, color, point, 3)
                    curves_drawn += 1
            except Exception as e:
                print(f"[WeightHistoryPanel] 绘制失败: {e}")
        
        # Y轴标签
        label_max = self._small_font.render(f"{max_weight:.1f}", True, self.WHITE)
        screen.blit(label_max, (self.x + 5, chart_origin_y - chart_height - 5))
        
        label_min = self._small_font.render(f"{min_weight:.1f}", True, self.WHITE)
        screen.blit(label_min, (self.x + 5, chart_origin_y - 10))
        
        # X轴标签（显示实际数据量）
        actual_steps = max(len(v) for v in weight_history.values() if len(v) > 0)
        if actual_steps > self.max_display_points:
            xlabel = self._small_font.render(f"更新次数 ({actual_steps}步, 显示{self.max_display_points}点)", 
                                           True, self.LIGHT_GRAY)
        else:
            xlabel = self._small_font.render(f"更新次数 ({actual_steps}步)", True, self.LIGHT_GRAY)
        screen.blit(xlabel, (chart_origin_x + chart_width - 150, chart_origin_y + 5))
        
        # 图例(紧凑显示)
        legend_y = self.y + self.height - 20
        legend_x = self.x + 10
        
        weight_labels = {
            'repulsionCoefficient': 'α1',
            'entropyCoefficient': 'α2',
            'distanceCoefficient': 'α3',
            'leaderRangeCoefficient': 'α4',
            'directionRetentionCoefficient': 'α5'
        }
        
        for i, (key, label) in enumerate(weight_labels.items()):
            if key in weight_history and len(weight_history[key]) > 0:
                color = weight_colors.get(key, self.WHITE)
                # 颜色块
                pygame.draw.rect(screen, color, (legend_x + i * 65, legend_y, 12, 12))
                # 标签
                text = self._small_font.render(label, True, self.WHITE)
                screen.blit(text, (legend_x + i * 65 + 15, legend_y - 2))
