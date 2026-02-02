"""
分层强化学习训练可视化器
用于展示高层DQN的任务区域划分、无人机任务分配、决策历史等
"""
import sys
import os
import threading
import time
from typing import Dict, List, Optional, Tuple
from collections import deque
import pygame
import numpy as np

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
multirotor_dir = os.path.dirname(current_dir)
if multirotor_dir not in sys.path:
    sys.path.insert(0, multirotor_dir)

try:
    from Algorithm.Vector3 import Vector3
except ImportError:
    Vector3 = None
    print("警告: 无法导入Vector3，部分功能可能受限")


class HierarchicalVisualizer:
    """分层强化学习可视化器"""
    
    def __init__(self, env, server=None):
        """
        初始化可视化器
        
        Args:
            env: HierarchicalMovementEnv 或 MultiDroneHierarchicalMovementEnv
            server: AlgorithmServer实例（可选）
        """
        self.env = env
        self.server = server
        
        # 窗口设置
        self.SCREEN_WIDTH = 1400
        self.SCREEN_HEIGHT = 900
        self.right_panel_width = 350
        
        # 颜色定义
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.YELLOW = (255, 255, 0)
        self.CYAN = (0, 255, 255)
        self.MAGENTA = (255, 0, 255)
        self.ORANGE = (255, 165, 0)
        self.PURPLE = (128, 0, 128)
        self.GRAY = (128, 128, 128)
        self.LIGHT_GRAY = (200, 200, 200)
        self.LIGHT_BLUE = (173, 216, 230)
        self.DARK_GREEN = (0, 128, 0)
        
        # 坐标系转换参数（动态调整）
        self.view_width = self.SCREEN_WIDTH - self.right_panel_width  # 主视图宽度
        self.view_height = self.SCREEN_HEIGHT  # 主视图高度
        self.origin_x = self.view_width // 2
        self.origin_y = self.view_height // 2
        self.scale = 5  # 默认比例尺，会动态调整
        self.auto_scale = True  # 启用自动缩放
        self.scale_updated = False  # 标记是否已更新缩放
        
        # pygame初始化标志
        self.pygame_initialized = False
        self.font = None
        self.screen = None
        self.clock = None
        self.running = False
        self.visualization_thread = None
        
        # 数据缓存
        self.hl_action_history = deque(maxlen=100)  # 高层动作历史
        self.hl_goal_history = deque(maxlen=100)  # 高层目标历史
        self.reward_history = deque(maxlen=200)  # 奖励历史
        self.drone_colors = {}  # 无人机颜色映射
        
        # 统计信息
        self.episode_count = 0
        self.total_steps = 0
        self.current_episode_reward = 0
        self._entropy_stats = {}  # 熵值统计
        
    def _update_scale(self):
        """根据Leader扫描范围自动调整缩放比例"""
        if not self.auto_scale or self.scale_updated:
            return
        
        try:
            center, radius = self._get_leader_info()
            if center and radius > 0:
                # 计算需要显示的5x5区域总范围（扫描半径的两倍）
                display_range = radius * 2.2  # 留一些边距
                
                # 根据视图尺寸计算最佳缩放比例
                scale_x = (self.view_width * 0.85) / (display_range * 2)
                scale_y = (self.view_height * 0.85) / (display_range * 2)
                self.scale = min(scale_x, scale_y)
                
                self.scale_updated = True
                print(f"  ✓ 自动调整缩放: 扫描半径={radius:.1f}m, 缩放比例={self.scale:.2f}px/m")
        except Exception as e:
            print(f"缩放调整失败: {str(e)}")
    
    def world_to_screen(self, vector) -> Tuple[int, int]:
        """世界坐标转屏幕坐标"""
        if hasattr(vector, 'x'):
            screen_x = self.origin_x + vector.x * self.scale
            screen_y = self.origin_y - vector.z * self.scale
        else:
            screen_x = self.origin_x + vector[0] * self.scale
            screen_y = self.origin_y - vector[2] * self.scale
        return int(screen_x), int(screen_y)
    
    def get_drone_color(self, drone_name: str) -> Tuple[int, int, int]:
        """为每个无人机分配固定颜色"""
        if drone_name not in self.drone_colors:
            # 预定义颜色列表
            colors = [
                self.GREEN, self.CYAN, self.MAGENTA, 
                self.ORANGE, self.PURPLE, (255, 192, 203),  # Pink
                (0, 255, 127), (255, 215, 0)  # SpringGreen, Gold
            ]
            idx = len(self.drone_colors) % len(colors)
            self.drone_colors[drone_name] = colors[idx]
        return self.drone_colors[drone_name]
    
    def draw_grid_regions(self):
        """绘制5x5任务区域划分（高层DQN的动作空间）"""
        try:
            # 获取Leader位置和扫描范围
            center, radius = self._get_leader_info()
            if center is None:
                return
            
            # 绘制5x5网格
            grid_size = (2 * radius) / 5
            
            # 创建半透明表面（用于网格填充）
            surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            
            # 获取最近的高层动作（用于高亮）
            recent_actions = {}
            for step, action, drone in list(self.hl_action_history)[-5:]:  # 最近5个动作
                recent_actions[drone] = action
            
            for row in range(5):
                for col in range(5):
                    # 计算网格中心
                    offset_x = (col - 2) * grid_size
                    offset_z = (row - 2) * grid_size
                    grid_center = Vector3(
                        center.x + offset_x,
                        center.y,  # 维持在Leader的高度或特定高度
                        center.z + offset_z
                    )
                    
                    # 计算网格四角 (在 X-Z 平面上)
                    corners = [
                        Vector3(grid_center.x - grid_size/2, grid_center.y, grid_center.z - grid_size/2),
                        Vector3(grid_center.x + grid_size/2, grid_center.y, grid_center.z - grid_size/2),
                        Vector3(grid_center.x + grid_size/2, grid_center.y, grid_center.z + grid_size/2),
                        Vector3(grid_center.x - grid_size/2, grid_center.y, grid_center.z + grid_size/2),
                    ]
                    
                    # 转换为屏幕坐标
                    screen_corners = [self.world_to_screen(c) for c in corners]
                    
                    # 计算该区域的平均熵值（如果有grid_data）
                    entropy_color = self._get_region_entropy_color(grid_center, grid_size)
                    
                    # 绘制半透明矩形（填充）
                    pygame.draw.polygon(surface, (*entropy_color, 40), screen_corners)
                    
                    # 判断该区域是否被选中
                    action_id = row * 5 + col
                    is_selected = action_id in recent_actions.values()
                    
                    # 绘制网格边框（被选中的区域用红色高亮）
                    if is_selected:
                        pygame.draw.polygon(self.screen, self.RED, screen_corners, 5)  # 红色加粗
                        # 绘制闪烁效果
                        pygame.draw.polygon(surface, (255, 0, 0, 80), screen_corners)
                    else:
                        pygame.draw.polygon(self.screen, self.YELLOW, screen_corners, 2)  # 正常黄色
                    
                    # 绘制区域编号（更大、更明显）
                    center_screen = self.world_to_screen(grid_center)
                    
                    # 绘制编号背景（半透明圆圈）
                    bg_color = (255, 0, 0, 150) if is_selected else (0, 0, 0, 120)
                    pygame.draw.circle(surface, bg_color, center_screen, 20 if is_selected else 18)
                    
                    # 绘制编号文字
                    if self.font:
                        text_color = self.WHITE if is_selected else self.YELLOW
                        text = self.font.render(str(action_id), True, text_color)
                        text_rect = text.get_rect(center=center_screen)
                        self.screen.blit(text, text_rect)
                    
                    # 显示行列标签（只在边缘显示）
                    if col == 0 and self._small_font:  # 左侧显示行号
                        row_label = self._small_font.render(f"R{row}", True, self.CYAN)
                        self.screen.blit(row_label, (center_screen[0] - 40, center_screen[1] - 8))
                    if row == 0 and self._small_font:  # 顶部显示列号
                        col_label = self._small_font.render(f"C{col}", True, self.CYAN)
                        self.screen.blit(col_label, (center_screen[0] - 8, center_screen[1] - 40))
            
            # 应用半透明表面
            self.screen.blit(surface, (0, 0))
            
            # 绘制整体边界框（外边框）
            outer_corners = [
                Vector3(center.x - radius, center.y, center.z - radius),
                Vector3(center.x + radius, center.y, center.z - radius),
                Vector3(center.x + radius, center.y, center.z + radius),
                Vector3(center.x - radius, center.y, center.z + radius),
            ]
            outer_screen_corners = [self.world_to_screen(c) for c in outer_corners]
            pygame.draw.polygon(self.screen, self.WHITE, outer_screen_corners, 4)
            
        except Exception as e:
            print(f"绘制网格区域时出错: {str(e)}")
    
    def draw_current_hl_goals(self):
        """绘制每架无人机当前的高层目标"""
        try:
            # 判断是单机还是多机环境
            if hasattr(self.env, 'envs'):
                # 多机环境
                for drone_name, sub_env in self.env.envs.items():
                    if sub_env.current_hl_goal:
                        self._draw_drone_goal(drone_name, sub_env.current_hl_goal, sub_env)
            else:
                # 单机环境
                if self.env.current_hl_goal:
                    self._draw_drone_goal(self.env.drone_name, self.env.current_hl_goal, self.env)
        except Exception as e:
            print(f"绘制高层目标时出错: {str(e)}")
    
    def _draw_drone_goal(self, drone_name: str, goal: Vector3, sub_env):
        """绘制单个无人机的目标"""
        try:
            goal_screen = self.world_to_screen(goal)
            color = self.get_drone_color(drone_name)
            
            # 绘制目标标记（十字）
            size = 15
            pygame.draw.line(self.screen, color, 
                           (goal_screen[0] - size, goal_screen[1]), 
                           (goal_screen[0] + size, goal_screen[1]), 3)
            pygame.draw.line(self.screen, color, 
                           (goal_screen[0], goal_screen[1] - size), 
                           (goal_screen[0], goal_screen[1] + size), 3)
            
            # 绘制目标圆圈
            pygame.draw.circle(self.screen, color, goal_screen, 20, 2)
            
            # 绘制无人机到目标的连线
            if self.server:
                try:
                    with self.server.data_lock:
                        rd = self.server.unity_runtime_data.get(drone_name)
                        if rd and rd.position:
                            drone_screen = self.world_to_screen(rd.position)
                            pygame.draw.line(self.screen, color, drone_screen, goal_screen, 2)
                except:
                    pass
            
            # 显示目标标签
            if self._small_font:
                text = self._small_font.render(f"{drone_name} Goal", True, color)
                self.screen.blit(text, (goal_screen[0] + 25, goal_screen[1] - 10))
                
        except Exception as e:
            print(f"绘制无人机目标时出错: {str(e)}")
    
    def draw_drones(self):
        """绘制所有无人机"""
        try:
            if not self.server:
                return
            
            with self.server.data_lock:
                runtime_data_dict = self.server.unity_runtime_data
                
                for drone_name, rd in runtime_data_dict.items():
                    if not rd or not rd.position:
                        continue
                    
                    screen_pos = self.world_to_screen(rd.position)
                    color = self.get_drone_color(drone_name)
                    
                    # 绘制无人机主体
                    pygame.draw.circle(self.screen, color, screen_pos, 12)
                    pygame.draw.circle(self.screen, self.WHITE, screen_pos, 12, 2)
                    
                    # 绘制方向指示
                    if rd.finalMoveDir:
                        dir_end = (
                            screen_pos[0] + rd.finalMoveDir.x * 25,
                            screen_pos[1] - rd.finalMoveDir.z * 25
                        )
                        pygame.draw.line(self.screen, color, screen_pos, dir_end, 3)
                    
                    # 绘制无人机名称
                    if self._small_font:
                        text = self._small_font.render(drone_name, True, self.WHITE)
                        self.screen.blit(text, (screen_pos[0] + 15, screen_pos[1] - 15))
                        
        except Exception as e:
            print(f"绘制无人机时出错: {str(e)}")
    
    def draw_leader(self):
        """绘制Leader位置和扫描范围"""
        try:
            center, radius = self._get_leader_info()
            if center is None:
                return
            
            screen_pos = self.world_to_screen(center)
            
            # 绘制扫描范围（圆圈）
            radius_pixels = int(radius * self.scale)
            pygame.draw.circle(self.screen, self.LIGHT_BLUE, screen_pos, radius_pixels, 2)
            
            # 绘制Leader标记
            pygame.draw.circle(self.screen, self.LIGHT_BLUE, screen_pos, 20)
            pygame.draw.circle(self.screen, self.WHITE, screen_pos, 20, 3)
            
            # 绘制标签
            if self.font:
                text = self.font.render("Leader", True, self.WHITE)
                text_rect = text.get_rect(center=(screen_pos[0], screen_pos[1] - 35))
                self.screen.blit(text, text_rect)
                
                # 显示扫描半径
                if self._small_font:
                    radius_text = self._small_font.render(f"R={radius:.0f}m", True, self.LIGHT_BLUE)
                    self.screen.blit(radius_text, (screen_pos[0] + 25, screen_pos[1] - 15))
                
        except Exception as e:
            print(f"绘制Leader时出错: {str(e)}")
    
    def draw_entropy_heatmap(self):
        """绘制熵值热力图（显示所有熵值点）"""
        try:
            if not self.server or not hasattr(self.server, 'grid_data'):
                return
            
            with self.server.grid_lock:
                grid_data = self.server.grid_data
                if not grid_data or not hasattr(grid_data, 'cells'):
                    return
                
                # 统计信息
                total_cells = len(grid_data.cells)
                scanned_cells = 0
                high_entropy_cells = 0
                total_entropy = 0
                rendered_count = 0
                
                # 绘制所有熵值点
                for cell in grid_data.cells:
                    # 统计数据
                    total_entropy += cell.entropy
                    if cell.entropy < 30:  # 已扫描
                        scanned_cells += 1
                    if cell.entropy > 70:  # 高熵值
                        high_entropy_cells += 1
                    
                    screen_pos = self.world_to_screen(cell.center)
                    
                    # 检查是否在屏幕可见范围
                    if 0 <= screen_pos[0] <= self.view_width and \
                       0 <= screen_pos[1] <= self.view_height:
                        
                        # 熵值颜色映射：高熵值=红色，低熵值=绿色
                        entropy_normalized = max(0, min(1, cell.entropy / 100.0))
                        if entropy_normalized < 0.5:
                            red = int(510 * entropy_normalized)
                            green = 255
                        else:
                            red = 255
                            green = int(255 * (2 - 2 * entropy_normalized))
                        
                        color = (red, green, 0)
                        
                        # 绘制点，已扫描的点更小
                        radius_px = 2 if cell.entropy < 30 else 4
                        pygame.draw.circle(self.screen, color, screen_pos, radius_px)
                        rendered_count += 1
                
                # 存储统计信息供其他面板使用
                self._entropy_stats = {
                    'total': total_cells,
                    'scanned': scanned_cells,
                    'high_entropy': high_entropy_cells,
                    'avg_entropy': total_entropy / total_cells if total_cells > 0 else 0,
                    'rendered': rendered_count
                }
                        
        except Exception as e:
            print(f"绘制熵值热力图时出错: {str(e)}")
    
    def draw_hl_action_history_panel(self):
        """绘制高层决策历史面板"""
        try:
            if not hasattr(self, '_action_font'):
                self._action_font = pygame.font.SysFont(['Microsoft YaHei', 'Arial'], 12)
            
            panel_width = self.right_panel_width
            panel_height = 250
            panel_x = self.SCREEN_WIDTH - panel_width - 10
            panel_y = self._right_panel_next_y
            
            # 背景
            s = pygame.Surface((panel_width, panel_height))
            s.set_alpha(200)
            s.fill(self.BLACK)
            self.screen.blit(s, (panel_x, panel_y))
            pygame.draw.rect(self.screen, self.YELLOW, 
                           pygame.Rect(panel_x, panel_y, panel_width, panel_height), 2)
            
            # 标题
            title = self._action_font.render("High-Level Actions History", True, self.YELLOW)
            self.screen.blit(title, (panel_x + 10, panel_y + 10))
            
            # 显示最近的动作
            y = panel_y + 35
            for i, (step, action, drone) in enumerate(list(self.hl_action_history)[-10:]):
                region = f"R{action // 5}C{action % 5}"
                color = self.get_drone_color(drone) if drone else self.WHITE
                text = self._action_font.render(
                    f"[{step:4d}] {drone}: {action:2d} ({region})", 
                    True, color
                )
                self.screen.blit(text, (panel_x + 15, y))
                y += 20
            
            self._right_panel_next_y = panel_y + panel_height + 10
            
        except Exception as e:
            print(f"绘制动作历史面板时出错: {str(e)}")
    
    def draw_training_stats_panel(self):
        """绘制训练统计信息面板（包含详细熵值信息）"""
        try:
            if not hasattr(self, '_stats_font'):
                self._stats_font = pygame.font.SysFont(['Microsoft YaHei', 'Arial'], 13)
            
            panel_width = self.right_panel_width
            panel_height = 280  # 增加高度以显示更多信息
            panel_x = self.SCREEN_WIDTH - panel_width - 10
            panel_y = self._right_panel_next_y
            
            # 背景
            s = pygame.Surface((panel_width, panel_height))
            s.set_alpha(200)
            s.fill(self.BLACK)
            self.screen.blit(s, (panel_x, panel_y))
            pygame.draw.rect(self.screen, self.GREEN, 
                           pygame.Rect(panel_x, panel_y, panel_width, panel_height), 2)
            
            # 标题
            title = self._stats_font.render("Training & Entropy Statistics", True, self.GREEN)
            self.screen.blit(title, (panel_x + 10, panel_y + 10))
            
            y = panel_y + 35
            
            # 训练统计
            stats = [
                ("=== Training ===", self.CYAN),
                (f"Episode: {self.episode_count}", self.WHITE),
                (f"Total Steps: {self.total_steps}", self.WHITE),
                (f"Episode Reward: {self.current_episode_reward:.2f}", self.WHITE),
            ]
            
            # 添加环境统计
            if hasattr(self.env, 'step_count'):
                stats.append((f"Env Steps: {self.env.step_count}", self.WHITE))
            
            # 添加熵值统计
            if self._entropy_stats:
                stats.append(("=== Entropy Info ===", self.CYAN))
                stats.append((f"Total Cells: {self._entropy_stats.get('total', 0)}", self.WHITE))
                
                scanned = self._entropy_stats.get('scanned', 0)
                total = self._entropy_stats.get('total', 1)
                ratio = (scanned / total * 100) if total > 0 else 0
                stats.append((f"Scanned: {scanned}/{total} ({ratio:.1f}%)", self.GREEN if ratio > 50 else self.YELLOW))
                
                high_ent = self._entropy_stats.get('high_entropy', 0)
                stats.append((f"High Entropy (>70): {high_ent}", self.RED if high_ent > 10 else self.WHITE))
                
                avg_ent = self._entropy_stats.get('avg_entropy', 0)
                stats.append((f"Avg Entropy: {avg_ent:.1f}", self.WHITE))
                
                rendered = self._entropy_stats.get('rendered', 0)
                stats.append((f"Rendered Points: {rendered}", self.LIGHT_GRAY))
            
            # 显示所有统计
            for text_str, color in stats:
                if "===" in text_str:  # 分组标题
                    text = self._stats_font.render(text_str, True, color)
                    self.screen.blit(text, (panel_x + 10, y))
                    y += 22
                else:
                    text = self._stats_font.render(text_str, True, color)
                    self.screen.blit(text, (panel_x + 15, y))
                    y += 20
            
            self._right_panel_next_y = panel_y + panel_height + 10
            
        except Exception as e:
            print(f"绘制统计面板时出错: {str(e)}")
    
    def draw_reward_curve(self):
        """绘制奖励曲线"""
        try:
            if not hasattr(self, '_curve_font'):
                self._curve_font = pygame.font.SysFont(['Microsoft YaHei', 'Arial'], 12)
            
            panel_width = self.right_panel_width
            panel_height = 200
            panel_x = self.SCREEN_WIDTH - panel_width - 10
            panel_y = self._right_panel_next_y
            
            # 背景
            s = pygame.Surface((panel_width, panel_height))
            s.set_alpha(200)
            s.fill(self.BLACK)
            self.screen.blit(s, (panel_x, panel_y))
            pygame.draw.rect(self.screen, self.CYAN, 
                           pygame.Rect(panel_x, panel_y, panel_width, panel_height), 2)
            
            # 标题
            title = self._curve_font.render("Reward History", True, self.CYAN)
            self.screen.blit(title, (panel_x + 10, panel_y + 10))
            
            # 绘制坐标轴
            chart_margin_x = 40
            chart_margin_y = 35
            chart_width = panel_width - chart_margin_x - 20
            chart_height = panel_height - chart_margin_y - 30
            chart_origin_x = panel_x + chart_margin_x
            chart_origin_y = panel_y + panel_height - chart_margin_y
            
            pygame.draw.line(self.screen, self.LIGHT_GRAY, 
                           (chart_origin_x, chart_origin_y), 
                           (chart_origin_x + chart_width, chart_origin_y), 1)
            pygame.draw.line(self.screen, self.LIGHT_GRAY, 
                           (chart_origin_x, chart_origin_y), 
                           (chart_origin_x, chart_origin_y - chart_height), 1)
            
            # 绘制曲线
            if len(self.reward_history) > 1:
                rewards = list(self.reward_history)
                min_reward = min(rewards)
                max_reward = max(rewards)
                reward_range = max(max_reward - min_reward, 1.0)
                
                points = []
                for i, reward in enumerate(rewards):
                    x = chart_origin_x + (i / len(rewards)) * chart_width
                    y = chart_origin_y - ((reward - min_reward) / reward_range) * chart_height
                    points.append((x, y))
                
                if len(points) > 1:
                    pygame.draw.lines(self.screen, self.CYAN, False, points, 2)
                
                # 显示最新值
                latest_text = self._curve_font.render(
                    f"Latest: {rewards[-1]:.2f}", True, self.CYAN
                )
                self.screen.blit(latest_text, (panel_x + 15, panel_y + panel_height - 20))
            
            self._right_panel_next_y = panel_y + panel_height + 10
            
        except Exception as e:
            print(f"绘制奖励曲线时出错: {str(e)}")
    
    def draw_instructions(self):
        """绘制操作说明"""
        try:
            if not hasattr(self, '_inst_font'):
                self._inst_font = pygame.font.SysFont(['Microsoft YaHei', 'Arial'], 12)
            
            panel_width = self.right_panel_width
            panel_height = 100
            panel_x = self.SCREEN_WIDTH - panel_width - 10
            panel_y = self._right_panel_next_y
            
            # 背景
            s = pygame.Surface((panel_width, panel_height))
            s.set_alpha(180)
            s.fill(self.BLACK)
            self.screen.blit(s, (panel_x, panel_y))
            pygame.draw.rect(self.screen, self.WHITE, 
                           pygame.Rect(panel_x, panel_y, panel_width, panel_height), 1)
            
            # 说明文字
            instructions = [
                "5x5 Grid: High-level action space",
                "Colored markers: Drone HL goals",
                "ESC: Quit visualization"
            ]
            
            y = panel_y + 10
            for inst in instructions:
                text = self._inst_font.render(inst, True, self.LIGHT_GRAY)
                self.screen.blit(text, (panel_x + 10, y))
                y += 25
            
            self._right_panel_next_y = panel_y + panel_height + 10
            
        except Exception as e:
            print(f"绘制说明时出错: {str(e)}")
    
    def _get_leader_info(self) -> Tuple[Optional[Vector3], float]:
        """获取Leader位置和扫描半径"""
        try:
            if self.server:
                with self.server.data_lock:
                    runtime_data_dict = self.server.unity_runtime_data
                    if runtime_data_dict:
                        first_drone_data = next(iter(runtime_data_dict.values()))
                        if first_drone_data:
                            center = first_drone_data.leader_position
                            radius = first_drone_data.leader_scan_radius
                            if center and radius > 0:
                                return center, radius
            
            # 默认值
            return Vector3(0, 0, 8), 50.0
        except:
            return Vector3(0, 0, 8), 50.0
    
    def _get_region_entropy_color(self, center: Vector3, size: float) -> Tuple[int, int, int]:
        """计算区域的平均熵值颜色"""
        try:
            if not self.server or not hasattr(self.server, 'grid_data'):
                return (128, 128, 0)
            
            with self.server.grid_lock:
                grid = self.server.grid_data
                if not grid or not hasattr(grid, 'cells'):
                    return (128, 128, 0)
                
                # 计算该区域内的平均熵值
                nearby_cells = [
                    c for c in grid.cells 
                    if abs(c.center.x - center.x) < size/2 and 
                       abs(c.center.y - center.y) < size/2
                ]
                
                if not nearby_cells:
                    return (128, 128, 0)
                
                avg_entropy = sum(c.entropy for c in nearby_cells) / len(nearby_cells)
                entropy_normalized = max(0, min(1, avg_entropy / 100.0))
                
                if entropy_normalized < 0.5:
                    red = int(510 * entropy_normalized)
                    green = 255
                else:
                    red = 255
                    green = int(255 * (2 - 2 * entropy_normalized))
                
                return (red, green, 0)
        except:
            return (128, 128, 0)
    
    def update_training_data(self, step: int, action: int, reward: float, drone_name: str = "UAV1"):
        """更新训练数据（由训练脚本调用）"""
        self.total_steps = step
        self.current_episode_reward += reward
        self.hl_action_history.append((step, action, drone_name))
        self.reward_history.append(reward)
    
    def on_episode_end(self, episode: int):
        """Episode结束时调用"""
        self.episode_count = episode
        self.current_episode_reward = 0
    
    def handle_events(self):
        """处理事件"""
        try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
        except Exception as e:
            print(f"事件处理出错: {str(e)}")
    
    def _init_pygame_basic(self):
        """基础pygame初始化（不创建窗口）"""
        if self.pygame_initialized:
            return True
        
        try:
            print("  [1/3] 初始化pygame核心...")
            pygame.init()
            pygame.font.init()
            self.pygame_initialized = True
            print("  ✓ Pygame核心初始化完成")
            return True
        except Exception as e:
            print(f"❌ Pygame核心初始化失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_window(self):
        """在渲染线程中创建窗口"""
        try:
            print("  [2/3] 准备字体...")
            try:
                self.font = pygame.font.SysFont(['Microsoft YaHei', 'Arial'], 18)
                self._small_font = pygame.font.SysFont(['Microsoft YaHei', 'Arial'], 12)
            except:
                self.font = pygame.font.Font(None, 18)
                self._small_font = pygame.font.Font(None, 12)
            
            print("  [3/3] 创建显示窗口（在渲染线程中）...")
            self.clock = pygame.time.Clock()
            os.environ['SDL_VIDEO_WINDOW_POS'] = '100,100'
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            pygame.display.set_caption("🎯 Hierarchical DQN Training Visualization")
            
            print("=" * 60)
            print("✅ 分层训练可视化窗口已创建")
            print("💡 按ESC键关闭可视化窗口")
            print("=" * 60)
            return True
        except Exception as e:
            print(f"❌ 窗口创建失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def run(self):
        """主循环（在线程中运行）"""
        self.running = True
        
        # 检查pygame是否已初始化
        if not self.pygame_initialized:
            print("❌ 错误: pygame未初始化")
            return
        
        # 在渲染线程中创建窗口
        if not self._create_window():
            print("❌ 错误: 窗口创建失败")
            return
        
        while self.running:
            try:
                self.handle_events()
                
                # 自动调整缩放比例（只执行一次）
                if not self.scale_updated:
                    self._update_scale()
                
                self.screen.fill(self.BLACK)
                
                # 重置右侧面板布局
                self._right_panel_next_y = 10
                
                # 绘制主视图
                self.draw_entropy_heatmap()  # 先绘制背景
                self.draw_grid_regions()  # 绘制5x5网格
                self.draw_leader()  # 绘制Leader
                self.draw_current_hl_goals()  # 绘制高层目标
                self.draw_drones()  # 绘制无人机
                
                # 绘制右侧面板
                self.draw_training_stats_panel()
                self.draw_hl_action_history_panel()
                self.draw_reward_curve()
                self.draw_instructions()
                
                pygame.display.flip()
                
                if self.clock:
                    self.clock.tick(30)  # 30 FPS
            except Exception as e:
                print(f"可视化主循环出错: {str(e)}")
                time.sleep(0.05)
        
        try:
            pygame.quit()
        except Exception as e:
            print(f"退出pygame时出错: {str(e)}")
    
    def start_visualization(self):
        """在独立线程中启动可视化（窗口在子线程创建）"""
        print("  初始化pygame基础模块...")
        # 只初始化pygame核心，不创建窗口
        if not self._init_pygame_basic():
            print("❌ 可视化启动失败: pygame初始化失败")
            return False
        
        print("  启动可视化线程（窗口将在线程中创建）...")
        # 启动渲染线程，窗口将在线程内创建
        if not self.visualization_thread or not self.visualization_thread.is_alive():
            self.visualization_thread = threading.Thread(target=self.run)
            self.visualization_thread.daemon = True
            self.visualization_thread.start()
            print("  ✓ 可视化线程已启动，等待窗口创建...")
            # 等待窗口创建
            time.sleep(1.5)
            return True
        return False
    
    def stop_visualization(self):
        """停止可视化"""
        self.running = False
        if self.visualization_thread and self.visualization_thread.is_alive():
            self.visualization_thread.join(timeout=2.0)
