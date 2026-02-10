"""
分层DQN训练可视化器 - 用于分层强化学习训练

替代原有的HierarchicalVisualizer,使用新的可视化底座架构
支持:
- 5x5高层任务区域显示
- 高层决策历史
- 训练统计
- 奖励曲线
- 熵值统计
"""
import sys
import os
import time
from typing import Dict, Any, Tuple, Optional
from collections import deque
import pygame
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    from multirotor.Algorithm.Vector3 import Vector3
except ImportError:
    Vector3 = None
    print("警告: 无法导入Vector3，部分功能可能受限")

from multirotor.Visualization.base_visualizer import BaseVisualizer
from multirotor.Visualization.panels.environment_panel import EnvironmentPanel
from multirotor.Visualization.panels.training_stats_panel import TrainingStatsPanel
from multirotor.Visualization.panels.reward_curve_panel import RewardCurvePanel
from multirotor.Visualization.panels.hierarchical_grid_panel import HierarchicalGridPanel
from multirotor.Visualization.panels.battery_panel import BatteryPanel


class HierarchicalTrainingVisualizer(BaseVisualizer):
    """
    分层DQN训练可视化器
    
    用于分层强化学习训练,显示:
    - 环境状态(熵值热力图)
    - 5x5高层任务区域
    - 高层决策历史
    - 训练统计
    - 奖励曲线
    """
    
    def __init__(self, env, server=None):
        super().__init__(
            server=server,
            env=env,
            window_title="Hierarchical DQN Training Visualization"
        )
        
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
        # Episode 时间统计
        self.episode_start_time = None
        self.last_episode_duration = 0.0
        self.total_training_time = 0.0
    
    def setup_panels(self):
        """注册分层DQN专用面板"""
        # 环境状态面板(包含熵值统计)
        env_panel = EnvironmentPanel(width=350, height=200)
        self.panel_manager.register_panel(env_panel, position='auto')
        
        # 训练统计面板
        training_panel = TrainingStatsPanel(width=370, height=280)
        self.panel_manager.register_panel(training_panel, position='auto')
        
        # 5x5高层网格面板
        grid_panel = HierarchicalGridPanel(width=370, height=300)
        self.panel_manager.register_panel(grid_panel, position='auto')
        
        # 奖励曲线面板
        reward_panel = RewardCurvePanel(width=370, height=200)
        self.panel_manager.register_panel(reward_panel, position='auto')
        
        # 电量面板
        battery_panel = BatteryPanel(width=370, height=260)
        self.panel_manager.register_panel(battery_panel, position='auto')
    
    def get_visualization_data(self) -> Dict[str, Any]:
        """收集分层DQN训练可视化数据"""
        data = {}
        
        # 训练统计数据
        data['episode_count'] = self.episode_count
        data['total_steps'] = self.total_steps
        data['current_episode_steps'] = getattr(self.env, 'step_count', 0) if self.env else 0
        data['current_episode_reward'] = self.current_episode_reward

        # Episode 时间信息
        if self.episode_start_time is not None:
            data['current_episode_time'] = time.time() - self.episode_start_time
        elif self.last_episode_duration > 0:
            data['last_episode_duration'] = self.last_episode_duration
        if self.total_training_time > 0:
            data['total_training_time'] = self.total_training_time
        
        # 奖励历史
        data['reward_history'] = list(self.reward_history)
        
        # 统计数据
        if self.reward_history:
            data['avg_reward'] = sum(self.reward_history) / len(self.reward_history)
            data['max_reward'] = max(self.reward_history)
            data['min_reward'] = min(self.reward_history)
        
        # 高层动作历史
        data['hl_action_history'] = list(self.hl_action_history)
        
        # 熵值统计(从grid_data计算)
        if data.get('grid_data'):
            self._update_entropy_stats(data['grid_data'])
            data['entropy_stats'] = self._entropy_stats
        
        return data
    
    def _update_entropy_stats(self, grid_data):
        """更新熵值统计"""
        try:
            if not grid_data or not hasattr(grid_data, 'cells') or not grid_data.cells:
                return
            
            total_cells = len(grid_data.cells)
            scanned_cells = sum(1 for cell in grid_data.cells if cell.entropy < 30)
            high_entropy_cells = sum(1 for cell in grid_data.cells if cell.entropy > 70)
            total_entropy = sum(cell.entropy for cell in grid_data.cells)
            avg_entropy = total_entropy / total_cells if total_cells > 0 else 0
            
            self._entropy_stats = {
                'total': total_cells,
                'scanned': scanned_cells,
                'high_entropy': high_entropy_cells,
                'avg_entropy': avg_entropy,
                'scan_ratio': (scanned_cells / total_cells * 100) if total_cells > 0 else 0
            }
        except Exception:
            pass
    
    def update_training_data(self, step: int, action: int, reward: float, drone_name: str = "UAV1"):
        """
        更新训练数据(由训练脚本调用)
        
        Args:
            step: 当前步数
            action: 高层动作ID
            reward: 奖励值
            drone_name: 无人机名称
        """
        # 若当前没有记录episode开始时间，则认为是新的一轮episode开始
        if self.episode_start_time is None:
            import time as _time
            self.episode_start_time = _time.time()

        self.total_steps = step
        self.current_episode_reward += reward
        self.hl_action_history.append((step, action, drone_name))
        self.reward_history.append(reward)
    
    def on_episode_end(self, episode: int):
        """
        Episode结束时调用
        
        Args:
            episode: Episode编号
        """
        # 计算本轮episode耗时并累加到总训练时间
        if self.episode_start_time is not None:
            import time as _time
            self.last_episode_duration = _time.time() - self.episode_start_time
            self.total_training_time += self.last_episode_duration
            self.episode_start_time = None

        self.episode_count = episode
        self.current_episode_reward = 0
    
    def draw_main_view(self, screen: pygame.Surface, data: Dict[str, Any]):
        """
        绘制主视图区域（左侧）
        显示：
        - 熵值热力图（底层）
        - 5x5高层网格（跟随Leader）
        - 无人机目标指引箭头
        """
        try:
            grid_data = data.get('grid_data')
            runtime_data = data.get('runtime_data')
            
            # 第1层：绘制熵值热力图（底层背景）
            if grid_data:
                self.draw_grid(grid_data)
            
            # 第2层：绘制Leader标记
            if runtime_data:
                self._draw_leader(screen, data)
            
            # 第3层：绘制5x5高层网格（在无人机之前，但透明度低）
            if Vector3 and runtime_data:
                center, radius = self._get_leader_info(data)
                if center and radius > 0:
                    self._draw_hl_grid(screen, data)
                    # 调试标记：确认网格已绘制
                    if self.small_font:
                        debug_text = self.small_font.render("[5x5 Grid ON]", True, (0, 255, 0))
                        screen.blit(debug_text, (10, 10))
            
            # 第4层：绘制无人机和指引箭头（最上层，确保可见）
            if runtime_data:
                self._draw_drones_with_arrows(screen, data)
                # 调试标记：确认箭头已绘制
                if self.small_font:
                    debug_text = self.small_font.render("[Arrows ON]", True, (0, 255, 0))
                    screen.blit(debug_text, (10, 30))
                
        except Exception as e:
            print(f"绘制失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _get_leader_info(self, data: Dict[str, Any]) -> Tuple[Optional[Vector3], float]:
        """获取Leader位置和扫描半径（增强兼容性）"""
        try:
            runtime_data = data.get('runtime_data', {})
            if runtime_data:
                first_drone_data = next(iter(runtime_data.values()), None)
                if first_drone_data:
                    # 优先从字典中获取（base_visualizer.update_data 转换后的格式）
                    if isinstance(first_drone_data, dict):
                        center = first_drone_data.get('leaderPosition')
                        radius = first_drone_data.get('leaderScanRadius')
                    else:
                        # 尝试从对象属性获取
                        center = getattr(first_drone_data, 'leader_position', None)
                        radius = getattr(first_drone_data, 'leader_scan_radius', None)
                    
                    if center and radius and radius > 0:
                        return center, float(radius)
            
            # 兜底：尝试直接从环境对象获取（如果环境支持）
            if self.env:
                # 检查是否是多机环境
                if hasattr(self.env, 'envs') and self.env.envs:
                    first_sub_env = next(iter(self.env.envs.values()))
                    if hasattr(first_sub_env, 'leader_position'):
                        return first_sub_env.leader_position, getattr(first_sub_env, 'leader_scan_radius', 15.0)
                # 检查是否是单机环境
                elif hasattr(self.env, 'leader_position'):
                    return self.env.leader_position, getattr(self.env, 'leader_scan_radius', 15.0)

            return Vector3(0, 8, 0), 15.0
        except Exception as e:
            return Vector3(0, 8, 0), 15.0
    
    def _draw_leader(self, screen: pygame.Surface, data: Dict[str, Any]):
        """绘制Leader位置和扫描范围"""
        center, radius = self._get_leader_info(data)
        if not center:
            return
        
        screen_pos = self.world_to_screen(center)
        
        # 绘制扫描范围（圆圈）
        radius_pixels = int(radius * self.scale)
        pygame.draw.circle(screen, self.LIGHT_BLUE, screen_pos, radius_pixels, 2)
        
        # 绘制Leader标记
        pygame.draw.circle(screen, self.LIGHT_BLUE, screen_pos, 20)
        pygame.draw.circle(screen, self.WHITE, screen_pos, 20, 3)
        
        # 绘制标签
        if self.font:
            text = self.font.render("Leader", True, self.WHITE)
            text_rect = text.get_rect(center=(screen_pos[0], screen_pos[1] - 35))
            screen.blit(text, text_rect)
    
    def _draw_hl_grid(self, screen: pygame.Surface, data: Dict[str, Any]):
        """绘制5x5高层任务网格（在Leader扫描圆圈内）"""
        center, radius = self._get_leader_info(data)
        if not center or radius <= 0:
            return
            
        # 5x5网格填充在Leader扫描圆圈的 70% 范围内（留有边距）
        grid_coverage = 0.7
        effective_radius = radius * grid_coverage
        grid_size = (2 * effective_radius) / 5.0
            
        # 获取最近的高层动作
        hl_action_history = data.get('hl_action_history', [])
        recent_actions = {}
        for step, action, drone in list(hl_action_history)[-5:]:
            recent_actions[drone] = action
            
        for row in range(5):
            for col in range(5):
                # 计算网格中心（以Leader为中心，分成5x5）
                offset_x = (col - 2) * grid_size
                offset_z = (row - 2) * grid_size
                grid_center = Vector3(
                    center.x + offset_x,
                    center.y,
                    center.z + offset_z
                )
                    
                # 计算网格四角
                half_size = grid_size / 2
                corners = [
                    Vector3(grid_center.x - half_size, grid_center.y, grid_center.z - half_size),
                    Vector3(grid_center.x + half_size, grid_center.y, grid_center.z - half_size),
                    Vector3(grid_center.x + half_size, grid_center.y, grid_center.z + half_size),
                    Vector3(grid_center.x - half_size, grid_center.y, grid_center.z + half_size),
                ]
                    
                # 转换为屏幕坐标
                screen_corners = [self.world_to_screen(c) for c in corners]
                    
                # 判断是否被选中
                action_id = row * 5 + col
                is_selected = action_id in recent_actions.values()
                    
                # 绘制网格边框（1像素细线，直接绘制到screen）
                if is_selected:
                    pygame.draw.polygon(screen, self.RED, screen_corners, 2)
                else:
                    pygame.draw.polygon(screen, self.YELLOW, screen_corners, 1)
                    
                # 绘制区域编号（小字体，直接绘制到screen）
                center_screen = self.world_to_screen(grid_center)
                if self.small_font:
                    # 文字颜色
                    text_color = self.RED if is_selected else self.YELLOW
                    text = self.small_font.render(str(action_id), True, text_color)
                    text_rect = text.get_rect(center=center_screen)
                    screen.blit(text, text_rect)
    
    def _draw_drones_with_arrows(self, screen: pygame.Surface, data: Dict[str, Any]):
        """绘制无人机和目标指引箭头"""
        runtime_data = data.get('runtime_data', {})
        if not runtime_data:
            print("[DEBUG] runtime_data为空，无法绘制无人机")
            return
        
        # print(f"[DEBUG] 开始绘制无人机，数量: {len(runtime_data)}")
        
        for drone_name, rd in runtime_data.items():
            if not rd:
                print(f"[DEBUG] {drone_name} 的runtime_data为None")
                continue
            
            # 支持字典和对象两种格式
            position = rd.get('position') if isinstance(rd, dict) else getattr(rd, 'position', None)
            
            if not position:
                print(f"[DEBUG] {drone_name} 的position为None")
                continue
            
            screen_pos = self.world_to_screen(position)
            color = self._get_drone_color(drone_name)
            
            # print(f"[DEBUG] 绘制{drone_name} - 位置:{position}, 屏幕坐标:{screen_pos}")
            
            # 绘制无人机主体
            pygame.draw.circle(screen, color, screen_pos, 12)
            pygame.draw.circle(screen, self.WHITE, screen_pos, 12, 2)
            
            # 绘制目标指引箭头（返回目标网格信息）
            goal_info = self._draw_goal_arrow(screen, drone_name, screen_pos, color)
            
            # 绘制目标网格高亮边框（与无人机颜色一致）
            if goal_info and Vector3:
                try:
                    row, col, grid_center = goal_info
                    self._draw_goal_grid_highlight(screen, grid_center, drone_name, color, data)
                except:
                    pass
            
            # 绘制无人机名称
            if self.small_font:
                text = self.small_font.render(drone_name, True, self.WHITE)
                screen.blit(text, (screen_pos[0] + 15, screen_pos[1] - 15))
    
    def _draw_goal_arrow(self, screen: pygame.Surface, drone_name: str, 
                        drone_screen_pos: Tuple[int, int], color: Tuple[int, int, int]):
        """绘制目标指引箭头（终点吸附到最近的高层网格中心）"""
        try:
            # 检查是否有环境
            if not self.env:
                return
            
            # 获取该无人机的当前高层目标
            goal = None
            
            if hasattr(self.env, 'envs'):
                # 多机环境
                sub_env = self.env.envs.get(drone_name)
                if sub_env and hasattr(sub_env, 'current_hl_goal'):
                    goal = sub_env.current_hl_goal
            elif hasattr(self.env, 'current_hl_goal'):
                # 单机环境
                goal = self.env.current_hl_goal
            
            if not goal:
                return
            
            # 检查是否有服务器
            if not self.server:
                return
            
            # 获取无人机当前位置
            drone_pos = None
            try:
                with self.server.data_lock:
                    rd = self.server.unity_runtime_data.get(drone_name)
                    if rd and rd.position:
                        drone_pos = rd.position
            except:
                return
            
            if not drone_pos:
                return
            
            # 获取Leader信息与高层网格参数
            center, radius = self._get_leader_info({'runtime_data': self.server.unity_runtime_data})
            if not center or radius <= 0:
                return
            
            # 5x5网格参数（与_draw_hl_grid保持一致）
            grid_coverage = 0.7
            effective_radius = radius * grid_coverage
            grid_size = (2 * effective_radius) / 5.0
            
            # 将目标位置吸附到最近的高层网格中心
            # 计算目标在5x5网格中的行列索引
            dx = goal.x - center.x
            dz = goal.z - center.z
            
            # 计算网格索引（四舍五入到最近的格子）
            col = int(round((dx / grid_size) + 2))  # +2 因为索引从0开始，中心在(2,2)
            row = int(round((dz / grid_size) + 2))
            
            # 限制索引范围 [0,4]
            col = max(0, min(4, col))
            row = max(0, min(4, row))
            
            # 计算网格中心坐标
            offset_x = (col - 2) * grid_size
            offset_z = (row - 2) * grid_size
            grid_center = Vector3(
                center.x + offset_x,
                center.y,
                center.z + offset_z
            )
            
            # 使用网格中心作为箭头终点（而不是原始goal）
            goal_screen_pos = self.world_to_screen(grid_center)
            
            # print(f"[DEBUG] {drone_name} - 箭头: {drone_pos} -> 网格[{row},{col}]中心{grid_center}")
            
            # 计算距离
            dx = grid_center.x - drone_pos.x
            dy = grid_center.y - drone_pos.y
            dz = grid_center.z - drone_pos.z
            distance = (dx**2 + dy**2 + dz**2)**0.5
            
            if distance < 0.5:  # 已经到达目标
                return
            
            # 计算屏幕距离
            screen_dx = goal_screen_pos[0] - drone_screen_pos[0]
            screen_dy = goal_screen_pos[1] - drone_screen_pos[1]
            screen_distance = (screen_dx**2 + screen_dy**2)**0.5
            
            if screen_distance < 5:  # 屏幕距离太近，不绘制
                return
            
            # 绘制主箭头线（从无人机直接指向目标网格中心）
            pygame.draw.line(screen, color, drone_screen_pos, goal_screen_pos, 4)
            
            # 计算箭头方向（用于绘制箭头三角形）
            angle = np.arctan2(screen_dy, screen_dx)
            
            # 箭头大小（根据距离调整）
            arrow_size = min(15, max(8, screen_distance * 0.1))
            
            # 以“目标点”为尖端，向后退两个点组成三角形
            back_angle_left = angle + np.pi * 0.75
            back_angle_right = angle - np.pi * 0.75
            
            arrow_left = (
                int(goal_screen_pos[0] + arrow_size * np.cos(back_angle_left)),
                int(goal_screen_pos[1] + arrow_size * np.sin(back_angle_left))
            )
            arrow_right = (
                int(goal_screen_pos[0] + arrow_size * np.cos(back_angle_right)),
                int(goal_screen_pos[1] + arrow_size * np.sin(back_angle_right))
            )
            
            # 绘制箭头三角形（填充）
            pygame.draw.polygon(screen, color, [goal_screen_pos, arrow_left, arrow_right])
            
            # 绘制白色边框（提高可见度）
            pygame.draw.line(screen, self.WHITE, drone_screen_pos, goal_screen_pos, 2)
            pygame.draw.polygon(screen, self.WHITE, [goal_screen_pos, arrow_left, arrow_right], 2)
            
            # 显示距离（在箭头中点）
            if self.small_font and distance > 1:
                dist_text = self.small_font.render(f"{distance:.1f}m", True, self.YELLOW)
                mid_x = int((drone_screen_pos[0] + goal_screen_pos[0]) / 2)
                mid_y = int((drone_screen_pos[1] + goal_screen_pos[1]) / 2)
                
                text_rect = dist_text.get_rect(center=(mid_x, mid_y - 10))
                s = pygame.Surface((text_rect.width + 6, text_rect.height + 4))
                s.set_alpha(180)
                s.fill(self.BLACK)
                screen.blit(s, (text_rect.x - 3, text_rect.y - 2))
                screen.blit(dist_text, text_rect)
            
            # 返回目标网格信息，用于高亮绘制
            return row, col, grid_center
                
        except Exception:
            return None
    
    def _draw_goal_grid_highlight(
        self,
        screen: pygame.Surface,
        grid_center: Vector3,
        drone_name: str,
        color: Tuple[int, int, int],
        data: Dict[str, Any]
    ):
        """绘制无人机当前决策网格的高亮边框（与无人机颜色一致）"""
        try:
            center, radius = self._get_leader_info(data)
            if not center or radius <= 0:
                return

            grid_coverage = 0.7
            effective_radius = radius * grid_coverage
            grid_size = (2 * effective_radius) / 5.0
            half_size = grid_size / 2.0

            corners = [
                Vector3(grid_center.x - half_size, grid_center.y, grid_center.z - half_size),
                Vector3(grid_center.x + half_size, grid_center.y, grid_center.z - half_size),
                Vector3(grid_center.x + half_size, grid_center.y, grid_center.z + half_size),
                Vector3(grid_center.x - half_size, grid_center.y, grid_center.z + half_size),
            ]
            screen_corners = [self.world_to_screen(c) for c in corners]

            # 高亮边框（粗线）
            pygame.draw.polygon(screen, color, screen_corners, 4)
            # 再叠一层白边，增强对比度
            pygame.draw.polygon(screen, self.WHITE, screen_corners, 2)
        except Exception:
            return

    def _get_drone_color(self, drone_name: str) -> Tuple[int, int, int]:
        """为无人机分配颜色"""
        if drone_name not in self.drone_colors:
            colors = [self.GREEN, self.CYAN, self.MAGENTA, 
                     self.ORANGE, self.PURPLE, (255, 192, 203)]
            self.drone_colors[drone_name] = colors[len(self.drone_colors) % len(colors)]
        return self.drone_colors[drone_name]
    
    def run(self):
        """主渲染循环（覆盖以使用draw_main_view）"""
        self.running = True
        
        # 初始化pygame（使用基类逻辑）
        try:
            if not self.pygame_initialized:
                pygame.init()
                pygame.font.init()
                self.pygame_initialized = True
                
                try:
                    self.font = pygame.font.SysFont(['SimHei', 'Microsoft YaHei', 'Arial'], 18)
                    self.small_font = pygame.font.SysFont(['SimHei', 'Microsoft YaHei', 'Arial'], 14)
                except:
                    self.font = pygame.font.Font(None, 18)
                    self.small_font = pygame.font.Font(None, 14)
                
                self.clock = pygame.time.Clock()
                self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
                pygame.display.set_caption(self.window_title)
                
                # 初始化面板
                self.setup_panels()
                
                print("="  * 60)
                print(f"✅ {self.window_title} 已启动")
                print("💡 按ESC键关闭窗口")
                print("💡 主视图：5x5网格跟随Leader，箭头指向目标")
                print("=" * 60)
        except Exception as e:
            print(f"❌ Pygame初始化失败: {str(e)}")
            self.running = False
            return
        
        # 主循环
        while self.running:
            try:
                self.handle_events()
                self.screen.fill(self.BLACK)
                
                # 更新数据
                grid_data, runtime_data_dict = self.update_data()
                
                # 绘制边框
                self.draw_center_area_border()
                
                # *** 关键：调用draw_main_view而不是基类的draw方法 ***
                vis_data = self.get_visualization_data()
                vis_data['grid_data'] = grid_data
                vis_data['runtime_data'] = runtime_data_dict
                
                # 添加电量数据
                battery_data = self.get_battery_data()
                if battery_data:
                    vis_data['battery_data'] = battery_data
                
                # 使用自定义的draw_main_view
                self.draw_main_view(self.screen, vis_data)
                
                # 绘制熵值图例
                self.draw_entropy_legend()
                
                # 绘制面板
                self.panel_manager.draw_all_panels(self.screen, vis_data)
                
                pygame.display.flip()
                self.clock.tick(30)
            except Exception as e:
                print(f"渲染循环出错: {str(e)}")
                time.sleep(0.05)
        
        # 退出
        try:
            pygame.quit()
        except:
            pass


# 兼容性别名
HierarchicalVisualizer = HierarchicalTrainingVisualizer


if __name__ == "__main__":
    # 测试代码
    print("分层DQN训练可视化器 - 独立测试模式")
    print("提示: 此模式需要HierarchicalMovementEnv实例才能完整显示")
    
    visualizer = HierarchicalTrainingVisualizer(env=None, server=None)
    visualizer.start_visualization()
    
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n正在关闭...")
        visualizer.stop_visualization()
