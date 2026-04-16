"""
可视化底座基类 - BaseVisualizer

提供所有可视化器的公共功能:
1. 环境渲染(网格、无人机、Leader)
2. 面板管理系统
3. 线程管理和事件处理
4. 数据更新接口
"""
import sys
import os
import threading
import time
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import pygame

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from multirotor.Algorithm.Vector3 import Vector3
from multirotor.Visualization.panel_system import PanelManager


class BaseVisualizer(ABC):
    """
    可视化器基类
    
    核心职责:
    1. 管理pygame窗口和渲染循环
    2. 提供公共绘制方法(网格、无人机、Leader等)
    3. 管理面板系统
    4. 处理线程安全的数据更新
    """
    
    def __init__(self, server=None, env=None, window_title: str = "Training Visualization"):
        """
        初始化基础可视化器
        
        Args:
            server: AlgorithmServer实例
            env: 训练环境实例(可选)
            window_title: 窗口标题
        """
        self.server = server
        self.env = env
        self.window_title = window_title

        self._apply_responsive_layout()
        self.render_fps = 30
        
        # 颜色定义
        self._init_colors()
        
        # Pygame初始化标志
        self.pygame_initialized = False
        self.font = None
        self.screen = None
        self.clock = None
        self.running = False
        self.visualization_thread = None
        
        # 面板管理器
        self.panel_manager = PanelManager(
            self.SCREEN_WIDTH, 
            self.SCREEN_HEIGHT, 
            self.left_panel_width,
            self.right_panel_width
        )
        
        # 数据缓存(减少锁竞争)
        self._cached_grid_data = None
        self._cached_runtime_data = {}
        self._cached_obstacles = []  # 障碍物数据缓存
        self._last_data_update = 0
        self._data_update_interval = 0.05  # 50ms更新一次

    def _apply_responsive_layout(self):
        """根据桌面可用空间自适应调整窗口布局。"""
        base_width = 1920
        base_height = 1080
        base_side_panel = 360
        min_center_width = 420

        desktop_size = self._get_desktop_size()
        if desktop_size is None:
            target_width = base_width
            target_height = base_height
        else:
            desktop_width, desktop_height = desktop_size
            target_width = min(base_width, max(960, int(desktop_width * 1.0)))
            target_height = min(base_height, max(540, int(desktop_height * 0.98)))

        scale = min(target_width / base_width, target_height / base_height)
        scale = max(scale, 0.5)

        self.SCREEN_WIDTH = int(base_width * scale)
        self.SCREEN_HEIGHT = int(base_height * scale)

        scaled_panel_width = max(220, int(base_side_panel * scale))
        max_panel_width = max(220, (self.SCREEN_WIDTH - min_center_width) // 2)
        panel_width = min(scaled_panel_width, max_panel_width)
        self.left_panel_width = panel_width
        self.right_panel_width = panel_width

        # 坐标系参数（中间热力图区域）
        self.view_width = (
            self.SCREEN_WIDTH - self.left_panel_width - self.right_panel_width
        )
        self.view_height = self.SCREEN_HEIGHT
        self.view_offset_x = self.left_panel_width
        self.origin_x = self.view_offset_x + self.view_width // 2
        self.origin_y = self.view_height // 2
        self.scale = max(10, int(20 * scale))

    def _get_desktop_size(self) -> Optional[Tuple[int, int]]:
        """获取桌面工作区大小，便于在不同分辨率和缩放环境中自适应。"""
        override = os.environ.get("VIS_DESKTOP_SIZE", "").strip().lower()
        if override:
            normalized = override.replace("*", "x")
            parts = normalized.split("x")
            if len(parts) == 2:
                try:
                    return int(parts[0]), int(parts[1])
                except ValueError:
                    pass

        if os.name != "nt":
            return None

        try:
            import ctypes

            class RECT(ctypes.Structure):
                _fields_ = [
                    ("left", ctypes.c_long),
                    ("top", ctypes.c_long),
                    ("right", ctypes.c_long),
                    ("bottom", ctypes.c_long),
                ]

            rect = RECT()
            spi_get_work_area = 0x0030
            if ctypes.windll.user32.SystemParametersInfoW(
                spi_get_work_area, 0, ctypes.byref(rect), 0
            ):
                width = int(rect.right - rect.left)
                height = int(rect.bottom - rect.top)
                if width > 0 and height > 0:
                    return width, height
        except Exception:
            return None

        return None

    def configure_side_panel_layout(
        self,
        left_panel_width: int,
        right_panel_width: int,
        *,
        min_center_width: int = 480,
    ) -> None:
        """允许子类在已初始化后覆盖左右侧栏宽度，形成更适合各场景的布局。"""
        requested_left = max(220, int(left_panel_width))
        requested_right = max(220, int(right_panel_width))
        available_for_panels = max(440, self.SCREEN_WIDTH - max(min_center_width, 320))
        total_requested = requested_left + requested_right

        if total_requested > available_for_panels:
            scale = available_for_panels / max(total_requested, 1)
            requested_left = max(220, int(requested_left * scale))
            requested_right = max(220, int(requested_right * scale))
            overflow = requested_left + requested_right - available_for_panels
            while overflow > 0 and (requested_left > 220 or requested_right > 220):
                if requested_right >= requested_left and requested_right > 220:
                    requested_right -= 1
                elif requested_left > 220:
                    requested_left -= 1
                overflow -= 1

        self.left_panel_width = requested_left
        self.right_panel_width = requested_right
        self.view_width = self.SCREEN_WIDTH - self.left_panel_width - self.right_panel_width
        self.view_height = self.SCREEN_HEIGHT
        self.view_offset_x = self.left_panel_width
        self.origin_x = self.view_offset_x + self.view_width // 2
        self.origin_y = self.view_height // 2
        self.panel_manager.left_panel_width = self.left_panel_width
        self.panel_manager.right_panel_width = self.right_panel_width

    def _scale_panel_heights(
        self,
        requested_heights: List[int],
        min_heights: Optional[List[int]] = None,
        row_gap: int = 10,
        outer_margin: int = 10,
    ) -> List[int]:
        """将固定面板高度按当前窗口高度压缩到可见范围内。"""
        if not requested_heights:
            return []

        if min_heights is None:
            min_heights = [110] * len(requested_heights)

        usable_height = max(
            0,
            self.SCREEN_HEIGHT - outer_margin * 2 - row_gap * (len(requested_heights) - 1),
        )
        total_requested = sum(requested_heights)
        if total_requested <= usable_height:
            return list(requested_heights)

        scale = usable_height / max(total_requested, 1)
        scaled = [
            max(min_heights[i], int(round(height * scale)))
            for i, height in enumerate(requested_heights)
        ]

        hard_min = 80
        overflow = sum(scaled) - usable_height
        while overflow > 0:
            reducible = [
                idx for idx, height in enumerate(scaled) if height > hard_min
            ]
            if not reducible:
                break
            scaled[reducible[0 if len(reducible) == 1 else max(
                range(len(reducible)),
                key=lambda pos: scaled[reducible[pos]]
            )]] -= 1
            overflow -= 1

        return scaled
    
    def _init_colors(self):
        """??????????"""
        self.SCREEN_BACKGROUND = (220, 225, 232)
        self.PANEL_BACKGROUND = (232, 236, 241)
        self.TEXT_PRIMARY = (44, 52, 64)
        self.TEXT_SECONDARY = (95, 106, 123)
        self.BLACK = self.SCREEN_BACKGROUND
        self.WHITE = self.TEXT_PRIMARY
        self.RED = (178, 108, 101)
        self.GREEN = (87, 137, 114)
        self.BLUE = (102, 145, 180)
        self.YELLOW = (176, 140, 82)
        self.CYAN = (86, 133, 166)
        self.MAGENTA = (165, 111, 126)
        self.ORANGE = (189, 133, 94)
        self.PURPLE = (126, 112, 170)
        self.GRAY = (205, 213, 223)
        self.LIGHT_GRAY = (95, 106, 123)
        self.DARK_GRAY = (137, 147, 162)
        self.LIGHT_BLUE = (72, 104, 146)
        self.DRONE_GREEN = (88, 145, 136)
        self.SCAN_RANGE_COLOR = (58, 120, 128)
    
    def world_to_screen(self, vector) -> Tuple[int, int]:
        """
        世界坐标转屏幕坐标
        
        Args:
            vector: Vector3对象或(x, y, z)元组
            
        Returns:
            (screen_x, screen_y)屏幕坐标
        """
        if hasattr(vector, 'x'):
            screen_x = self.origin_x + vector.x * self.scale
            screen_y = self.origin_y - vector.z * self.scale
        else:
            screen_x = self.origin_x + vector[0] * self.scale
            screen_y = self.origin_y - vector[2] * self.scale
        return int(screen_x), int(screen_y)
    
    def draw_grid(self, grid_data):
        """
        绘制网格熵值热力图
        
        Args:
            grid_data: HexGridDataModel实例
        """
        # 如果没有网格数据或网格被清空，直接返回（Pygame主循环每帧会fill BLACK，所以不需要额外操作）
        if not grid_data or not hasattr(grid_data, 'cells') or len(grid_data.cells) == 0:
            return
        
        # 缓存小字体
        if not hasattr(self, '_small_font'):
            try:
                self._small_font = pygame.font.SysFont(['SimHei', 'Microsoft YaHei', 'Arial'], 10)
            except:
                self._small_font = None
        
        for cell in grid_data.cells:
            screen_x, screen_y = self.world_to_screen(cell.center)
            
            # 熵值颜色映射: 0(绿) -> 100(红)
            entropy_value = cell.entropy
            entropy_normalized = max(0, min(1, entropy_value / 100.0))
            
            if entropy_normalized < 0.5:
                ratio = entropy_normalized / 0.5
                low = (72, 116, 176)
                mid = (178, 132, 74)
                color = tuple(int(low[i] + (mid[i] - low[i]) * ratio) for i in range(3))
            else:
                ratio = (entropy_normalized - 0.5) / 0.5
                mid = (178, 132, 74)
                high = (154, 78, 78)
                color = tuple(int(mid[i] + (high[i] - mid[i]) * ratio) for i in range(3))
            
            # 只绘制可见区域
            if 0 <= screen_x <= self.SCREEN_WIDTH and 0 <= screen_y <= self.SCREEN_HEIGHT:
                radius = 2 if cell.entropy < 45 else (2 if cell.entropy < 80 else 3)
                pygame.draw.circle(self.screen, color, (screen_x, screen_y), radius)
    
    def draw_drones(self, runtime_data_dict: Dict):
        """
        绘制所有无人机
        
        Args:
            runtime_data_dict: {drone_name: runtime_data}
        """
        if not runtime_data_dict:
            return
        
        for drone_name, drone_info in runtime_data_dict.items():
            if not drone_info or 'position' not in drone_info or not drone_info['position']:
                continue
            
            screen_x, screen_y = self.world_to_screen(drone_info['position'])
            
            # 绘制扫描范围
            scan_radius_meters = 1.0
            if self.server and hasattr(self.server, 'config_data'):
                scan_radius_meters = self.server.config_data.scanRadius
            
            scan_radius_pixels = scan_radius_meters * self.scale
            scan_surface = pygame.Surface((int(scan_radius_pixels * 2) + 6, int(scan_radius_pixels * 2) + 6), pygame.SRCALPHA)
            pygame.draw.circle(scan_surface, (*self.SCAN_RANGE_COLOR, 156), (int(scan_radius_pixels) + 3, int(scan_radius_pixels) + 3), int(scan_radius_pixels), 4)
            pygame.draw.circle(scan_surface, (46, 86, 98, 72), (int(scan_radius_pixels) + 3, int(scan_radius_pixels) + 3), int(scan_radius_pixels) + 1, 1)
            pygame.draw.circle(scan_surface, (*self.WHITE, 52), (int(scan_radius_pixels) + 3, int(scan_radius_pixels) + 3), int(scan_radius_pixels), 1)
            self.screen.blit(scan_surface, (screen_x - int(scan_radius_pixels) - 3, screen_y - int(scan_radius_pixels) - 3))
            
            # 绘制无人机主体
            pygame.draw.circle(self.screen, self.DRONE_GREEN, (screen_x, screen_y), 10)
            pygame.draw.circle(self.screen, self.WHITE, (screen_x, screen_y), 10, 2)
            
            # 绘制方向指示
            if 'finalMoveDir' in drone_info and drone_info['finalMoveDir']:
                dir_x = screen_x + drone_info['finalMoveDir'].x * 20
                dir_y = screen_y - drone_info['finalMoveDir'].z * 20
                pygame.draw.line(self.screen, self.WHITE, (screen_x, screen_y), (dir_x, dir_y), 3)
            
            # 绘制无人机名称
            if not hasattr(self, '_drone_name_cache'):
                self._drone_name_cache = {}
            
            if drone_name not in self._drone_name_cache:
                self._drone_name_cache[drone_name] = self.font.render(drone_name, True, self.WHITE)
            
            self.screen.blit(self._drone_name_cache[drone_name], (screen_x + 15, screen_y - 10))
    
    def draw_leader(self, runtime_data_dict: Dict):
        """
        绘制Leader位置和扫描范围
        
        Args:
            runtime_data_dict: {drone_name: runtime_data}
        """
        if not runtime_data_dict:
            return
        
        try:
            first_drone_data = next(iter(runtime_data_dict.values()))
            if not first_drone_data or 'leaderPosition' not in first_drone_data:
                return
            
            leader_pos = first_drone_data['leaderPosition']
            if not leader_pos:
                return
            
            screen_x, screen_y = self.world_to_screen(leader_pos)
            
            # 绘制Leader标记
            pygame.draw.circle(self.screen, self.LIGHT_BLUE, (screen_x, screen_y), 20)
            pygame.draw.circle(self.screen, self.WHITE, (screen_x, screen_y), 20, 3)
            
            # 绘制扫描范围
            if 'leaderScanRadius' in first_drone_data and first_drone_data['leaderScanRadius'] > 0:
                radius = first_drone_data['leaderScanRadius'] * self.scale
                leader_surface = pygame.Surface((int(radius * 2) + 6, int(radius * 2) + 6), pygame.SRCALPHA)
                pygame.draw.circle(leader_surface, (52, 76, 112, 92), (int(radius) + 3, int(radius) + 3), int(radius) + 1, 1)
                pygame.draw.circle(leader_surface, (*self.LIGHT_BLUE, 182), (int(radius) + 3, int(radius) + 3), int(radius), 6)
                pygame.draw.circle(leader_surface, (*self.WHITE, 86), (int(radius) + 3, int(radius) + 3), int(radius), 1)
                self.screen.blit(leader_surface, (screen_x - int(radius) - 3, screen_y - int(radius) - 3))
            
            # 绘制标签
            if self.font:
                text = self.font.render("Leader", True, self.WHITE)
                text_rect = text.get_rect(center=(screen_x, screen_y - 35))
                self.screen.blit(text, text_rect)
        except Exception as e:
            pass
    
    def draw_entropy_legend(self):
        """绘制熵值图例(右上角)"""
        try:
            if not self.font:
                return
            
            legend_x = self.SCREEN_WIDTH - 130
            legend_y = 10
            legend_width = 120
            legend_height = 70
            
            # 半透明背景
            background_rect = pygame.Rect(legend_x, legend_y, legend_width, legend_height)
            s = pygame.Surface((legend_width, legend_height))
            s.set_alpha(180)
            s.fill(self.PANEL_BACKGROUND)
            self.screen.blit(s, (legend_x, legend_y))
            pygame.draw.rect(self.screen, self.DARK_GRAY, background_rect, 1)
            
            # 标题
            if not hasattr(self, '_legend_font'):
                try:
                    self._legend_font = pygame.font.SysFont(['SimHei', 'Microsoft YaHei', 'Arial'], 14)
                except:
                    self._legend_font = pygame.font.Font(None, 14)
            
            title = self._legend_font.render("Entropy", True, self.DARK_GRAY)
            self.screen.blit(title, (legend_x + 5, legend_y + 5))
            
            # 渐变颜色条
            bar_x = legend_x + 5
            bar_y = legend_y + 25
            bar_width = legend_width - 10
            bar_height = 15
            
            for i in range(bar_width):
                entropy_normalized = i / bar_width
                if entropy_normalized < 0.5:
                    ratio = entropy_normalized / 0.5
                    low = (72, 116, 176)
                    mid = (178, 132, 74)
                    color = tuple(int(low[j] + (mid[j] - low[j]) * ratio) for j in range(3))
                else:
                    ratio = (entropy_normalized - 0.5) / 0.5
                    mid = (178, 132, 74)
                    high = (154, 78, 78)
                    color = tuple(int(mid[j] + (high[j] - mid[j]) * ratio) for j in range(3))
                pygame.draw.line(self.screen, color, (bar_x + i, bar_y), (bar_x + i, bar_y + bar_height))
            
            # 刻度标签
            label_0 = self._legend_font.render("0", True, self.DARK_GRAY)
            self.screen.blit(label_0, (bar_x, bar_y + bar_height + 2))
            
            label_100 = self._legend_font.render("100", True, self.DARK_GRAY)
            label_100_rect = label_100.get_rect(right=bar_x + bar_width)
            self.screen.blit(label_100, (label_100_rect.x, bar_y + bar_height + 2))
        except Exception as e:
            pass
    
    def draw_center_area_border(self):
        """绘制中间热力图区域边框"""
        try:
            border_rect = pygame.Rect(
                self.view_offset_x, 
                0, 
                self.view_width, 
                self.view_height
            )
            # 绘制边框
            pygame.draw.rect(self.screen, self.DARK_GRAY, border_rect, 2)
        except Exception as e:
            pass

    def draw_obstacles(self, obstacles: list):
        """
        绘制障碍物（支持多边形和圆形）

        Args:
            obstacles: 障碍物列表，每个障碍物包含:
                - shapeType: Unity枚举 0=Point, 1=Sphere, 2=Polygon, 3=Circle, 4=Box
                - category: 0=Normal, 1=RestrictedZone
                - vertices: 多边形顶点列表（Polygon/Box时使用）
                - center: 圆心坐标（Circle/Sphere时使用）
                - radius: 半径（Circle/Sphere时使用）
        """
        if not obstacles:
            return

        for obstacle in obstacles:
            try:
                # 获取形状类型（Unity枚举）
                raw_shape_type = obstacle.get('shapeType', 0)
                if isinstance(raw_shape_type, int):
                    shape_type = raw_shape_type
                else:
                    shape_type_str = str(raw_shape_type).lower()
                    # Unity枚举: Point=0, Sphere=1, Polygon=2, Circle=3, Box=4
                    shape_type_map = {'point': 0, 'sphere': 1, 'polygon': 2, 'circle': 3, 'box': 4}
                    shape_type = shape_type_map.get(shape_type_str, 0)

                # 获取类别（普通障碍物或禁飞区）
                category = obstacle.get('category', 0)
                if isinstance(category, str):
                    is_restricted = category.lower() in ['restrictedzone', 'restricted']
                else:
                    # Unity枚举: Normal=0, RestrictedZone=1
                    is_restricted = category == 1

                # 根据类别选择颜色
                if is_restricted:
                    # 禁飞区：红色半透明
                    color = (190, 140, 132, 72)
                    border_color = self.RED
                else:
                    # 普通障碍物：橙色半透明
                    color = (202, 174, 138, 64)
                    border_color = self.ORANGE

                # 处理多边形类型（Unity: Polygon=2, Box=4）
                if shape_type in [2, 4]:
                    # 绘制多边形障碍物
                    vertices_data = obstacle.get('vertices', [])
                    if vertices_data:
                        # 转换顶点坐标
                        screen_points = []
                        for v in vertices_data:
                            pos = Vector3(v.get('x', 0), v.get('y', 0), v.get('z', 0))
                            screen_points.append(self.world_to_screen(pos))

                        # 绘制填充多边形（需要创建带alpha的surface）
                        if len(screen_points) >= 3:
                            min_x = min(p[0] for p in screen_points)
                            max_x = max(p[0] for p in screen_points)
                            min_y = min(p[1] for p in screen_points)
                            max_y = max(p[1] for p in screen_points)

                            # 创建临时surface绘制半透明填充
                            temp_surface = pygame.Surface((max_x - min_x + 4, max_y - min_y + 4), pygame.SRCALPHA)
                            offset_points = [(p[0] - min_x + 2, p[1] - min_y + 2) for p in screen_points]
                            pygame.draw.polygon(temp_surface, color, offset_points)
                            self.screen.blit(temp_surface, (min_x - 2, min_y - 2))

                        # 绘制边框
                        pygame.draw.polygon(self.screen, border_color, screen_points, 3)

                # 处理圆形类型（Unity: Circle=3, Sphere=1）
                elif shape_type in [1, 3]:
                    # 绘制圆形障碍物
                    center_data = obstacle.get('center', {})
                    if center_data:
                        center = Vector3(
                            center_data.get('x', 0),
                            center_data.get('y', 0),
                            center_data.get('z', 0)
                        )
                        screen_x, screen_y = self.world_to_screen(center)
                        radius = obstacle.get('radius', 5.0)
                        radius_pixels = int(radius * self.scale)

                        # 绘制填充圆（半透明）
                        temp_surface = pygame.Surface((radius_pixels * 2 + 4, radius_pixels * 2 + 4), pygame.SRCALPHA)
                        pygame.draw.circle(temp_surface, color, (radius_pixels + 2, radius_pixels + 2), radius_pixels)
                        self.screen.blit(temp_surface, (screen_x - radius_pixels - 2, screen_y - radius_pixels - 2))

                        # 绘制边框
                        pygame.draw.circle(self.screen, border_color, (screen_x, screen_y), radius_pixels, 3)

            except Exception as e:
                # 输出错误以便调试
                import traceback
                print(f"[BaseVisualizer] 绘制障碍物出错: {e}")
                traceback.print_exc()

    def update_data(self) -> Tuple[Optional[object], Dict]:
        """
        更新可视化数据(线程安全,带缓存)
        
        Returns:
            (grid_data, runtime_data_dict)
        """
        if not self.server:
            return None, {}
        
        # 如果服务端显式要求刷新（例如reset_environment后），跳过缓存并清空
        try:
            if self.server and getattr(self.server, '_vis_snapshot_cache', None) is None:
                self._cached_grid_data = None
                self._cached_runtime_data = {}
                self._cached_obstacles = []
                self._last_data_update = 0
        except Exception:
            pass

        current_time = time.time()
        if current_time - self._last_data_update < self._data_update_interval:
            return self._cached_grid_data, self._cached_runtime_data
        
        # 获取网格数据
        grid_data = None
        try:
            if hasattr(self.server, 'grid_data'):
                grid_data = self.server.grid_data
        except:
            pass
        
        # 获取运行时数据
        runtime_data_dict = {}
        try:
            if hasattr(self.server, 'unity_runtime_data'):
                unity_data = self.server.unity_runtime_data
                for drone_name, runtime_data in unity_data.items():
                    if runtime_data:
                        drone_info = {
                            'position': runtime_data.position,
                            'finalMoveDir': runtime_data.finalMoveDir,
                            'leaderPosition': runtime_data.leader_position,
                            'leaderScanRadius': runtime_data.leader_scan_radius
                        }
                        runtime_data_dict[drone_name] = drone_info
        except:
            pass

        # 获取障碍物数据
        obstacles = []
        try:
            if hasattr(self.server, 'obstacles'):
                obstacles = self.server.obstacles if self.server.obstacles else []
        except:
            pass

        # 缓存数据
        self._cached_grid_data = grid_data
        self._cached_runtime_data = runtime_data_dict
        self._cached_obstacles = obstacles
        self._last_data_update = current_time

        return grid_data, runtime_data_dict
    
    @abstractmethod
    def setup_panels(self):
        """
        设置面板(子类必须实现)
        
        在此方法中注册所需的面板
        """
        pass
    
    @abstractmethod
    def get_visualization_data(self) -> Dict:
        """
        获取可视化数据(子类必须实现)
        
        Returns:
            数据字典,传递给所有面板
        """
        pass
    
    def get_battery_data(self) -> Dict:
        """
        获取电量数据(公共方法)
        
        Returns:
            {drone_name: battery_info_dict}
        """
        try:
            if self.server and hasattr(self.server, 'get_all_battery_data'):
                return self.server.get_all_battery_data()
        except Exception as e:
            pass
        return {}

    def get_entropy_visualization_data(self) -> Dict:
        """获取熵值概览/趋势面板所需的数据。"""
        data: Dict = {}
        if not self.server:
            return data

        try:
            if hasattr(self.server, "get_entropy_history"):
                data["entropy_history"] = self.server.get_entropy_history(limit=300)
            elif hasattr(self.server, "entropy_history"):
                data["entropy_history"] = list(
                    getattr(self.server, "entropy_history", [])[-300:]
                )
        except Exception:
            pass

        try:
            if hasattr(self.server, "get_scan_progress_history"):
                data["scan_progress_history"] = self.server.get_scan_progress_history(limit=300)
            elif hasattr(self.server, "scan_progress_history"):
                data["scan_progress_history"] = list(
                    getattr(self.server, "scan_progress_history", [])[-300:]
                )
        except Exception:
            pass

        try:
            if hasattr(self.server, "get_entropy_distribution"):
                data["entropy_distribution"] = self.server.get_entropy_distribution(limit=1)
            elif hasattr(self.server, "entropy_distribution"):
                data["entropy_distribution"] = list(
                    getattr(self.server, "entropy_distribution", [])[-1:]
                )
        except Exception:
            pass

        try:
            bins = getattr(self.server, "entropy_bins", None)
            if bins:
                data["entropy_bins"] = list(bins)
        except Exception:
            pass

        return data

    def get_obstacles_data(self) -> list:
        """
        获取障碍物数据(公共方法)

        Returns:
            障碍物列表
        """
        return self._cached_obstacles if self._cached_obstacles else []

    def handle_events(self):
        """处理pygame事件"""
        try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
        except Exception as e:
            print(f"事件处理出错: {str(e)}")
    
    def run(self):
        """主渲染循环"""
        self.running = True
        
        # 初始化pygame
        try:
            if not self.pygame_initialized:
                pygame.init()
                pygame.font.init()
                self.pygame_initialized = True
                
                try:
                    self.font = pygame.font.SysFont(['SimHei', 'Microsoft YaHei', 'Arial'], 18)
                except:
                    self.font = pygame.font.Font(None, 18)
                
                self.clock = pygame.time.Clock()
                self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
                pygame.display.set_caption(self.window_title)
                
                # 初始化面板
                self.setup_panels()
                
                print("="  * 60)
                print(f"[OK] {self.window_title} 已启动")
                print("按ESC键关闭窗口")
                print("=" * 60)
        except Exception as e:
            print(f"[ERROR] Pygame初始化失败: {str(e)}")
            self.running = False
            return
        
        # 主循环
        while self.running:
            try:
                self.handle_events()
                self.screen.fill(self.SCREEN_BACKGROUND)
                
                # 更新数据
                grid_data, runtime_data_dict = self.update_data()
                
                # 绘制环境
                self.draw_center_area_border()  # 绘制中间区域边框
                self.draw_grid(grid_data)
                self.draw_obstacles(self._cached_obstacles)  # 绘制障碍物
                self.draw_leader(runtime_data_dict)
                self.draw_drones(runtime_data_dict)
                
                # 获取可视化数据并绘制面板
                vis_data = self.get_visualization_data()
                vis_data['grid_data'] = grid_data
                vis_data['runtime_data'] = runtime_data_dict
                
                # 添加电量数据
                battery_data = self.get_battery_data()
                if battery_data:
                    vis_data['battery_data'] = battery_data

                self.panel_manager.update_all_panels(vis_data)
                self.panel_manager.draw_all_panels(self.screen, vis_data)
                
                pygame.display.flip()
                self.clock.tick(max(1, int(getattr(self, "render_fps", 30))))  # render FPS
            except Exception as e:
                print(f"渲染循环出错: {str(e)}")
                time.sleep(0.05)
        
        # 退出
        try:
            pygame.quit()
        except:
            pass
    
    def start_visualization(self) -> bool:
        """在独立线程中启动可视化"""
        if not self.visualization_thread or not self.visualization_thread.is_alive():
            self.visualization_thread = threading.Thread(target=self.run, daemon=True)
            self.visualization_thread.start()
            return True
        return False
    
    def stop_visualization(self):
        """停止可视化"""
        self.running = False
        if self.visualization_thread and self.visualization_thread.is_alive():
            self.visualization_thread.join(timeout=2.0)
