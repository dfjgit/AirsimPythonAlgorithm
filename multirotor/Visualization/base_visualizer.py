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
        
        # 窗口设置（左右两侧面板 + 中间热力图）
        self.SCREEN_WIDTH = 1920   # 扩大窗口宽度
        self.SCREEN_HEIGHT = 1080  # 扩大窗口高度（1080p）
        self.left_panel_width = 360   # 略微减小左侧面板
        self.right_panel_width = 360  # 略微减小右侧面板
        
        # 坐标系参数（中间热力图区域）
        self.view_width = self.SCREEN_WIDTH - self.left_panel_width - self.right_panel_width
        self.view_height = self.SCREEN_HEIGHT
        self.view_offset_x = self.left_panel_width  # 热力图区域起始 x
        self.origin_x = self.view_offset_x + self.view_width // 2
        self.origin_y = self.view_height // 2
        self.scale = 20  # 像素/米
        
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
        self._last_data_update = 0
        self._data_update_interval = 0.05  # 50ms更新一次
    
    def _init_colors(self):
        """初始化颜色常量"""
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
        self.DARK_GRAY = (64, 64, 64)
        self.LIGHT_BLUE = (173, 216, 230)
        self.DRONE_GREEN = (50, 205, 50)
        self.SCAN_RANGE_COLOR = (0, 255, 0)
    
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
                red = int(510 * entropy_normalized)
                green = 255
            else:
                red = 255
                green = int(255 * (2 - 2 * entropy_normalized))
            
            color = (red, green, 0)
            
            # 只绘制可见区域
            if 0 <= screen_x <= self.SCREEN_WIDTH and 0 <= screen_y <= self.SCREEN_HEIGHT:
                radius = 2 if cell.entropy < 30 else (3 if cell.entropy < 70 else 4)
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
            pygame.draw.circle(self.screen, self.SCAN_RANGE_COLOR, (screen_x, screen_y), 
                             int(scan_radius_pixels), 2)
            
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
                pygame.draw.circle(self.screen, self.LIGHT_BLUE, (screen_x, screen_y), int(radius), 3)
            
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
            s.fill((0, 0, 0))
            self.screen.blit(s, (legend_x, legend_y))
            pygame.draw.rect(self.screen, self.WHITE, background_rect, 1)
            
            # 标题
            if not hasattr(self, '_legend_font'):
                try:
                    self._legend_font = pygame.font.SysFont(['SimHei', 'Microsoft YaHei', 'Arial'], 14)
                except:
                    self._legend_font = pygame.font.Font(None, 14)
            
            title = self._legend_font.render("Entropy", True, self.WHITE)
            self.screen.blit(title, (legend_x + 5, legend_y + 5))
            
            # 渐变颜色条
            bar_x = legend_x + 5
            bar_y = legend_y + 25
            bar_width = legend_width - 10
            bar_height = 15
            
            for i in range(bar_width):
                entropy_normalized = i / bar_width
                if entropy_normalized < 0.5:
                    red = int(510 * entropy_normalized)
                    green = 255
                else:
                    red = 255
                    green = int(255 * (2 - 2 * entropy_normalized))
                
                color = (red, green, 0)
                pygame.draw.line(self.screen, color, (bar_x + i, bar_y), (bar_x + i, bar_y + bar_height))
            
            # 刻度标签
            label_0 = self._legend_font.render("0", True, self.WHITE)
            self.screen.blit(label_0, (bar_x, bar_y + bar_height + 2))
            
            label_100 = self._legend_font.render("100", True, self.WHITE)
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
        
        # 缓存数据
        self._cached_grid_data = grid_data
        self._cached_runtime_data = runtime_data_dict
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
                self.screen.fill(self.BLACK)
                
                # 更新数据
                grid_data, runtime_data_dict = self.update_data()
                
                # 绘制环境
                self.draw_center_area_border()  # 绘制中间区域边框
                self.draw_grid(grid_data)
                self.draw_leader(runtime_data_dict)
                self.draw_drones(runtime_data_dict)
                self.draw_entropy_legend()
                
                # 获取可视化数据并绘制面板
                vis_data = self.get_visualization_data()
                vis_data['grid_data'] = grid_data
                vis_data['runtime_data'] = runtime_data_dict
                
                # 添加电量数据
                battery_data = self.get_battery_data()
                if battery_data:
                    vis_data['battery_data'] = battery_data
                
                self.panel_manager.draw_all_panels(self.screen, vis_data)
                
                pygame.display.flip()
                self.clock.tick(30)  # 30 FPS
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
