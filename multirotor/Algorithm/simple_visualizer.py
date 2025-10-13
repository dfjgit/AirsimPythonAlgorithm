import sys
import math
import os
import threading
import time
from typing import Dict, List, Optional
import pygame

# 添加项目根目录到Python路径，使导入正常工作
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# 使用绝对导入
from multirotor.Algorithm.Vector3 import Vector3
from multirotor.Algorithm.scanner_runtime_data import ScannerRuntimeData
from multirotor.Algorithm.HexGridDataModel import HexGridDataModel

class SimpleVisualizer:
    def __init__(self, server=None):
        # 存储AlgorithmServer引用，用于获取实时数据
        self.server = server
        
        # 窗口设置
        self.SCREEN_WIDTH = 1000
        self.SCREEN_HEIGHT = 800
        
        # 标记是否已经初始化pygame
        self.pygame_initialized = False
        self.font_available = False
        self.font = None
        self.screen = None
        
        # 颜色定义
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.YELLOW = (255, 255, 0)
        self.CYAN = (0, 255, 255)
        self.MAGENTA = (255, 0, 255)
        self.GRAY = (128, 128, 128)
        self.LIGHT_GRAY = (200, 200, 200)
        
        # 新增颜色定义
        self.LIGHT_BLUE = (173, 216, 230)  # 淡蓝色用于Leader
        self.DRONE_GREEN = (50, 205, 50)   # 无人机绿色
        self.SCAN_RANGE_COLOR = (0, 255, 0)  # 扫描范围颜色
        
        # 坐标系转换参数
        self.origin_x = self.SCREEN_WIDTH // 2
        self.origin_y = self.SCREEN_HEIGHT // 2
        self.scale = 20  # 1单位=20像素
        
        # 可视化控制
        self.running = False
        self.clock = None  # 延迟初始化，在pygame.init()之后创建
        self.visualization_thread = None
    
    def world_to_screen(self, vector):
        """将世界坐标转换为屏幕坐标"""
        # 修正X轴方向：使用x轴作为x轴，z轴作为y轴进行2D投影
        screen_x = self.origin_x + vector.x * self.scale
        screen_y = self.origin_y - vector.z * self.scale  # 负号是因为pygame的y轴向下
        return int(screen_x), int(screen_y)
        
    def draw_grid(self, grid_data):
        """绘制网格（优化性能）"""
        if not grid_data or not hasattr(grid_data, 'cells'):
            return
        
        # 缓存小字体，避免重复创建
        if not hasattr(self, '_small_font'):
            try:
                self._small_font = pygame.font.SysFont(['SimHei', 'Microsoft YaHei', 'Arial'], 12)
            except:
                self._small_font = None
        
        # 限制绘制的网格数量，避免性能问题
        max_cells = 500  # 最多绘制500个网格点
        cells_to_draw = grid_data.cells[:max_cells] if len(grid_data.cells) > max_cells else grid_data.cells
        
        for cell in cells_to_draw:
            screen_x, screen_y = self.world_to_screen(cell.center)
            
            # 根据熵值决定颜色（红色到绿色渐变，熵值越小越绿）
            entropy_normalized = min(1.0, cell.entropy)
            # 红色分量：熵值高时红色多
            red = int(255 * entropy_normalized)
            # 绿色分量：熵值低时绿色多
            green = int(255 * (1 - entropy_normalized))
            # 蓝色分量：固定为0
            blue = 0
            color = (red, green, blue)
            
            # 绘制单一的点
            pygame.draw.circle(self.screen, color, (screen_x, screen_y), 3)
            
            # 显示熵值数值（减少文本渲染频率）
            if self._small_font and cell.entropy > 0.1:  # 只显示有意义的熵值
                entropy_text = f"{cell.entropy:.1f}"  # 减少精度
                text_surface = self._small_font.render(entropy_text, True, self.WHITE)
                text_rect = text_surface.get_rect(center=(screen_x, screen_y - 15))
                self.screen.blit(text_surface, text_rect)
    
    def draw_hexagon(self, x, y, color):
        """绘制六边形"""
        size = self.scale * 0.4
        points = []
        for i in range(6):
            angle_deg = 60 * i - 30
            angle_rad = math.pi / 180 * angle_deg
            points.append((x + size * math.cos(angle_rad), y + size * math.sin(angle_rad)))
        pygame.draw.polygon(self.screen, color, points, 1)
    
    def draw_drones(self, runtime_data_dict):
        """绘制所有无人机（优化性能）"""
        try:
            if not runtime_data_dict:
                return
            
            for drone_name, drone_info in runtime_data_dict.items():
                if not drone_info or 'position' not in drone_info or not drone_info['position']:
                    continue
                
                screen_x, screen_y = self.world_to_screen(drone_info['position'])
                
                # 绘制无人机扫描范围
                # 从服务器配置中获取扫描半径
                scan_radius_meters = 1.0  # 默认值
                if self.server and hasattr(self.server, 'config_data') and hasattr(self.server.config_data, 'scanRadius'):
                    scan_radius_meters = self.server.config_data.scanRadius
                
                scan_radius_pixels = scan_radius_meters * self.scale
                pygame.draw.circle(self.screen, self.SCAN_RANGE_COLOR, (screen_x, screen_y), int(scan_radius_pixels), 2)
                
                # 绘制无人机主体（绿色）
                pygame.draw.circle(self.screen, self.DRONE_GREEN, (screen_x, screen_y), 10)
                pygame.draw.circle(self.screen, self.WHITE, (screen_x, screen_y), 10, 2)  # 白色边框
                
                # 绘制方向指示
                if 'finalMoveDir' in drone_info and drone_info['finalMoveDir']:
                    dir_x = screen_x + drone_info['finalMoveDir'].x * 20
                    dir_y = screen_y - drone_info['finalMoveDir'].z * 20  # 注意坐标系转换
                    pygame.draw.line(self.screen, self.WHITE, (screen_x, screen_y), (dir_x, dir_y), 3)
                
                # 绘制无人机名称（缓存文本渲染）
                if not hasattr(self, '_drone_name_cache'):
                    self._drone_name_cache = {}
                
                if drone_name not in self._drone_name_cache:
                    self._drone_name_cache[drone_name] = self.font.render(drone_name, True, self.WHITE)
                
                self.screen.blit(self._drone_name_cache[drone_name], (screen_x + 15, screen_y - 10))
        except Exception as e:
            print(f"绘制无人机时出错: {str(e)}")
    
    def draw_leader(self, runtime_data_dict):
        """绘制领导者位置和扫描范围"""
        try:
            # 从第一个无人机的运行时数据中获取leader信息
            if runtime_data_dict:
                first_drone_data = next(iter(runtime_data_dict.values()))
                
                if first_drone_data and 'leaderPosition' in first_drone_data and first_drone_data['leaderPosition']:
                    screen_x, screen_y = self.world_to_screen(first_drone_data['leaderPosition'])
                    
                    # 绘制领导者位置（更大的圆圈，淡蓝色）
                    pygame.draw.circle(self.screen, self.LIGHT_BLUE, (screen_x, screen_y), 20)
                    pygame.draw.circle(self.screen, self.WHITE, (screen_x, screen_y), 20, 3)
                    
                    # 绘制扫描范围圆圈（淡蓝色边框）
                    if 'leaderScanRadius' in first_drone_data and first_drone_data['leaderScanRadius'] > 0:
                        radius = first_drone_data['leaderScanRadius'] * self.scale
                        # 绘制范围圆圈（淡蓝色边框）
                        pygame.draw.circle(self.screen, self.LIGHT_BLUE, (screen_x, screen_y), radius, 3)
                        
                        # 绘制范围信息
                        if self.font:
                            range_text = f"Range: {first_drone_data['leaderScanRadius']:.1f}m"
                            text_surface = self.font.render(range_text, True, self.LIGHT_BLUE)
                            text_rect = text_surface.get_rect(center=(screen_x, screen_y + radius + 20))
                            self.screen.blit(text_surface, text_rect)
                    
                    # 绘制领导者标签
                    if self.font:
                        text = self.font.render("Leader", True, self.WHITE)
                        text_rect = text.get_rect(center=(screen_x, screen_y - 35))
                        self.screen.blit(text, text_rect)
        except Exception as e:
            print(f"绘制领导者时出错: {str(e)}")
    
    def draw_status_info(self):
        """绘制状态信息"""
        # 创建小字体用于状态信息
        if not hasattr(self, '_status_font'):
            try:
                self._status_font = pygame.font.SysFont(['SimHei', 'Microsoft YaHei', 'Arial'], 16)
            except:
                self._status_font = self.font
        
        # 绘制状态面板背景
        panel_rect = pygame.Rect(10, 10, 300, 120)
        pygame.draw.rect(self.screen, self.GRAY, panel_rect)
        pygame.draw.rect(self.screen, self.WHITE, panel_rect, 2)
        
        # 绘制标题
        title = self._status_font.render("可视化状态", True, self.WHITE)
        self.screen.blit(title, (20, 15))
        
        # 绘制无人机数量
        if self.server and hasattr(self.server, 'drone_names'):
            drone_count = len(self.server.drone_names)
            text = self._status_font.render(f"无人机数量: {drone_count}", True, self.WHITE)
            self.screen.blit(text, (20, 40))
        
        # 绘制是否启用学习
        if self.server and hasattr(self.server, 'enable_learning'):
            learning_status = "已启用" if self.server.enable_learning else "已禁用"
            text = self._status_font.render(f"DQN学习: {learning_status}", True, self.WHITE)
            self.screen.blit(text, (20, 60))
        
        # 绘制平均熵值
        avg_entropy = self._calculate_average_entropy()
        if avg_entropy is not None:
            text = self._status_font.render(f"平均熵值: {avg_entropy:.3f}", True, self.WHITE)
            self.screen.blit(text, (20, 80))
    
    def _calculate_average_entropy(self):
        """计算平均熵值"""
        try:
            if not self.server or not hasattr(self.server, 'grid_data') or not self.server.grid_data:
                return None
            
            grid_data = self.server.grid_data
            if not hasattr(grid_data, 'cells') or not grid_data.cells:
                return None
            
            total_entropy = sum(cell.entropy for cell in grid_data.cells)
            avg_entropy = total_entropy / len(grid_data.cells)
            return avg_entropy
        except Exception:
            return None
    
    def draw_instructions(self):
        """绘制操作说明"""
        # 创建小字体用于操作说明
        if not hasattr(self, '_instruction_font'):
            try:
                self._instruction_font = pygame.font.SysFont(['SimHei', 'Microsoft YaHei', 'Arial'], 14)
            except:
                self._instruction_font = self.font
        
        instructions = [
            "- 实时显示环境、无人机和领导者位置",
            "- ESC键退出可视化"
        ]
        
        instruction_y = self.SCREEN_HEIGHT - 60
        for instruction in instructions:
            text = self._instruction_font.render(instruction, True, self.LIGHT_GRAY)
            self.screen.blit(text, (20, instruction_y))
            instruction_y += 25
    
    def handle_events(self):
        """处理基本事件，确保窗口响应"""
        # 设置事件队列不阻塞，防止窗口冻结
        try:
            # 使用pygame.NOEVENT来清空事件队列，确保窗口响应
            event_queue = pygame.event.get()
            if not event_queue:
                # 如果没有事件，添加一个自定义事件以保持循环活跃
                pygame.event.post(pygame.event.Event(pygame.NOEVENT))
            else:
                for event in event_queue:
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self.running = False
        except Exception as e:
            print(f"事件处理出错: {str(e)}")
    
    def update_data(self):
        """更新可视化数据，优化性能避免卡顿"""
        try:
            if not self.server:
                return None, {}
            
            # 使用缓存机制，减少数据访问频率
            current_time = time.time()
            if hasattr(self, '_last_data_update') and current_time - self._last_data_update < 0.05:  # 20fps数据更新
                return getattr(self, '_cached_grid_data', None), getattr(self, '_cached_runtime_data', {})
            
            # 获取网格数据（快速访问）
            grid_data = None
            try:
                if hasattr(self.server, 'grid_data'):
                    grid_data = self.server.grid_data
            except Exception:
                pass
            
            # 获取运行时数据（优化访问）
            runtime_data_dict = {}
            try:
                if hasattr(self.server, 'unity_runtime_data'):
                    # 使用更高效的数据访问方式
                    unity_data = self.server.unity_runtime_data
                    for drone_name, runtime_data in unity_data.items():
                        if runtime_data:
                            # 直接访问属性，避免getattr开销
                            drone_info = {
                                'position': runtime_data.position,
                                'finalMoveDir': runtime_data.finalMoveDir,
                                'leaderPosition': runtime_data.leader_position,
                                'leaderScanRadius': runtime_data.leader_scan_radius
                            }
                            runtime_data_dict[drone_name] = drone_info
            except Exception:
                pass
            
            # 缓存数据
            self._cached_grid_data = grid_data
            self._cached_runtime_data = runtime_data_dict
            self._last_data_update = current_time
            
            return grid_data, runtime_data_dict
        except Exception as e:
            print(f"更新可视化数据时出错: {str(e)}")
            return getattr(self, '_cached_grid_data', None), getattr(self, '_cached_runtime_data', {})
    
    def run(self):
        """主循环"""
        self.running = True
        
        # 确保pygame正确初始化
        try:
            if not self.pygame_initialized:
                pygame.init()
                self.pygame_initialized = True
                
                # 初始化时钟
                self.clock = pygame.time.Clock()
                
                # 初始化字体系统
                pygame.font.init()
                
                # 设置中文字体支持
                try:
                    self.font = pygame.font.SysFont(['SimHei', 'Microsoft YaHei', 'Arial'], 24)
                    self.font_available = True
                except Exception as font_error:
                    print(f"字体加载失败: {str(font_error)}")
                    # 使用默认字体作为备选
                    self.font = pygame.font.Font(None, 24)
                    self.font_available = False
                
                # 创建窗口
                self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
                pygame.display.set_caption("无人机环境实时可视化")
        except Exception as e:
            print(f"Pygame初始化失败: {str(e)}")
            self.running = False
            return
        
        while self.running:
            try:
                # 处理事件
                self.handle_events()
                
                # 清屏
                self.screen.fill(self.BLACK)
                
                # 更新数据（优化后的方法）
                grid_data, runtime_data_dict = self.update_data()
                
                # 绘制元素（优化绘制顺序）
                try:
                    self.draw_grid(grid_data)
                except Exception as e:
                    print(f"绘制网格时出错: {str(e)}")
                
                try:
                    self.draw_leader(runtime_data_dict)
                except Exception as e:
                    print(f"绘制领导者时出错: {str(e)}")
                
                try:
                    self.draw_drones(runtime_data_dict)
                except Exception as e:
                    print(f"绘制无人机时出错: {str(e)}")
                
                try:
                    self.draw_status_info()
                    self.draw_instructions()
                except Exception as e:
                    print(f"绘制UI时出错: {str(e)}")
                
                # 更新屏幕
                pygame.display.flip()
                
                # 优化帧率控制，使用更稳定的帧率
                if self.clock:
                    self.clock.tick(60)  # 提高到60fps，使用标准tick方法
            except Exception as e:
                print(f"可视化主循环出错: {str(e)}")
                # 短暂暂停后继续，避免错误导致程序崩溃
                time.sleep(0.05)  # 减少错误恢复时间
        
        # 退出pygame
        try:
            pygame.quit()
        except Exception as e:
            print(f"退出pygame时出错: {str(e)}")
    
    def start_visualization(self):
        """在独立线程中启动可视化"""
        if not self.visualization_thread or not self.visualization_thread.is_alive():
            self.visualization_thread = threading.Thread(target=self.run)
            self.visualization_thread.daemon = True  # 设置为守护线程，主程序结束时自动退出
            self.visualization_thread.start()
            return True
        return False
    
    def stop_visualization(self):
        """停止可视化"""
        self.running = False
        if self.visualization_thread and self.visualization_thread.is_alive():
            self.visualization_thread.join(timeout=2.0)  # 等待线程结束，最多等待2秒