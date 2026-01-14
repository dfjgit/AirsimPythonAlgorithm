"""
训练专用可视化模块
显示训练过程中的实时信息：episode统计、奖励曲线、权重变化等
"""
import sys
import math
import os
import threading
import time
from typing import Dict, List, Optional, Deque
from collections import deque
import pygame

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# 使用绝对导入
from multirotor.Algorithm.Vector3 import Vector3
from multirotor.Algorithm.scanner_runtime_data import ScannerRuntimeData
from multirotor.Algorithm.HexGridDataModel import HexGridDataModel


class TrainingVisualizer:
    """训练专用可视化器 - 显示训练统计和环境状态"""
    
    def __init__(self, server=None, env=None):
        """
        初始化训练可视化器
        :param server: AlgorithmServer实例
        :param env: SimpleWeightEnv训练环境实例
        """
        # 存储引用
        self.server = server
        self.env = env
        
        # 窗口设置（更大的窗口以容纳训练信息）
        self.SCREEN_WIDTH = 1400
        self.SCREEN_HEIGHT = 900
        
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
        self.DARK_GRAY = (64, 64, 64)
        self.LIGHT_BLUE = (173, 216, 230)
        self.DRONE_GREEN = (50, 205, 50)
        self.SCAN_RANGE_COLOR = (0, 255, 0)
        self.ORANGE = (255, 165, 0)
        self.PURPLE = (160, 32, 240)
        self.DARK_ORANGE = (255, 140, 0)
        
        # 坐标系转换参数（所有面板在右侧，环境视图可以占据左侧和中央）
        self.origin_x = (self.SCREEN_WIDTH - 400) // 2  # 环境视图居中显示（考虑右侧面板宽度）
        self.origin_y = self.SCREEN_HEIGHT // 2
        self.scale = 20  # 1单位=20像素
        
        # 训练统计数据
        self.episode_rewards: Deque[float] = deque(maxlen=100)  # 最近100个episode的奖励
        self.episode_lengths: Deque[int] = deque(maxlen=100)
        self.episode_count = 0
        self.total_steps = 0
        self.current_episode_reward = 0.0
        self.current_episode_steps = 0
        
        # 步骤速率统计
        self.step_timestamps: Deque[float] = deque(maxlen=100)  # 最近100步的时间戳
        self.training_start_time = time.time()
        self.last_step_time = time.time()
        
        # 奖励曲线历史（用于绘图）
        self.reward_history: Deque[float] = deque(maxlen=50)  # 最近50个数据点
        
        # 权重历史
        self.weight_history: Dict[str, Deque[float]] = {
            'repulsionCoefficient': deque(maxlen=50),
            'entropyCoefficient': deque(maxlen=50),
            'distanceCoefficient': deque(maxlen=50),
            'leaderRangeCoefficient': deque(maxlen=50),
            'directionRetentionCoefficient': deque(maxlen=50)
        }
        
        # 可视化控制
        self.running = False
        self.clock = None
        self.visualization_thread = None
        
        # 数据缓存
        self._cached_grid_data = None
        self._cached_runtime_data = {}
        self._last_data_update = 0
    
    def update_training_stats(self, episode_reward: float = None, episode_length: int = None, 
                             current_step_reward: float = None, is_episode_done: bool = False):
        """
        更新训练统计信息
        :param episode_reward: 当前episode的总奖励
        :param episode_length: 当前episode的长度
        :param current_step_reward: 当前步的奖励
        :param is_episode_done: 是否episode结束
        """
        if current_step_reward is not None:
            self.current_episode_reward += current_step_reward
            self.current_episode_steps += 1
            self.total_steps += 1
            
            # 记录步骤时间戳（用于计算速率）
            current_time = time.time()
            self.step_timestamps.append(current_time)
            self.last_step_time = current_time
        
        if is_episode_done and episode_reward is not None:
            self.episode_rewards.append(episode_reward)
            self.reward_history.append(episode_reward)
            if episode_length is not None:
                self.episode_lengths.append(episode_length)
            self.episode_count += 1
            
            # 重置当前episode统计
            self.current_episode_reward = 0.0
            self.current_episode_steps = 0
    
    def update_weight_history(self, weights: Dict[str, float]):
        """更新权重历史"""
        for key, value in weights.items():
            if key in self.weight_history:
                self.weight_history[key].append(value)
    
    def world_to_screen(self, vector):
        """将世界坐标转换为屏幕坐标"""
        screen_x = self.origin_x + vector.x * self.scale
        screen_y = self.origin_y - vector.z * self.scale
        return int(screen_x), int(screen_y)
    
    def draw_grid(self, grid_data):
        """绘制网格（显示所有熵值）"""
        if not grid_data or not hasattr(grid_data, 'cells'):
            return
        
        # 缓存小字体
        if not hasattr(self, '_small_font'):
            try:
                self._small_font = pygame.font.SysFont(['SimHei', 'Microsoft YaHei', 'Arial'], 10)
            except:
                self._small_font = None
        
        for cell in grid_data.cells:
            screen_x, screen_y = self.world_to_screen(cell.center)
            
            # 根据熵值决定颜色（绿色到红色渐变）
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
    
    def draw_drones(self, runtime_data_dict):
        """绘制无人机"""
        try:
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
                pygame.draw.circle(self.screen, self.SCAN_RANGE_COLOR, (screen_x, screen_y), int(scan_radius_pixels), 2)
                
                # 绘制无人机
                pygame.draw.circle(self.screen, self.DRONE_GREEN, (screen_x, screen_y), 10)
                pygame.draw.circle(self.screen, self.WHITE, (screen_x, screen_y), 10, 2)
                
                # 绘制方向指示
                if 'finalMoveDir' in drone_info and drone_info['finalMoveDir']:
                    dir_x = screen_x + drone_info['finalMoveDir'].x * 20
                    dir_y = screen_y - drone_info['finalMoveDir'].z * 20
                    pygame.draw.line(self.screen, self.WHITE, (screen_x, screen_y), (dir_x, dir_y), 3)
                
                # 绘制电量信息
                if 'battery_voltage' in drone_info:
                    voltage = drone_info['battery_voltage']
                    
                    # 根据电量决定颜色
                    if voltage >= 3.7:
                        battery_color = self.GREEN
                    elif voltage >= 3.5:
                        battery_color = self.ORANGE
                    else:
                        battery_color = self.RED
                    
                    # 绘制电量文本
                    voltage_text = f"{voltage:.2f}V"
                    if not hasattr(self, '_battery_font'):
                        try:
                            self._battery_font = pygame.font.SysFont(['SimHei', 'Microsoft YaHei', 'Arial'], 10)
                        except:
                            self._battery_font = None
                    
                    if self._battery_font:
                        text_surface = self._battery_font.render(voltage_text, True, battery_color)
                        self.screen.blit(text_surface, (screen_x - 15, screen_y - 25))
                    
                    # 绘制电量条
                    battery_width = 20
                    battery_height = 6
                    battery_x = screen_x - battery_width // 2
                    battery_y = screen_y - 35
                    
                    # 电量百分比 (4.2V为100%, 3.0V为0%)
                    battery_percent = max(0, min(1, (voltage - 3.0) / (4.2 - 3.0)))
                    
                    # 背景条
                    pygame.draw.rect(self.screen, self.DARK_GRAY, (battery_x, battery_y, battery_width, battery_height))
                    
                    # 电量填充
                    fill_width = int(battery_width * battery_percent)
                    if fill_width > 0:
                        pygame.draw.rect(self.screen, battery_color, (battery_x, battery_y, fill_width, battery_height))
                    
                    # 边框
                    pygame.draw.rect(self.screen, self.WHITE, (battery_x, battery_y, battery_width, battery_height), 1)
                
                # 绘制名称
                if not hasattr(self, '_drone_name_cache'):
                    self._drone_name_cache = {}
                
                if drone_name not in self._drone_name_cache:
                    self._drone_name_cache[drone_name] = self.font.render(drone_name, True, self.WHITE)
                
                self.screen.blit(self._drone_name_cache[drone_name], (screen_x + 15, screen_y - 10))
        except Exception as e:
            print(f"绘制无人机时出错: {str(e)}")
    
    def draw_leader(self, runtime_data_dict):
        """绘制领导者位置"""
        try:
            if runtime_data_dict:
                first_drone_data = next(iter(runtime_data_dict.values()))
                
                if first_drone_data and 'leaderPosition' in first_drone_data and first_drone_data['leaderPosition']:
                    screen_x, screen_y = self.world_to_screen(first_drone_data['leaderPosition'])
                    
                    pygame.draw.circle(self.screen, self.LIGHT_BLUE, (screen_x, screen_y), 20)
                    pygame.draw.circle(self.screen, self.WHITE, (screen_x, screen_y), 20, 3)
                    
                    if 'leaderScanRadius' in first_drone_data and first_drone_data['leaderScanRadius'] > 0:
                        radius = first_drone_data['leaderScanRadius'] * self.scale
                        pygame.draw.circle(self.screen, self.LIGHT_BLUE, (screen_x, screen_y), radius, 3)
                    
                    if self.font:
                        text = self.font.render("Leader", True, self.WHITE)
                        text_rect = text.get_rect(center=(screen_x, screen_y - 35))
                        self.screen.blit(text, text_rect)
        except Exception as e:
            print(f"绘制领导者时出错: {str(e)}")
    
    def draw_training_info_panel(self):
        """绘制训练信息面板（右上角）- 增强版"""
        if not hasattr(self, '_info_font'):
            try:
                self._info_font = pygame.font.SysFont(['SimHei', 'Microsoft YaHei', 'Arial'], 14)
            except:
                self._info_font = self.font
        
        # 创建大字体用于步骤计数器
        if not hasattr(self, '_big_font'):
            try:
                self._big_font = pygame.font.SysFont(['SimHei', 'Microsoft YaHei', 'Arial'], 24, bold=True)
            except:
                self._big_font = pygame.font.Font(None, 24)
        
        panel_x = self.SCREEN_WIDTH - 380
        panel_y = 10
        panel_width = 370
        panel_height = 340  # 增加高度以容纳更多信息
        
        # 半透明背景
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        s = pygame.Surface((panel_width, panel_height))
        s.set_alpha(220)
        s.fill((0, 0, 0))
        self.screen.blit(s, (panel_x, panel_y))
        pygame.draw.rect(self.screen, self.YELLOW, panel_rect, 2)
        
        y = panel_y + 10
        
        # 标题
        title = self._info_font.render("🎯 训练状态", True, self.YELLOW)
        self.screen.blit(title, (panel_x + 10, y))
        y += 25
        
        # ========== 醒目的步骤计数器 ==========
        step_text = self._big_font.render(f"步数: {self.total_steps}", True, self.CYAN)
        self.screen.blit(step_text, (panel_x + 15, y))
        y += 30
        
        # 计算并显示训练进度（如果env有max_steps信息）
        if self.env and hasattr(self.env, 'reward_config'):
            max_steps = getattr(self.env.reward_config, 'max_steps', 50)
# 假设训练目标是完成一定数量的episodes
            # 这里可以显示当前episode内的进度
            if max_steps > 0:
                progress = min(self.current_episode_steps / max_steps * 100, 100)
                
                # 进度条
                bar_x = panel_x + 15
                bar_y = y + 3
                bar_width = 340
                bar_height = 12
                
                # 背景条
                pygame.draw.rect(self.screen, self.DARK_GRAY, (bar_x, bar_y, bar_width, bar_height))
                
                # 填充条
                fill_width = int(bar_width * (progress / 100))
                if fill_width > 0:
                    # 根据进度改变颜色
                    if progress < 33:
                        color = self.RED
                    elif progress < 66:
                        color = self.ORANGE
                    else:
                        color = self.GREEN
                    pygame.draw.rect(self.screen, color, (bar_x, bar_y, fill_width, bar_height))
                
                # 边框
                pygame.draw.rect(self.screen, self.WHITE, (bar_x, bar_y, bar_width, bar_height), 1)
                
                # 进度文本
                progress_text = self._info_font.render(f"Episode进度: {progress:.1f}%", True, self.WHITE)
                self.screen.blit(progress_text, (bar_x, bar_y - 15))
                
                y += 30
        
        # 计算步骤速率
        steps_per_sec = 0.0
        if len(self.step_timestamps) > 1:
            time_span = self.step_timestamps[-1] - self.step_timestamps[0]
            if time_span > 0:
                steps_per_sec = len(self.step_timestamps) / time_span
        
        # 显示步骤速率
        rate_text = self._info_font.render(f"速率: {steps_per_sec:.2f} steps/s", True, self.GREEN)
        self.screen.blit(rate_text, (panel_x + 15, y))
        y += 18
        # 计算训练已用时间
        elapsed_time = time.time() - self.training_start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        time_text = self._info_font.render(f"已用时间: {hours:02d}:{minutes:02d}:{seconds:02d}", True, self.WHITE)
        self.screen.blit(time_text, (panel_x + 15, y))
        y += 23
        
        # 分隔线
        pygame.draw.line(self.screen, self.GRAY, (panel_x + 10, y), (panel_x + panel_width - 10, y), 1)
        y += 5
        
        # Episode信息
        text = self._info_font.render(f"Episode: {self.episode_count}", True, self.WHITE)
        self.screen.blit(text, (panel_x + 15, y))
        y += 18
        
        # 当前episode信息
        text = self._info_font.render(f"当前Episode步数: {self.current_episode_steps}", True, self.CYAN)
        self.screen.blit(text, (panel_x + 15, y))
        y += 18
        
        text = self._info_font.render(f"当前Episode奖励: {self.current_episode_reward:.2f}", True, self.CYAN)
        self.screen.blit(text, (panel_x + 15, y))
        y += 23
        
        # 统计信息
        if len(self.episode_rewards) > 0:
            avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)
            max_reward = max(self.episode_rewards)
            min_reward = min(self.episode_rewards)
            
            text = self._info_font.render(f"平均奖励: {avg_reward:.2f}", True, self.GREEN)
            self.screen.blit(text, (panel_x + 15, y))
            y += 18
            
            text = self._info_font.render(f"最佳奖励: {max_reward:.2f}", True, self.GREEN)
            self.screen.blit(text, (panel_x + 15, y))
            y += 18
            
            text = self._info_font.render(f"最差奖励: {min_reward:.2f}", True, self.RED)
            self.screen.blit(text, (panel_x + 15, y))
            y += 18
        
        if len(self.episode_lengths) > 0:
            avg_length = sum(self.episode_lengths) / len(self.episode_lengths)
            text = self._info_font.render(f"平均步长: {avg_length:.1f}", True, self.WHITE)
            self.screen.blit(text, (panel_x + 15, y))
            y += 25
        
        # 环境信息
        if self.env:
            max_steps = getattr(self.env.reward_config, 'max_steps', 50)
            text = self._info_font.render(f"Episode最大步数: {max_steps}", True, self.GRAY)
            self.screen.blit(text, (panel_x + 15, y))
            y += 18
    
    def draw_reward_curve(self):
        """绘制奖励曲线（右侧，训练统计面板下方）"""
        if len(self.reward_history) < 2:
            return
        
        if not hasattr(self, '_curve_font'):
            try:
                self._curve_font = pygame.font.SysFont(['SimHei', 'Microsoft YaHei', 'Arial'], 12)
            except:
                self._curve_font = self.font
        
        panel_x = self.SCREEN_WIDTH - 380
        panel_y = 360  # 在训练统计面板（340高）下方，留20px间距
        panel_width = 370
        panel_height = 200  # 缩小高度
        
        # 半透明背景
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        s = pygame.Surface((panel_width, panel_height))
        s.set_alpha(220)
        s.fill((0, 0, 0))
        self.screen.blit(s, (panel_x, panel_y))
        pygame.draw.rect(self.screen, self.CYAN, panel_rect, 2)
        
        # 标题
        title = self._curve_font.render("📈 奖励曲线（最近50个Episode）", True, self.CYAN)
        self.screen.blit(title, (panel_x + 10, panel_y + 5))
        
        # 图表区域
        graph_x = panel_x + 40
        graph_y = panel_y + 30
        graph_width = panel_width - 50
        graph_height = panel_height - 60
        
        # 绘制坐标轴
        pygame.draw.line(self.screen, self.LIGHT_GRAY, 
                        (graph_x, graph_y + graph_height), 
                        (graph_x + graph_width, graph_y + graph_height), 2)  # X轴
        pygame.draw.line(self.screen, self.LIGHT_GRAY, 
                        (graph_x, graph_y), 
                        (graph_x, graph_y + graph_height), 2)  # Y轴
        
        # 计算缩放
        rewards = list(self.reward_history)
        if not rewards:
            return
        
        max_reward = max(rewards)
        min_reward = min(rewards)
        reward_range = max_reward - min_reward if max_reward != min_reward else 1.0
        
        # 绘制曲线
        points = []
        for i, reward in enumerate(rewards):
            x = graph_x + (i / (len(rewards) - 1)) * graph_width
            y = graph_y + graph_height - ((reward - min_reward) / reward_range) * graph_height
            points.append((int(x), int(y)))
        
        if len(points) > 1:
            pygame.draw.lines(self.screen, self.GREEN, False, points, 2)
        
        # 绘制数据点
        for point in points:
            pygame.draw.circle(self.screen, self.YELLOW, point, 3)
        
        # Y轴标签
        label_max = self._curve_font.render(f"{max_reward:.1f}", True, self.WHITE)
        self.screen.blit(label_max, (graph_x - 35, graph_y - 5))
        
        label_min = self._curve_font.render(f"{min_reward:.1f}", True, self.WHITE)
        self.screen.blit(label_min, (graph_x - 35, graph_y + graph_height - 10))
        
        # 平均线
        if rewards:
            avg_reward = sum(rewards) / len(rewards)
            avg_y = graph_y + graph_height - ((avg_reward - min_reward) / reward_range) * graph_height
            pygame.draw.line(self.screen, self.ORANGE, 
                           (graph_x, int(avg_y)), 
                           (graph_x + graph_width, int(avg_y)), 1, )
            
            label_avg = self._curve_font.render(f"Avg: {avg_reward:.1f}", True, self.ORANGE)
            self.screen.blit(label_avg, (graph_x + graph_width - 60, int(avg_y) - 15))
    
    def draw_current_weights(self):
        """绘制当前权重（右侧，环境信息面板下方）"""
        if not self.server or not self.server.drone_names:
            return
        
        if not hasattr(self, '_weight_font'):
            try:
                self._weight_font = pygame.font.SysFont(['SimHei', 'Microsoft YaHei', 'Arial'], 13)
            except:
                self._weight_font = self.font
        
        panel_x = self.SCREEN_WIDTH - 380
        panel_y = 720  # 在环境信息面板下方
        panel_width = 370
        panel_height = 170
        
        # 半透明背景
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        s = pygame.Surface((panel_width, panel_height))
        s.set_alpha(220)
        s.fill((0, 0, 0))
        self.screen.blit(s, (panel_x, panel_y))
        pygame.draw.rect(self.screen, self.PURPLE, panel_rect, 2)
        
        y = panel_y + 10
        
        # 标题
        title = self._weight_font.render("⚙️ 当前APF权重（训练中动态调整）", True, self.PURPLE)
        self.screen.blit(title, (panel_x + 10, y))
        y += 25
        
        # 获取第一个无人机的权重
        first_drone = self.server.drone_names[0]
        if first_drone in self.server.algorithms:
            try:
                weights = self.server.algorithms[first_drone].get_current_coefficients()
                
                weight_info = [
                    ("α1 排斥", weights.get('repulsionCoefficient', 0)),
                    ("α2 熵值", weights.get('entropyCoefficient', 0)),
                    ("α3 距离", weights.get('distanceCoefficient', 0)),
                    ("α4 Leader", weights.get('leaderRangeCoefficient', 0)),
                    ("α5 方向", weights.get('directionRetentionCoefficient', 0))
                ]
                
                for name, value in weight_info:
                    text = self._weight_font.render(f"{name}: {value:.2f}", True, self.LIGHT_BLUE)
                    self.screen.blit(text, (panel_x + 15, y))
                    
                    # 权重条
                    bar_x = panel_x + 130
                    bar_y = y + 3
                    bar_width = 120
                    bar_height = 10
                    
                    pygame.draw.rect(self.screen, self.GRAY, (bar_x, bar_y, bar_width, bar_height))
                    
                    fill_width = int(bar_width * min((value - 0.5) / 4.5, 1.0))
                    if fill_width > 0:
                        color = self.GREEN if value < 1.5 else (self.YELLOW if value < 3.0 else self.RED)
                        pygame.draw.rect(self.screen, color, (bar_x, bar_y, fill_width, bar_height))
                    
                    pygame.draw.rect(self.screen, self.WHITE, (bar_x, bar_y, bar_width, bar_height), 1)
                    
                    # 数值
                    value_text = self._weight_font.render(f"{value:.2f}", True, self.WHITE)
                    self.screen.blit(value_text, (bar_x + bar_width + 5, y))
                    
                    y += 20
                    
            except Exception as e:
                error_text = self._weight_font.render(f"权重获取失败", True, self.RED)
                self.screen.blit(error_text, (panel_x + 15, y))
    
    def draw_env_info(self):
        """绘制环境信息面板（左上角）- 增强版"""
        if not hasattr(self, '_env_font'):
            try:
                self._env_font = pygame.font.SysFont(['SimHei', 'Microsoft YaHei', 'Arial'], 14)
            except:
                self._env_font = self.font
        
        panel_x = 10
        panel_y = 10
        panel_width = 300
        panel_height = 160  # 增加高度以容纳电量信息
        
        # 半透明背景
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        s = pygame.Surface((panel_width, panel_height))
        s.set_alpha(220)
        s.fill((0, 0, 0))
        self.screen.blit(s, (panel_x, panel_y))
        pygame.draw.rect(self.screen, self.GREEN, panel_rect, 2)
        
        y = panel_y + 10
        
        # 标题
        title = self._env_font.render("🌍 环境状态", True, self.GREEN)
        self.screen.blit(title, (panel_x + 10, y))
        y += 25
        
        # 网格统计
        grid_stats = self._calculate_grid_stats()
        if grid_stats:
            text1 = self._env_font.render(f"网格单元: {grid_stats['total']}", True, self.WHITE)
            self.screen.blit(text1, (panel_x + 15, y))
            y += 20
            
            text2 = self._env_font.render(f"平均熵值: {grid_stats['avg']:.1f}", True, self.WHITE)
            self.screen.blit(text2, (panel_x + 15, y))
            y += 20
            
            text3 = self._env_font.render(f"已扫描: {grid_stats['scanned']} ({grid_stats['scan_ratio']:.1f}%)", True, self.CYAN)
            self.screen.blit(text3, (panel_x + 15, y))
            y += 20
        
        # 电量统计
        battery_stats = self._calculate_battery_stats()
        if battery_stats:
            avg_voltage = battery_stats['avg_voltage']
            min_voltage = battery_stats['min_voltage']
            
            # 根据平均电量决定颜色
            if avg_voltage >= 3.7:
                voltage_color = self.GREEN
            elif avg_voltage >= 3.5:
                voltage_color = self.ORANGE
            else:
                voltage_color = self.RED
            
            text4 = self._env_font.render(f"平均电量: {avg_voltage:.2f}V", True, voltage_color)
            self.screen.blit(text4, (panel_x + 15, y))
            y += 20
            
            text5 = self._env_font.render(f"最低电量: {min_voltage:.2f}V", True, voltage_color)
            self.screen.blit(text5, (panel_x + 15, y))
            y += 20
        
        # 训练模式提示
        mode_text = self._env_font.render("模式: DQN权重训练", True, self.ORANGE)
        self.screen.blit(mode_text, (panel_x + 15, y))
    
    def _calculate_grid_stats(self):
        """计算网格统计信息"""
        try:
            if not self.server or not hasattr(self.server, 'grid_data'):
                return None
            
            grid_data = self.server.grid_data
            if not hasattr(grid_data, 'cells') or not grid_data.cells:
                return None
            
            total = len(grid_data.cells)
            total_entropy = sum(cell.entropy for cell in grid_data.cells)
            avg_entropy = total_entropy / total
            
            scanned = sum(1 for cell in grid_data.cells if cell.entropy < 30)
            scan_ratio = (scanned / total * 100) if total > 0 else 0
            
            return {
                'total': total,
                'avg': avg_entropy,
                'scanned': scanned,
                'scan_ratio': scan_ratio
            }
        except Exception:
            return None
    
    def _calculate_battery_stats(self):
        """计算电量统计信息"""
        try:
            if not self.server or not hasattr(self.server, 'get_all_battery_data'):
                return None
            
            battery_data = self.server.get_all_battery_data()
            if not battery_data:
                return None
            
            voltages = []
            for drone_name, battery_info in battery_data.items():
                voltage = battery_info.get('voltage', 4.2)
                voltages.append(voltage)
            
            if voltages:
                return {
                    'avg_voltage': sum(voltages) / len(voltages),
                    'min_voltage': min(voltages),
                    'max_voltage': max(voltages),
                    'drone_count': len(voltages)
                }
            return None
        except Exception:
            return None
    
    def update_data(self):
        """更新可视化数据"""
        try:
            if not self.server:
                return None, {}
            
            current_time = time.time()
            if hasattr(self, '_last_data_update') and current_time - self._last_data_update < 0.05:
                return getattr(self, '_cached_grid_data', None), getattr(self, '_cached_runtime_data', {})
            
            # 获取网格数据
            grid_data = None
            try:
                if hasattr(self.server, 'grid_data'):
                    grid_data = self.server.grid_data
            except Exception:
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
            except Exception:
                pass
            
            # 获取电量数据
            try:
                if hasattr(self.server, 'get_all_battery_data'):
                    battery_data = self.server.get_all_battery_data()
                    for drone_name, battery_info in battery_data.items():
                        if drone_name in runtime_data_dict:
                            runtime_data_dict[drone_name]['battery_voltage'] = battery_info.get('voltage', 4.2)
            except Exception:
                pass
            
            self._cached_grid_data = grid_data
            self._cached_runtime_data = runtime_data_dict
            self._last_data_update = current_time
            
            return grid_data, runtime_data_dict
        except Exception as e:
            print(f"更新可视化数据时出错: {str(e)}")
            return getattr(self, '_cached_grid_data', None), getattr(self, '_cached_runtime_data', {})
    
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
    
    def run(self):
        """主循环"""
        self.running = True
        
        try:
            if not self.pygame_initialized:
                pygame.init()
                self.pygame_initialized = True
                self.clock = pygame.time.Clock()
                pygame.font.init()
                
                try:
                    self.font = pygame.font.SysFont(['SimHei', 'Microsoft YaHei', 'Arial'], 18)
                    self.font_available = True
                except Exception:
                    self.font = pygame.font.Font(None, 18)
                    self.font_available = False
                
                self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
                pygame.display.set_caption("🎯 DQN训练实时可视化")
                
                print("=" * 60)
                print("✅ 训练可视化窗口已创建")
                print("💡 按ESC键关闭可视化窗口")
                print("=" * 60)
        except Exception as e:
            print(f"❌ Pygame初始化失败: {str(e)}")
            self.running = False
            return
        
        while self.running:
            try:
                self.handle_events()
                self.screen.fill(self.BLACK)
                
                # 更新数据
                grid_data, runtime_data_dict = self.update_data()
                
                # 绘制环境
                try:
                    self.draw_grid(grid_data)
                    self.draw_leader(runtime_data_dict)
                    self.draw_drones(runtime_data_dict)
                except Exception as e:
                    pass
                
                # 绘制UI面板
                try:
                    self.draw_env_info()
                    self.draw_training_info_panel()
                    self.draw_reward_curve()
                    self.draw_current_weights()
                except Exception as e:
                    pass
                
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