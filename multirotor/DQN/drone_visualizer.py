import pygame
import sys
import math
import os
from typing import List

# 添加项目根目录到Python路径，使导入正常工作
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# 使用绝对导入
from multirotor.Algorithm.Vector3 import Vector3
from multirotor.Algorithm.scanner_runtime_data import ScannerRuntimeData
from multirotor.Algorithm.HexGridDataModel import HexGridDataModel, HexCell

class DroneVisualizer:
    def __init__(self):
        # 初始化pygame
        pygame.init()
        
        # 设置中文字体支持
        pygame.font.init()
        self.font = pygame.font.SysFont(['SimHei', 'Microsoft YaHei', 'Arial'], 24)
        
        # 窗口设置
        self.SCREEN_WIDTH = 1000
        self.SCREEN_HEIGHT = 800
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("无人机环境可视化器")
        
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
        
        # 坐标系转换参数
        self.origin_x = self.SCREEN_WIDTH // 2
        self.origin_y = self.SCREEN_HEIGHT // 2
        self.scale = 20  # 1单位=20像素
        
        # 模拟数据
        self.drones = {}
        self.leader_position = Vector3(0, 0, 0)
        self.leader_scan_radius = 10.0
        self.grid_model = HexGridDataModel()
        self.create_sample_grid()
        
        # 算法参数（可调整）
        self.params = {
            "maxRepulsionDistance": 5.0,
            "minSafeDistance": 2.0,
            "revisitCooldown": 10.0,
            "entropyWeight": 0.7,
            "distanceWeight": 0.3
        }
        
        # UI元素状态
        self.selected_param = None
        self.dragging = False
        
        # 初始化无人机
        self.add_drone("UAV1", Vector3(-10, 0, 5))
        self.add_drone("UAV2", Vector3(10, 0, -5))
        
        # 主循环控制
        self.running = True
        self.clock = pygame.time.Clock()
        
    def add_drone(self, name, position):
        """添加无人机"""
        runtime_data = ScannerRuntimeData()
        runtime_data.uavname = name
        runtime_data.position = position
        runtime_data.finalMoveDir = Vector3(0, 0, 1)  # 默认向前
        self.drones[name] = runtime_data
        
    def create_sample_grid(self):
        """创建示例网格"""
        radius = 15
        cells = []
        
        # 创建六边形网格
        for q in range(-radius, radius + 1):
            r1 = max(-radius, -q - radius)
            r2 = min(radius, -q + radius)
            for r in range(r1, r2 + 1):
                # 计算六边形中心坐标
                x = q * 1.5
                y = (r + q/2) * math.sqrt(3)
                
                # 计算与原点的距离，用于生成熵值
                distance = math.sqrt(x*x + y*y)
                # 熵值随距离增加而增加（0-1之间）
                entropy = min(1.0, distance / (radius * 2))
                
                cells.append(HexCell(Vector3(x, y, 0), entropy))
        
        self.grid_model.cells = cells
        
    def world_to_screen(self, vector):
        """将世界坐标转换为屏幕坐标"""
        # 使用z轴作为x轴，x轴作为y轴进行2D投影
        screen_x = self.origin_x + vector.z * self.scale
        screen_y = self.origin_y - vector.x * self.scale  # 负号是因为pygame的y轴向下
        return int(screen_x), int(screen_y)
        
    def screen_to_world(self, screen_x, screen_y):
        """将屏幕坐标转换为世界坐标"""
        world_z = (screen_x - self.origin_x) / self.scale
        world_x = (self.origin_y - screen_y) / self.scale
        return Vector3(world_x, 0, world_z)
        
    def draw_grid(self):
        """绘制网格"""
        for cell in self.grid_model.cells:
            screen_x, screen_y = self.world_to_screen(cell.center)
            
            # 根据熵值决定颜色（熵值越低颜色越深）
            color_intensity = int(255 * (1 - min(1.0, cell.entropy)))
            color = (color_intensity, color_intensity, 255)
            
            # 绘制六边形
            self.draw_hexagon(screen_x, screen_y, color)
        
    def draw_hexagon(self, x, y, color):
        """绘制六边形"""
        size = self.scale * 0.4
        points = []
        for i in range(6):
            angle_deg = 60 * i - 30
            angle_rad = math.pi / 180 * angle_deg
            points.append((x + size * math.cos(angle_rad), y + size * math.sin(angle_rad)))
        pygame.draw.polygon(self.screen, color, points, 1)
        
    def draw_drones(self):
        """绘制无人机"""
        for name, drone in self.drones.items():
            screen_x, screen_y = self.world_to_screen(drone.position)
            
            # 绘制无人机主体
            pygame.draw.circle(self.screen, self.BLUE, (screen_x, screen_y), 10)
            
            # 绘制方向指示
            dir_x = screen_x + drone.finalMoveDir.z * 20
            dir_y = screen_y - drone.finalMoveDir.x * 20  # 注意坐标系转换
            pygame.draw.line(self.screen, self.RED, (screen_x, screen_y), (dir_x, dir_y), 3)
            
            # 绘制无人机名称
            text = self.font.render(name, True, self.WHITE)
            self.screen.blit(text, (screen_x + 15, screen_y - 10))
        
    def draw_leader(self):
        """绘制领导者位置和扫描范围"""
        screen_x, screen_y = self.world_to_screen(self.leader_position)
        
        # 绘制领导者位置
        pygame.draw.circle(self.screen, self.YELLOW, (screen_x, screen_y), 15)
        
        # 绘制扫描范围圆圈
        radius = self.leader_scan_radius * self.scale
        pygame.draw.circle(self.screen, self.YELLOW, (screen_x, screen_y), radius, 2)
        
        # 绘制领导者标签
        text = self.font.render("Leader", True, self.WHITE)
        self.screen.blit(text, (screen_x + 20, screen_y - 10))
        
    def draw_ui(self):
        """绘制UI控制面板"""
        # 绘制面板背景
        panel_rect = pygame.Rect(10, 10, 300, 200)
        pygame.draw.rect(self.screen, self.GRAY, panel_rect)
        pygame.draw.rect(self.screen, self.WHITE, panel_rect, 2)
        
        # 绘制标题
        title = self.font.render("算法参数调整", True, self.WHITE)
        self.screen.blit(title, (20, 15))
        
        # 绘制参数控制
        y_offset = 50
        for param_name, param_value in self.params.items():
            # 绘制参数名称
            text = self.font.render(f"{param_name}: {param_value:.2f}", True, self.WHITE)
            text_rect = text.get_rect(topleft=(20, y_offset))
            self.screen.blit(text, text_rect)
            
            # 绘制可点击区域
            param_rect = pygame.Rect(20, y_offset, 280, 30)
            if self.selected_param == param_name:
                pygame.draw.rect(self.screen, self.CYAN, param_rect, 2)
            
            y_offset += 40
        
        # 绘制操作说明
        instructions = [
            "- 点击参数进行调整（上/下箭头）",
            "- 鼠标拖拽可移动无人机",
            "- 'R'键重置所有位置",
            "- ESC键退出程序"
        ]
        
        instruction_y = self.SCREEN_HEIGHT - 120
        for instruction in instructions:
            text = self.font.render(instruction, True, self.LIGHT_GRAY)
            self.screen.blit(text, (20, instruction_y))
            instruction_y += 30
            
    def handle_events(self):
        """处理用户输入事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_r:
                    # 重置所有位置
                    self.reset_positions()
                elif self.selected_param is not None:
                    # 调整选中的参数
                    if event.key == pygame.K_UP:
                        self.params[self.selected_param] *= 1.1
                    elif event.key == pygame.K_DOWN:
                        self.params[self.selected_param] *= 0.9
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 左键点击
                    # 检查是否点击了UI参数
                    self.check_ui_click(event.pos)
                    # 检查是否点击了无人机
                    self.check_drone_click(event.pos)
            elif event.type == pygame.MOUSEMOTION:
                if self.dragging and self.dragging in self.drones:
                    # 拖拽无人机移动
                    world_pos = self.screen_to_world(event.pos[0], event.pos[1])
                    self.drones[self.dragging].position = world_pos
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # 左键释放
                    self.dragging = False
                    
    def check_ui_click(self, pos):
        """检查是否点击了UI参数"""
        x, y = pos
        if 10 <= x <= 310 and 10 <= y <= 210:
            # 在面板范围内
            y_offset = 50
            for param_name in self.params.keys():
                if y_offset - 10 <= y <= y_offset + 20:
                    self.selected_param = param_name
                    break
                y_offset += 40
        else:
            self.selected_param = None
            
    def check_drone_click(self, pos):
        """检查是否点击了无人机"""
        x, y = pos
        for name, drone in self.drones.items():
            drone_x, drone_y = self.world_to_screen(drone.position)
            distance = math.sqrt((x - drone_x)**2 + (y - drone_y)** 2)
            if distance <= 15:  # 无人机半径+5像素的容差
                self.dragging = name
                break
                
    def reset_positions(self):
        """重置所有位置"""
        self.add_drone("UAV1", Vector3(-10, 0, 5))
        self.add_drone("UAV2", Vector3(10, 0, -5))
        self.leader_position = Vector3(0, 0, 0)
        
    def run(self):
        """主循环"""
        while self.running:
            # 处理事件
            self.handle_events()
            
            # 清屏
            self.screen.fill(self.BLACK)
            
            # 绘制元素
            self.draw_grid()
            self.draw_leader()
            self.draw_drones()
            self.draw_ui()
            
            # 更新屏幕
            pygame.display.flip()
            
            # 控制帧率
            self.clock.tick(60)
        
        # 退出程序
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    visualizer = DroneVisualizer()
    visualizer.run()