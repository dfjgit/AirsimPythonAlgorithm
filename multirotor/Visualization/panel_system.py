"""
可视化面板系统 - 可插拔面板架构

提供统一的面板基类和面板管理器,支持动态注册、布局管理和数据更新
"""
import pygame
from typing import Dict, Any, Optional, List, Tuple
from abc import ABC, abstractmethod


class BasePanel(ABC):
    """
    面板基类 - 所有可视化面板的抽象基类
    
    每个面板负责:
    1. 绘制自己的内容
    2. 返回自己的尺寸需求
    3. 处理数据更新
    """
    
    def __init__(self, name: str, width: int = 350, height: int = 200):
        """
        初始化面板
        
        Args:
            name: 面板名称(唯一标识)
            width: 面板宽度
            height: 面板高度
        """
        self.name = name
        self.width = width
        self.height = height
        self.x = 0  # 由PanelManager设置
        self.y = 0  # 由PanelManager设置
        self.visible = True
        
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
        self.DARK_GRAY = (64, 64, 64)
        self.LIGHT_BLUE = (173, 216, 230)
        
        # 字体(延迟初始化)
        self._font = None
        self._small_font = None
        self._title_font = None
    
    def _init_fonts(self):
        """初始化字体(懒加载)"""
        if self._font is None:
            try:
                self._font = pygame.font.SysFont(['SimHei', 'Microsoft YaHei', 'Arial'], 14)
                self._small_font = pygame.font.SysFont(['SimHei', 'Microsoft YaHei', 'Arial'], 12)
                self._title_font = pygame.font.SysFont(['SimHei', 'Microsoft YaHei', 'Arial'], 16, bold=True)
            except:
                self._font = pygame.font.Font(None, 14)
                self._small_font = pygame.font.Font(None, 12)
                self._title_font = pygame.font.Font(None, 16)
    
    @abstractmethod
    def draw(self, screen: pygame.Surface, data: Dict[str, Any]):
        """
        绘制面板内容(子类必须实现)
        
        Args:
            screen: pygame屏幕对象
            data: 可视化数据字典
        """
        pass
    
    def draw_panel_background(self, screen: pygame.Surface, border_color: Tuple[int, int, int] = None, 
                             alpha: int = 200):
        """
        绘制面板背景和边框(公共方法)
        
        Args:
            screen: pygame屏幕对象
            border_color: 边框颜色,默认白色
            alpha: 背景透明度(0-255)
        """
        if border_color is None:
            border_color = self.WHITE
        
        # 半透明背景
        panel_rect = pygame.Rect(self.x, self.y, self.width, self.height)
        surface = pygame.Surface((self.width, self.height))
        surface.set_alpha(alpha)
        surface.fill(self.BLACK)
        screen.blit(surface, (self.x, self.y))
        
        # 边框
        pygame.draw.rect(screen, border_color, panel_rect, 2)
    
    def draw_title(self, screen: pygame.Surface, title: str, color: Tuple[int, int, int] = None) -> int:
        """
        绘制面板标题
        
        Args:
            screen: pygame屏幕对象
            title: 标题文本
            color: 标题颜色
            
        Returns:
            标题占用的垂直空间(像素)
        """
        self._init_fonts()
        if color is None:
            color = self.YELLOW
        
        text = self._title_font.render(title, True, color)
        screen.blit(text, (self.x + 10, self.y + 10))
        return 30  # 标题高度 + 间距
    
    def update_data(self, data: Dict[str, Any]):
        """
        更新面板数据(可选实现,用于预处理)
        
        Args:
            data: 新的数据字典
        """
        pass


class PanelManager:
    """
    面板管理器 - 负责面板的注册、布局和绘制顺序管理
    
    支持左右两侧布局，中间留给热力图
    """
    
    def __init__(self, screen_width: int, screen_height: int, 
                 left_panel_width: int = 380, right_panel_width: int = 380):
        """
        初始化面板管理器
        
        Args:
            screen_width: 屏幕宽度
            screen_height: 屏幕高度
            left_panel_width: 左侧面板区域宽度
            right_panel_width: 右侧面板区域宽度
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.left_panel_width = left_panel_width
        self.right_panel_width = right_panel_width
        
        # 面板注册表
        self.panels: Dict[str, BasePanel] = {}
        self.panel_order: List[str] = []  # 绘制顺序
        
        # 布局参数
        self.margin = 10  # 边距
        self.row_gap = 10  # 行间距
        
        # 布局区域定义
        self.layout_areas = {
            'top_left': (10, 10),
            'top_right': (screen_width - right_panel_width + 10, 10),
            'bottom_left': (10, screen_height - 250),
            'bottom_right': (screen_width - right_panel_width + 10, screen_height - 250)
        }
    
    def register_panel(self, panel: BasePanel, position: str = 'top_right'):
        """
        注册面板到管理器
        
        Args:
            panel: 面板实例
            position: 布局位置 ('top_left', 'top_right', 'bottom_left', 'bottom_right', 'auto')
        """
        if panel.name in self.panels:
            print(f"警告: 面板 '{panel.name}' 已存在,将被覆盖")
        
        self.panels[panel.name] = panel
        self.panel_order.append(panel.name)
        
        # 设置面板位置
        if position == 'auto':
            self._auto_layout()
        elif position in self.layout_areas:
            panel.x, panel.y = self.layout_areas[position]
    
    def unregister_panel(self, panel_name: str):
        """移除面板"""
        if panel_name in self.panels:
            del self.panels[panel_name]
            self.panel_order.remove(panel_name)
    
    def _auto_layout(self):
        """
        自动布局 - 左右两侧布局
        
        策略：
        1. 将面板分配到左右两侧
        2. 尽量均衡两侧的高度
        3. 中间留给热力图
        """
        if not self.panel_order:
            return
        
        # 获取所有可见面板
        visible_panels = [self.panels[name] for name in self.panel_order 
                         if name in self.panels and self.panels[name].visible]
        
        if not visible_panels:
            return
        
        # 使用贪心算法分配面板到左右两侧，尽量均衡高度
        left_panels = []
        right_panels = []
        left_height = 0
        right_height = 0
        
        # 按高度降序排序，先放置高面板
        sorted_panels = sorted(visible_panels, key=lambda p: p.height, reverse=True)
        
        for panel in sorted_panels:
            # 找到当前高度最小的一侧
            if left_height <= right_height:
                left_panels.append(panel)
                left_height += panel.height + self.row_gap
            else:
                right_panels.append(panel)
                right_height += panel.height + self.row_gap
        
        # 设置左侧面板位置
        current_y = self.margin
        for panel in left_panels:
            panel.width = min(panel.width, self.left_panel_width - 2 * self.margin)
            panel.x = self.margin
            panel.y = current_y
            current_y += panel.height + self.row_gap
        
        # 设置右侧面板位置
        current_y = self.margin
        right_start_x = self.screen_width - self.right_panel_width + self.margin
        for panel in right_panels:
            panel.width = min(panel.width, self.right_panel_width - 2 * self.margin)
            panel.x = right_start_x
            panel.y = current_y
            current_y += panel.height + self.row_gap
    
    def draw_all_panels(self, screen: pygame.Surface, data: Dict[str, Any]):
        """
        绘制所有已注册的面板
        
        Args:
            screen: pygame屏幕对象
            data: 可视化数据
        """
        for name in self.panel_order:
            panel = self.panels.get(name)
            if panel and panel.visible:
                try:
                    panel.draw(screen, data)
                except Exception as e:
                    print(f"绘制面板 '{name}' 时出错: {str(e)}")
    
    def update_all_panels(self, data: Dict[str, Any]):
        """
        更新所有面板数据
        
        Args:
            data: 新的数据字典
        """
        for panel in self.panels.values():
            try:
                panel.update_data(data)
            except Exception as e:
                print(f"更新面板 '{panel.name}' 数据时出错: {str(e)}")
    
    def set_panel_visibility(self, panel_name: str, visible: bool):
        """设置面板可见性"""
        if panel_name in self.panels:
            self.panels[panel_name].visible = visible
            self._auto_layout()
    
    def get_panel(self, panel_name: str) -> Optional[BasePanel]:
        """获取指定面板"""
        return self.panels.get(panel_name)
