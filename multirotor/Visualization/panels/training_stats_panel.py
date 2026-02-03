"""训练统计面板 - 显示Episode、步数、奖励等训练指标"""
import sys
import os
import pygame
from typing import Dict, Any
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from multirotor.Visualization.panel_system import BasePanel


class TrainingStatsPanel(BasePanel):
    """训练统计面板"""
    
    def __init__(self, width: int = 370, height: int = 280):
        super().__init__("training_stats", width, height)
        self.training_start_time = time.time()
    
    def draw(self, screen: pygame.Surface, data: Dict[str, Any]):
        """绘制训练统计"""
        self._init_fonts()
        
        # 背景和边框
        self.draw_panel_background(screen, border_color=self.YELLOW)
        
        # 标题
        y_offset = self.draw_title(screen, "🎯 训练状态", self.YELLOW)
        
        text_x = self.x + 15
        y = self.y + y_offset
        
        # 获取训练数据
        episode_count = data.get('episode_count', 0)
        total_steps = data.get('total_steps', 0)
        current_episode_steps = data.get('current_episode_steps', 0)
        current_episode_reward = data.get('current_episode_reward', 0.0)
        
        # 创建大字体用于步骤计数器
        if not hasattr(self, '_big_font'):
            try:
                self._big_font = pygame.font.SysFont(['SimHei', 'Microsoft YaHei', 'Arial'], 24, bold=True)
            except:
                self._big_font = pygame.font.Font(None, 24)
        
        # 醒目的步骤计数器
        step_text = self._big_font.render(f"步数: {total_steps}", True, self.CYAN)
        screen.blit(step_text, (text_x, y))
        y += 35
        
        # Episode信息
        text = self._font.render(f"Episode: {episode_count}", True, self.WHITE)
        screen.blit(text, (text_x, y))
        y += 20
        
        text = self._font.render(f"当前Episode步数: {current_episode_steps}", True, self.CYAN)
        screen.blit(text, (text_x, y))
        y += 20
        
        text = self._font.render(f"当前Episode奖励: {current_episode_reward:.2f}", True, self.CYAN)
        screen.blit(text, (text_x, y))
        y += 25
        
        # 训练速率
        steps_per_sec = data.get('steps_per_sec', 0.0)
        rate_text = self._font.render(f"速率: {steps_per_sec:.2f} steps/s", True, self.GREEN)
        screen.blit(rate_text, (text_x, y))
        y += 20
        
        # 已用时间
        elapsed_time = time.time() - self.training_start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        time_text = self._font.render(f"已用时间: {hours:02d}:{minutes:02d}:{seconds:02d}", 
                                      True, self.WHITE)
        screen.blit(time_text, (text_x, y))
        y += 25
        
        # 统计信息
        avg_reward = data.get('avg_reward', 0.0)
        max_reward = data.get('max_reward', 0.0)
        min_reward = data.get('min_reward', 0.0)
        
        if avg_reward != 0:
            pygame.draw.line(screen, self.GRAY, (self.x + 10, y), 
                           (self.x + self.width - 10, y), 1)
            y += 10
            
            text = self._font.render(f"平均奖励: {avg_reward:.2f}", True, self.GREEN)
            screen.blit(text, (text_x, y))
            y += 18
            
            text = self._font.render(f"最佳奖励: {max_reward:.2f}", True, self.GREEN)
            screen.blit(text, (text_x, y))
            y += 18
            
            text = self._font.render(f"最差奖励: {min_reward:.2f}", True, self.RED)
            screen.blit(text, (text_x, y))
