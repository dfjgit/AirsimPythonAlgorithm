"""
强制显示pygame窗口测试 - 使用最大化模式
"""
import sys
import os
import time

# 设置SDL视频驱动为Windows（必须在导入pygame之前）
os.environ['SDL_VIDEODRIVER'] = 'windows'
os.environ['SDL_VIDEO_CENTERED'] = '1'  # 居中显示

# 添加路径
sys.path.insert(0, 'multirotor')

import pygame

print("=" * 60)
print("强制显示窗口测试")
print("=" * 60)
print("\n正在创建一个简单的pygame窗口...")

# 初始化pygame
pygame.init()

# 创建窗口（较小尺寸，便于看到）
screen = pygame.display.set_mode((800, 600), pygame.SHOWN | pygame.RESIZABLE)
pygame.display.set_caption('🎯 测试窗口 - 请确认能看到此窗口')

# 强制置顶并激活
try:
    import ctypes
    time.sleep(0.2)
    hwnd = pygame.display.get_wm_info()['window']
    # 显示窗口
    ctypes.windll.user32.ShowWindow(hwnd, 9)  # SW_RESTORE
    # 激活窗口
    ctypes.windll.user32.SetForegroundWindow(hwnd)
    # 闪烁任务栏图标引起注意
    ctypes.windll.user32.FlashWindow(hwnd, True)
    time.sleep(0.5)
    ctypes.windll.user32.FlashWindow(hwnd, False)
    print("✓ 窗口已创建并尝试置顶")
except Exception as e:
    print(f"警告: 窗口置顶失败 - {e}")

print("\n窗口标题: '🎯 测试窗口 - 请确认能看到此窗口'")
print("窗口大小: 800x600")
print("窗口位置: 屏幕居中")
print("\n如果能看到窗口，说明pygame工作正常")
print("窗口将显示一个绿色方块")
print("\n按ESC键或关闭窗口退出...")

# 主循环
clock = pygame.time.Clock()
running = True
frame_count = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
    
    # 绘制
    screen.fill((0, 0, 0))  # 黑色背景
    
    # 绘制一个跳动的绿色方块
    size = 100 + 50 * (frame_count % 60) / 60
    pygame.draw.rect(screen, (0, 255, 0), (350, 250, size, size))
    
    # 绘制文字
    font = pygame.font.Font(None, 36)
    text = font.render("Can you see this window?", True, (255, 255, 255))
    text_rect = text.get_rect(center=(400, 150))
    screen.blit(text, text_rect)
    
    text2 = font.render(f"Frame: {frame_count}", True, (255, 255, 0))
    text2_rect = text2.get_rect(center=(400, 450))
    screen.blit(text2, text2_rect)
    
    pygame.display.flip()
    clock.tick(60)
    frame_count += 1

pygame.quit()
print("\n窗口已关闭")
