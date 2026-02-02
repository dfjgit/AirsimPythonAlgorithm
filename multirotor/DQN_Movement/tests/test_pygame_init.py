"""
测试pygame初始化是否会阻塞
"""
import sys
import os
import time

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

print("=" * 60)
print("测试 1: 直接初始化pygame")
print("=" * 60)

try:
    import pygame
    print("✓ pygame导入成功")
    
    print("\n开始初始化pygame...")
    start_time = time.time()
    
    pygame.init()
    print(f"✓ pygame.init() 完成 (耗时: {time.time() - start_time:.2f}秒)")
    
    pygame.font.init()
    print(f"✓ pygame.font.init() 完成")
    
    print("\n创建显示窗口...")
    start_time = time.time()
    screen = pygame.display.set_mode((800, 600))
    print(f"✓ pygame.display.set_mode() 完成 (耗时: {time.time() - start_time:.2f}秒)")
    
    pygame.display.set_caption("Test Window")
    print(f"✓ pygame.display.set_caption() 完成")
    
    print("\n显示窗口并等待3秒...")
    for i in range(3):
        pygame.event.pump()  # 处理事件队列
        time.sleep(1)
        print(f"  {i+1}秒...")
    
    print("\n关闭pygame...")
    pygame.quit()
    print("✓ pygame.quit() 完成")
    
    print("\n" + "=" * 60)
    print("✅ 测试成功！pygame可以正常初始化")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ 测试失败: {str(e)}")
    import traceback
    traceback.print_exc()
