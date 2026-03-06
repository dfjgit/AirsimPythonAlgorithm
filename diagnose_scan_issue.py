"""
诊断扫描功能和数据更新问题

检查内容：
1. 算法线程是否正常运行
2. Runtime数据是否发送到Unity
3. Grid数据是否从Unity接收
4. 熵值是否正常更新
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("扫描功能诊断")
print("=" * 60)

# 检查AlgorithmServer代码
print("\n[1] 检查算法线程运行逻辑...")

with open('multirotor/AlgorithmServer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 检查重置期间的发送逻辑
if "if not self.running:" in content and "send_runtime" in content:
    print("  [OK] 找到runtime数据发送逻辑")
    # 检查是否有重置期间停止发送的逻辑
    if "resetting" in content and "send_runtime" in content:
        # 查找包含resetting的发送逻辑
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'send_runtime' in line and i > 0:
                # 检查前面几行是否有resetting相关的逻辑
                context = '\n'.join(lines[max(0, i-5):i+5])
                if 'resetting' in context:
                    print("  [!] 发现重置相关的发送逻辑:")
                    print(context)
                    break
    else:
        print("  [INFO] 未发现重置期间停止发送的逻辑")
else:
    print("  [ERROR] 未找到runtime数据发送逻辑")

# 检查熵值保护机制
print("\n[2] 检查熵值保护机制...")

with open('multirotor/Algorithm/HexGridDataModel.py', 'r', encoding='utf-8') as f:
    grid_content = f.read()

if '_preserve_entropy' in grid_content:
    print("  [OK] 熵值保护机制已添加")
    if 'set_preserve_entropy' in grid_content:
        print("  [OK] set_preserve_entropy方法存在")
        # 检查update_from_dict是否使用了保护标志
        if 'if not self._preserve_entropy:' in grid_content:
            print("  [OK] update_from_dict使用保护标志")
        else:
            print("  [WARNING] update_from_dict可能未使用保护标志")
    else:
        print("  [ERROR] set_preserve_entropy方法不存在")
else:
    print("  [WARNING] 熵值保护机制未添加")

# 检查reset_environment是否设置保护标志
print("\n[3] 检查reset_environment中的熵值保护...")

if 'set_preserve_entropy' in content:
    print("  [OK] reset_environment调用了set_preserve_entropy")
    # 检查是否根据reset_grid参数设置
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'set_preserve_entropy' in line:
            # 显示上下文
            context = '\n'.join(lines[max(0, i-3):i+3])
            print("  找到的代码:")
            print(context)
            break
else:
    print("  [WARNING] reset_environment未调用set_preserve_entropy")

# 检查可视化缓存逻辑
print("\n[4] 检查可视化缓存逻辑...")

if '_last_reset_time' in content and '_vis_snapshot_cache' in content:
    print("  [OK] 可视化缓存使用重置时间戳")
    # 检查是否比较时间戳
    if '_vis_snapshot_cache_time < self._last_reset_time' in content:
        print("  [OK] 检测到缓存时间戳比较逻辑")
    else:
        print("  [WARNING] 可能缺少时间戳比较逻辑")
else:
    print("  [WARNING] 可视化缓存可能未使用重置时间戳")

# 总结
print("\n" + "=" * 60)
print("诊断总结")
print("=" * 60)

print("\n可能的问题:")
print("1. 如果熵值保护机制启用，Unity的扫描结果无法更新Python端的熵值")
print("2. 解决方案：在reset_environment中根据reset_grid参数正确设置保护标志")
print("   - reset_grid=True: 重置熵值，允许Unity更新（完全重新扫描）")
print("   - reset_grid=False: 保持熵值，阻止Unity更新（累积扫描进度）")

print("\n建议检查:")
print("1. 训练日志中是否有'[重置] 网格熵值已重置'或'[重置] 保持网格熵值'")
print("2. Unity日志中是否正确接收了runtime数据")
print("3. Unity日志中是否执行了扫描并更新了熵值")

print("\n下一步:")
print("1. 如果用户想要每次重置后重新扫描，使用reset_grid=True")
print("2. 如果用户想要累积扫描进度，使用reset_grid=False")
print("3. 检查Unity端的扫描功能是否正常工作")

print("\n" + "=" * 60)
