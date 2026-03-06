"""
系统性诊断：重置后熵值不降低的问题

检查内容：
1. Python端数据发送流程
2. Unity数据接收流程
3. 算法线程运行状态
4. 网格数据更新流程
"""

import sys
import os

print("=" * 60)
print("系统性诊断：重置后熵值不降低")
print("=" * 60)

# ============================================================================
# 1. 检查算法线程重置后的行为
# ============================================================================
print("\n[1] 检查算法线程重置后的行为")

with open('multirotor/AlgorithmServer.py', 'r', encoding='utf-8') as f:
    server_content = f.read()

# 检查算法线程是否会在重置后立即发送数据
issues = []

# 检查ready_event设置时机
if 'self.ready_event.set()' in server_content:
    lines = server_content.split('\n')
    ready_event_sets = []
    for i, line in enumerate(lines):
        if 'self.ready_event.set()' in line:
            # 获取上下文
            context_start = max(0, i-10)
            context_end = min(len(lines), i+5)
            context = '\n'.join(lines[context_start:context_end])
            ready_event_sets.append((i+1, context))

    print(f"  找到 {len(ready_event_sets)} 处ready_event.set()调用")

    # 检查reset_environment中的set调用
    for line_num, context in ready_event_sets:
        if 'reset_environment' in context or '[重置]' in context:
            print(f"  [OK] reset_environment中设置ready_event (行{line_num})")
            # 检查是否在start_simulation之后
            if 'send_start_simulation_command' in context:
                print("  [OK] ready_event在start_simulation之后设置")
            else:
                print("  [WARNING] ready_event可能在start_simulation之前设置")
                issues.append("ready_event时序问题")

# ============================================================================
# 2. 检查send_processed_data的调用时机
# ============================================================================
print("\n[2] 检查send_processed_data的调用时机")

# 检查算法线程中何时调用send_processed_data
with open('multirotor/AlgorithmServer.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

in_algorithm_thread = False
for i, line in enumerate(lines):
    if 'def _algorithm_thread' in line:
        in_algorithm_thread = True
        algorithm_thread_start = i
    elif in_algorithm_thread and 'def ' in line and 'def _algorithm_thread' not in line:
        in_algorithm_thread = False
    elif in_algorithm_thread and '_send_processed_data' in line:
        # 检查上下文
        context_start = max(0, i-15)
        context_end = min(len(lines), i+5)
        context = ''.join(lines[context_start:context_end])

        print(f"  找到_send_processed_data调用 (行{i+1})")

        # 检查是否在ready_event检查之后
        if 'ready_event' in context:
            print("  [OK] 在ready_event检查区域内")
        else:
            print("  [WARNING] 可能不在ready_event检查区域内")
            issues.append("send_processed_data时序问题")
        break

# ============================================================================
# 3. 检查resetting标志的使用
# ============================================================================
print("\n[3] 检查resetting标志的使用")

if 'if not self.running or self.resetting:' in server_content:
    print("  [OK] send_processed_data检查resetting标志")
else:
    print("  [ERROR] send_processed_data未检查resetting标志")
    issues.append("缺少resetting检查")

# 检查算法线程主循环是否检查resetting
in_algorithm_loop = False
for i, line in enumerate(lines):
    if 'while self.running:' in line and i > 1150 and i < 1250:
        in_algorithm_loop = True
        algorithm_loop_start = i
    elif in_algorithm_loop and ('def ' in line or 'class ' in line):
        in_algorithm_loop = False
    elif in_algorithm_loop and 'resetting' in line:
        print(f"  [OK] 算法循环检查resetting (行{i+1}): {line.strip()}")

# ============================================================================
# 4. 检查start_simulation的发送和等待
# ============================================================================
print("\n[4] 检查start_simulation的发送和等待")

# 查找send_start_simulation_command的调用
start_sim_calls = []
for i, line in enumerate(lines):
    if 'send_start_simulation_command()' in line:
        # 获取后续几行
        context = ''.join(lines[i:i+5])
        start_sim_calls.append((i+1, context))

print(f"  找到 {len(start_sim_calls)} 处send_start_simulation_command调用")

for line_num, context in start_sim_calls:
    print(f"\n  调用位置 (行{line_num}):")
    print(context)

    # 检查等待时间
    if 'sleep(2.0)' in context or 'sleep(2)' in context:
        print("  [OK] 等待2秒")
    elif 'sleep(0.5)' in context or 'sleep(0.5)' in context:
        print("  [WARNING] 只等待0.5秒 (太短)")
        issues.append("start_simulation等待时间太短")
    else:
        print("  [INFO] 未找到明确的sleep调用")

# ============================================================================
# 5. 检查Unity数据接收逻辑
# ============================================================================
print("\n[5] 检查Unity数据接收逻辑")

with open('multirotor/AlgorithmServer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 检查grid_data的更新
if 'update_from_dict' in content:
    print("  [OK] 使用update_from_dict更新grid_data")

    # 检查是否在锁保护下
    if 'with self.grid_lock:' in content:
        print("  [OK] 使用grid_lock保护")
    else:
        print("  [WARNING] 可能缺少锁保护")
        issues.append("grid_data更新缺少锁保护")

# ============================================================================
# 6. 检查熵值保护机制是否影响
# ============================================================================
print("\n[6] 检查熵值保护机制")

with open('multirotor/Algorithm/HexGridDataModel.py', 'r', encoding='utf-8') as f:
    grid_content = f.read()

if '_preserve_entropy' in grid_content:
    print("  [OK] 熵值保护机制存在")

    # 检查默认值
    if 'self._preserve_entropy = False' in grid_content:
        print("  [INFO] 默认值为False (不保护)")
    elif 'self._preserve_entropy = True' in grid_content:
        print("  [WARNING] 默认值为True (保护，Unity数据无法更新)")
        issues.append("熵值保护默认值可能导致Unity数据无法更新")

    # 检查update_from_dict中的逻辑
    if 'if not self._preserve_entropy:' in grid_content:
        print("  [OK] update_from_dict检查保护标志")
    else:
        print("  [WARNING] update_from_dict可能未检查保护标志")

# ============================================================================
# 7. 关键问题诊断
# ============================================================================
print("\n[7] 关键问题诊断")

print("\n  可能导致熵值不降低的原因:")
print("  A. Unity端问题:")
print("     - Unity未正确接收start_simulation指令")
print("     - Unity启动熵值收集失败")
print("     - Unity扫描逻辑有问题")

print("\n  B. Python端问题:")
print("     - 算法线程未发送runtime数据到Unity")
print("     - 熵值保护机制阻止了Unity数据更新")
print("     - 时序问题：数据发送太早或太晚")

print("\n  需要检查的日志:")
print("  1. Python日志:")
print("     - '[重置] 4/5 发送 start_simulation 指令'")
print("     - '[重置] 等待完成，Unity应该已启动熵值收集'")
print("     - '[drone_name] 首帧同步完成，开始决策循环'")

print("\n  2. Unity日志:")
print("     - 是否收到start_simulation指令")
print("     - 是否启动了熵值收集")
print("     - 是否收到runtime数据")
print("     - 是否执行了扫描")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "=" * 60)
print("诊断结果")
print("=" * 60)

if issues:
    print(f"\n发现 {len(issues)} 个潜在问题:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
else:
    print("\n未发现明显的Python端问题")

print("\n建议的排查步骤:")
print("  1. 检查训练日志，确认Python端流程正确")
print("  2. 检查Unity日志，确认Unity端收到指令并启动")
print("  3. 在Unity端添加调试日志，确认扫描逻辑执行")
print("  4. 检查网络通信，确认数据正确传输")

print("\n如果是Python端问题，最可能的原因:")
print("  - 熵值保护机制阻止了Unity数据更新")
print("  - 算法线程未正确发送数据")

print("\n如果是Unity端问题，最可能的原因:")
print("  - start_simulation指令未正确处理")
print("  - 扫描逻辑未启动")
print("  - 扫描结果未正确返回")

print("\n" + "=" * 60)
