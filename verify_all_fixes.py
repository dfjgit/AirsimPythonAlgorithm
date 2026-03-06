"""
验证所有修复是否正确应用

检查内容：
1. 可视化缓存修复
2. 熵值保护机制
3. reset_environment修复
4. send_processed_data修复
5. 默认配置修复
"""

import sys
import os

print("=" * 60)
print("最终验证：所有修复")
print("=" * 60)

all_ok = True

# ============================================================================
# 1. 验证可视化缓存修复
# ============================================================================
print("\n[1] 可视化缓存修复")

with open('multirotor/AlgorithmServer.py', 'r', encoding='utf-8') as f:
    server_content = f.read()

checks = [
    ('_last_reset_time检测', '_vis_snapshot_cache_time < self._last_reset_time' in server_content),
    ('清除缓存日志', '[可视化] 检测到重置，清除快照缓存' in server_content),
]

for name, result in checks:
    if result:
        print(f"  [OK] {name}")
    else:
        print(f"  [ERROR] {name} 未找到")
        all_ok = False

# ============================================================================
# 2. 验证熵值保护机制
# ============================================================================
print("\n[2] 熵值保护机制")

with open('multirotor/Algorithm/HexGridDataModel.py', 'r', encoding='utf-8') as f:
    grid_content = f.read()

checks = [
    ('保护标志', '_preserve_entropy' in grid_content),
    ('set_preserve_entropy方法', 'def set_preserve_entropy' in grid_content),
    ('update_from_dict使用保护', 'if not self._preserve_entropy:' in grid_content),
]

for name, result in checks:
    if result:
        print(f"  [OK] {name}")
    else:
        print(f"  [ERROR] {name} 未找到")
        all_ok = False

# ============================================================================
# 3. 验证reset_environment修复
# ============================================================================
print("\n[3] reset_environment修复")

checks = [
    ('根据reset_grid参数重置', 'if reset_grid:' in server_content and 'self.grid_data.reset_entropy()' in server_content),
    ('设置保护标志False', 'set_preserve_entropy(False)' in server_content),
    ('设置保护标志True', 'set_preserve_entropy(True)' in server_content),
    ('日志输出', '[重置] 网格熵值已重置为100' in server_content),
]

for name, result in checks:
    if result:
        print(f"  [OK] {name}")
    else:
        print(f"  [ERROR] {name} 未找到")
        all_ok = False

# ============================================================================
# 4. 验证send_processed_data修复
# ============================================================================
print("\n[4] send_processed_data修复")

checks = [
    ('检查resetting标志', 'if not self.running or self.resetting:' in server_content),
    ('返回避免发送脏数据', 'return  # 重置期间不发送数据' in server_content),
]

for name, result in checks:
    if result:
        print(f"  [OK] {name}")
    else:
        print(f"  [ERROR] {name} 未找到")
        all_ok = False

# ============================================================================
# 5. 验证默认配置修复
# ============================================================================
print("\n[5] 默认配置修复")

with open('multirotor/DDPG_Weight/train_with_airsim_improved.py', 'r', encoding='utf-8') as f:
    train_content = f.read()

checks = [
    ('默认值为True', '"reset_grid_entropy", True)' in train_content),
    ('注释已更新', '每次重置时重新扫描' in train_content),
    ('环境创建使用参数', 'reset_grid_entropy=reset_grid_entropy' in train_content),
]

for name, result in checks:
    if result:
        print(f"  [OK] {name}")
    else:
        print(f"  [ERROR] {name} 未找到")
        all_ok = False

# ============================================================================
# 6. 验证配置文件
# ============================================================================
print("\n[6] 配置文件")

config_path = 'multirotor/DDPG_Weight/configs/training_config_reset_scan.json'
if os.path.exists(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = f.read()
    if 'reset_grid_entropy' in config and 'true' in config.lower():
        print(f"  [OK] 配置文件存在且正确")
    else:
        print(f"  [ERROR] 配置文件内容不正确")
        all_ok = False
else:
    print(f"  [WARNING] 配置文件不存在（可选）")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "=" * 60)
if all_ok:
    print("✅ 所有修复已正确应用！")
    print("\n可以开始训练了：")
    print("  python train_with_airsim_improved.py")
    print("\n预期效果：")
    print("  1. 每个Episode重置时，熵值重置为100")
    print("  2. 重置后可视化界面立即更新")
    print("  3. 扫描功能正常工作，熵值能够降低")
    print("  4. 每个Episode的扫描格子数 > 0")
else:
    print("⚠️  部分修复未正确应用，请检查上述错误")
print("=" * 60)

sys.exit(0 if all_ok else 1)
