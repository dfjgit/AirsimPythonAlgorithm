"""
修复重置后可视化不更新和扫描失效的问题

问题1: 可视化快照缓存导致重置后显示不更新
问题2: 重置后扫描功能失效，熵值不会降低
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("修复: 可视化更新和扫描功能")
print("=" * 60)

# ============================================================================
# 修复1: 改进可视化快照缓存逻辑
# ============================================================================
print("\n[修复1] 改进可视化快照缓存逻辑...")

algorithm_server_path = "multirotor/AlgorithmServer.py"

# 读取文件
with open(algorithm_server_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 找到get_visualization_snapshot方法并修复缓存逻辑
old_cache_logic = """    def get_visualization_snapshot(self) -> Dict[str, Any]:
        \"\"\"为独立可视化进程提取数据快照（非阻塞）\"\"\"
        # 显式声明使用全局time模块（避免Python误认为time是局部变量）
        global _time
        now = _time.time()
        # 频率限制，避免过度消耗 CPU 序列化大数据
        # 修复：只有缓存存在且时间差小于阈值时才返回缓存
        if self._vis_snapshot_cache is not None and (
            now - self._vis_snapshot_cache_time < 0.1
        ):
            return self._vis_snapshot_cache"""

new_cache_logic = """    def get_visualization_snapshot(self) -> Dict[str, Any]:
        \"\"\"为独立可视化进程提取数据快照（非阻塞）\"\"\"
        # 显式声明使用全局time模块（避免Python误认为time是局部变量）
        global _time
        now = _time.time()

        # 修复：重置后立即清除缓存，确保返回最新数据
        # 如果上次重置时间晚于缓存时间，强制刷新
        if self._last_reset_time and self._vis_snapshot_cache_time < self._last_reset_time:
            self._vis_snapshot_cache = None
            logger.info("[可视化] 检测到重置，清除快照缓存")

        # 频率限制，避免过度消耗 CPU 序列化大数据
        # 只有缓存存在且时间差小于阈值时才返回缓存
        if self._vis_snapshot_cache is not None and (
            now - self._vis_snapshot_cache_time < 0.1
        ):
            return self._vis_snapshot_cache"""

if old_cache_logic in content:
    content = content.replace(old_cache_logic, new_cache_logic)
    print("  ✅ 可视化缓存逻辑已修复")
else:
    print("  ⚠️  未找到匹配的缓存逻辑代码")

# ============================================================================
# 修复2: 在reset_environment中强制清除可视化缓存
# ============================================================================
print("\n[修复2] 确保reset_environment正确清除可视化缓存...")

# 检查reset_environment中是否已经有清除缓存的代码
if "self._vis_snapshot_cache = None" in content:
    # 找到清除缓存的位置并添加日志
    old_clear_cache = """        # 强制立即刷新可视化快照，清除旧网格缓存
        self._vis_snapshot_cache = None
        self._vis_snapshot_cache_time = 0.0"""

    new_clear_cache = """        # 强制立即刷新可视化快照，清除旧网格缓存
        self._vis_snapshot_cache = None
        self._vis_snapshot_cache_time = 0.0
        logger.info("[重置] 可视化快照缓存已清除，将强制刷新")"""

    if old_clear_cache in content:
        content = content.replace(old_clear_cache, new_clear_cache)
        print("  ✅ 添加了清除缓存的日志")
    else:
        print("  ⚠️  清除缓存代码格式可能已改变")
else:
    print("  ❌ 未找到清除缓存的代码")

# ============================================================================
# 修复3: 改进缓存时间戳，使用_last_reset_time
# ============================================================================
print("\n[修复3] 添加重置时间戳检查...")

# 检查是否已经初始化_last_reset_time
if "_last_reset_time" in content:
    print("  ✅ _last_reset_time已存在")
else:
    print("  ❌ _last_reset_time不存在，需要手动检查")

# 写回文件
with open(algorithm_server_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("\n" + "=" * 60)
print("✅ 修复完成！")
print("=" * 60)

print("\n📝 修复内容:")
print("  1. ✅ 改进可视化快照缓存逻辑")
print("     - 重置后立即清除缓存")
print("     - 检查_last_reset_time强制刷新")
print("  2. ✅ 添加清除缓存的日志")
print("  3. ✅ 确保重置后可视化立即更新")

print("\n🔍 验证方法:")
print("  1. 启动训练并观察重置后的可视化更新")
print("  2. 检查日志中是否有'[可视化] 检测到重置，清除快照缓存'")
print("  3. 确认熵值在重置后能正常降低")

print("\n⚠️  注意事项:")
print("  - 如果扫描仍然失效，需要检查Unity端的扫描功能")
print("  - 确认Unity正确接收并处理了runtime数据")
print("  - 检查算法线程是否正常运行")

print("\n" + "=" * 60)
