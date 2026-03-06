"""
日志过滤脚本：只显示关键的网格更新日志

使用方法：
1. 运行训练
2. 在另一个终端运行此脚本
3. 实时查看网格更新日志
"""

import sys
import os
import time
from pathlib import Path

def filter_logs(log_file):
    """过滤日志文件，只显示网格更新相关的内容"""

    if not os.path.exists(log_file):
        print(f"[错误] 日志文件不存在: {log_file}")
        print(f"请先运行训练，然后等待日志文件生成")
        return

    print("=" * 60)
    print("实时监控网格更新日志")
    print("=" * 60)
    print("\n只显示网格更新相关的日志...")
    print("按 Ctrl+C 停止\n")

    # 获取文件初始大小
    file_size = os.path.getsize(log_file)

    try:
        while True:
            # 检查文件是否有新内容
            current_size = os.path.getsize(log_file)

            if current_size > file_size:
                # 读取新内容
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    f.seek(file_size)
                    new_lines = f.readlines()

                # 过滤并显示关键日志
                for line in new_lines:
                    line = line.strip()

                    # 只显示以下关键日志：
                    if any(keyword in line for keyword in [
                        '[网格更新]',
                        '平均熵值=',
                        '低熵格子=',
                        '[重置] 网格熵值已重置',
                        '[重置] 保持网格熵值',
                        '[重置] 4/5 发送 start_simulation',
                        '[重置] 等待完成',
                        '[可视化] 检测到重置',
                    ]):
                        print(line)

                    # 显示警告和错误
                    elif '[WARNING]' in line or '[ERROR]' in line or '🔴' in line:
                        print(line)

                # 更新文件大小
                file_size = current_size

            # 短暂休眠
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("停止监控")
        print("=" * 60)
    except Exception as e:
        print(f"\n[错误] {e}")


def main():
    """主函数"""

    # 查找最新的日志文件
    log_dir = Path("multirotor/DDPG_Weight/logs/ddpg_airsim")

    if not log_dir.exists():
        print(f"[错误] 日志目录不存在: {log_dir}")
        print("请先运行训练，等待日志目录生成")
        return

    # 查找最新的日志文件
    log_files = list(log_dir.glob("*.log"))

    if not log_files:
        print(f"[错误] 在 {log_dir} 中未找到日志文件")
        print("请先运行训练，等待日志文件生成")
        return

    # 按修改时间排序，获取最新的
    latest_log = max(log_files, key=lambda p: p.stat().st_mtime)

    print(f"找到最新日志文件: {latest_log}")
    print(f"修改时间: {time.ctime(latest_log.stat().st_mtime)}")

    # 开始过滤日志
    filter_logs(str(latest_log))


if __name__ == "__main__":
    main()
