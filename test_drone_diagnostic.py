#!/usr/bin/env python3
"""
无人机徘徊问题诊断测试脚本
运行此脚本可以详细查看每台无人机的状态差异
"""

import sys
import os
import time
import threading
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from multirotor.AlgorithmServer import MultiDroneAlgorithmServer
from multirotor.Algorithm.scanner_config_data import ScannerConfigData
from multirotor.Algorithm.scanner_algorithm import ScannerAlgorithm
from multirotor.Algorithm.hex_grid_data_model import HexGridDataModel
from multirotor.Algorithm.scanner_runtime_data import ScannerRuntimeData, Vector3


class DroneDiagnosticLogger:
    """无人机诊断日志记录器"""

    def __init__(self, server: MultiDroneAlgorithmServer):
        self.server = server
        self.running = True
        self.log_data = {name: [] for name in server.drone_names}
        self.diagnostic_thread = None

    def start(self):
        """启动诊断日志记录"""
        self.diagnostic_thread = threading.Thread(target=self._log_loop, daemon=True)
        self.diagnostic_thread.start()
        print("✅ 诊断日志记录器已启动")

    def stop(self):
        """停止诊断日志记录"""
        self.running = False
        if self.diagnostic_thread:
            self.diagnostic_thread.join(timeout=2)
        print("🛑 诊断日志记录器已停止")

    def _log_loop(self):
        """诊断日志循环"""
        log_count = 0
        while self.running:
            log_count += 1
            print(f"\n{'=' * 80}")
            print(f"🔍 诊断日志 #{log_count} - {time.strftime('%H:%M:%S')}")
            print(f"{'=' * 80}")

            for drone_name in self.server.drone_names:
                self._log_drone_status(drone_name, log_count)

            # 对比分析
            self._compare_drones()

            time.sleep(5)  # 每5秒记录一次

    def _log_drone_status(self, drone_name: str, log_count: int):
        """记录单个无人机的详细状态"""
        print(f"\n📡 {drone_name} 状态:")
        print("-" * 40)

        with self.server.data_lock:
            runtime_data = self.server.unity_runtime_data.get(drone_name)
            if not runtime_data:
                print(f"  ❌ 无运行时数据")
                return

            # 1. 位置信息
            pos = runtime_data.position
            if pos:
                print(f"  📍 位置: ({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f})")
                print(f"     高度: {pos.y:.2f}m {'✅' if pos.y > 1.0 else '⚠️ 过低!'}")
            else:
                print(f"  ❌ 位置为None")

            # 2. 方向信息
            final_dir = runtime_data.finalMoveDir
            if final_dir:
                mag = final_dir.magnitude()
                print(
                    f"  🧭 最终方向: ({final_dir.x:.2f}, {final_dir.y:.2f}, {final_dir.z:.2f})"
                )
                print(f"     大小: {mag:.3f} {'✅' if mag > 0.1 else '⚠️ 过小!'}")
            else:
                print(f"  ❌ 最终方向为None")

            # 3. 算法中间结果
            score_dir = runtime_data.scoreDir
            collide_dir = runtime_data.collideDir
            leader_dir = runtime_data.leaderRangeDir

            print(f"  📊 算法分量:")
            if score_dir:
                print(
                    f"     熵方向: ({score_dir.x:.2f}, {score_dir.y:.2f}, {score_dir.z:.2f})"
                )
            if collide_dir:
                print(
                    f"     排斥力: ({collide_dir.x:.2f}, {collide_dir.y:.2f}, {collide_dir.z:.2f})"
                )
            if leader_dir:
                print(
                    f"     Leader: ({leader_dir.x:.2f}, {leader_dir.y:.2f}, {leader_dir.z:.2f})"
                )

            # 4. Leader相关信息
            leader_pos = runtime_data.leader_position
            if leader_pos and pos:
                dist_to_leader = (pos - leader_pos).magnitude()
                scan_radius = runtime_data.leader_scan_radius
                print(f"  🎯 Leader距离: {dist_to_leader:.2f}m / {scan_radius:.2f}m")
                if dist_to_leader > scan_radius:
                    print(f"     ⚠️ 出圈!")

            # 5. 算法权重
            algo = self.server.algorithms.get(drone_name)
            if algo:
                weights = algo.get_current_coefficients()
                print(f"  ⚖️  当前权重:")
                print(f"     排斥力: {weights.get('repulsionCoefficient', 'N/A'):.2f}")
                print(f"     熵值: {weights.get('entropyCoefficient', 'N/A'):.2f}")
                print(f"     距离: {weights.get('distanceCoefficient', 'N/A'):.2f}")

            # 6. 其他扫描器位置（排斥力计算用）
            other_scanners = runtime_data.otherScannerPositions
            print(f"  👥 其他无人机位置: {len(other_scanners)}个")
            for i, other_pos in enumerate(other_scanners[:3]):  # 只显示前3个
                if pos and other_pos:
                    dist = (pos - other_pos).magnitude()
                    print(f"     无人机{i + 1}: 距离{dist:.2f}m")

    def _compare_drones(self):
        """对比不同无人机的状态差异"""
        print(f"\n📊 无人机对比分析:")
        print("-" * 40)

        with self.server.data_lock:
            positions = {}
            heights = {}
            directions = {}

            for drone_name in self.server.drone_names:
                runtime_data = self.server.unity_runtime_data.get(drone_name)
                if runtime_data and runtime_data.position:
                    positions[drone_name] = runtime_data.position
                    heights[drone_name] = runtime_data.position.y
                    if runtime_data.finalMoveDir:
                        directions[drone_name] = runtime_data.finalMoveDir.magnitude()

            # 高度对比
            if heights:
                print(f"  高度对比:")
                for name, h in heights.items():
                    status = "✅" if h > 1.0 else "⚠️ 过低"
                    print(f"    {name}: {h:.2f}m {status}")

            # 方向大小对比
            if directions:
                print(f"  方向大小对比:")
                for name, d in directions.items():
                    status = "✅" if d > 0.1 else "⚠️ 过小"
                    print(f"    {name}: {d:.3f} {status}")

            # 位置差异
            if len(positions) >= 2:
                print(f"  无人机间距:")
                names = list(positions.keys())
                for i in range(len(names)):
                    for j in range(i + 1, len(names)):
                        dist = (positions[names[i]] - positions[names[j]]).magnitude()
                        print(f"    {names[i]} ↔ {names[j]}: {dist:.2f}m")


def run_diagnostic_test():
    """运行诊断测试"""
    print("🔧 无人机徘徊问题诊断测试")
    print("=" * 80)

    # 1. 创建服务器
    print("\n[1/3] 创建AlgorithmServer...")
    drone_names = ["UAV1", "UAV2", "UAV3"]
    server = MultiDroneAlgorithmServer(
        drone_names=drone_names,
        use_learned_weights=False,
        enable_visualization=False,
        control_mode="apf",
    )

    # 2. 启动服务器
    print("\n[2/3] 启动服务器...")
    if not server.start():
        print("❌ 服务器启动失败")
        return
    print("✅ 服务器启动成功")

    # 3. 启动任务
    print("\n[3/3] 启动无人机任务...")
    if not server.start_mission():
        print("❌ 任务启动失败")
        return
    print("✅ 任务启动成功")

    # 4. 启动诊断日志
    print("\n" + "=" * 80)
    print("开始记录诊断日志（每5秒记录一次，按Ctrl+C停止）...")
    print("=" * 80 + "\n")

    logger = DroneDiagnosticLogger(server)
    logger.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 收到停止信号")
        logger.stop()
        server.stop()
        print("✅ 测试完成")


if __name__ == "__main__":
    run_diagnostic_test()
