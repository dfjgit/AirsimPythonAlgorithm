"""
无人机诊断日志记录器
自动记录详细的诊断信息到文件，用于分析多机协同问题
"""

import logging
import os
from datetime import datetime
from pathlib import Path


class DroneDiagnosticLogger:
    """
    无人机诊断日志记录器
    独立记录每台无人机的详细状态到诊断日志文件
    """

    def __init__(self, log_dir: str = None):
        """
        初始化诊断日志记录器

        Args:
            log_dir: 日志目录，默认为项目根目录下的 logs/diagnostic
        """
        if log_dir is None:
            project_root = Path(__file__).parent.parent.parent
            log_dir = project_root / "logs" / "diagnostic"

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 创建带时间戳的日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"drone_diagnostic_{timestamp}.log"

        # 配置日志记录器
        self.logger = logging.getLogger("DroneDiagnostic")
        self.logger.setLevel(logging.DEBUG)

        # 文件处理器
        file_handler = logging.FileHandler(self.log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
        )
        file_handler.setFormatter(formatter)

        # 清除旧的处理器
        self.logger.handlers.clear()
        self.logger.addHandler(file_handler)

        # 控制台处理器（只显示重要信息）
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(formatter)
        # self.logger.addHandler(console_handler)  # 禁用控制台输出

        self.logger.info(f"=" * 80)
        self.logger.info(f"无人机诊断日志记录器启动")
        self.logger.info(f"日志文件: {self.log_file}")
        self.logger.info(f"=" * 80)

        # 状态记录
        self.frame_count = 0
        self.last_positions = {}
        self.last_move_commands = {}

    def log_drone_status(self, drone_name: str, server, algorithm, runtime_data):
        """
        记录单个无人机的详细状态

        Args:
            drone_name: 无人机名称
            server: AlgorithmServer实例
            algorithm: ScannerAlgorithm实例
            runtime_data: ScannerRuntimeData实例
        """
        self.frame_count += 1

        try:
            # 获取位置信息
            pos = runtime_data.position if runtime_data else None

            # 获取方向信息
            final_dir = runtime_data.finalMoveDir if runtime_data else None

            # 获取算法中间结果
            score_dir = runtime_data.scoreDir if runtime_data else None
            collide_dir = runtime_data.collideDir if runtime_data else None
            leader_dir = runtime_data.leaderRangeDir if runtime_data else None

            # 获取Leader信息
            leader_pos = runtime_data.leader_position if runtime_data else None
            leader_radius = runtime_data.leader_scan_radius if runtime_data else 0

            # 获取其他无人机位置
            other_positions = runtime_data.otherScannerPositions if runtime_data else []

            # 获取权重信息
            weights = algorithm.get_current_coefficients() if algorithm else {}

            # 构建日志消息
            log_msg = f"\n{'=' * 60}\n"
            log_msg += f"帧 #{self.frame_count} | 无人机: {drone_name} | 时间: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}\n"
            log_msg += f"{'=' * 60}\n"

            # 1. 位置信息
            if pos:
                log_msg += f"📍 位置: ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})\n"
                log_msg += (
                    f"   高度: {pos.y:.2f}m {'✅' if pos.y > 1.0 else '⚠️ 过低!'}\n"
                )

                # 计算移动距离
                if drone_name in self.last_positions:
                    last_pos = self.last_positions[drone_name]
                    dist = (
                        (pos.x - last_pos["x"]) ** 2
                        + (pos.y - last_pos["y"]) ** 2
                        + (pos.z - last_pos["z"]) ** 2
                    ) ** 0.5
                    log_msg += f"   距离上次: {dist:.3f}m\n"

                self.last_positions[drone_name] = {"x": pos.x, "y": pos.y, "z": pos.z}
            else:
                log_msg += f"❌ 位置: None\n"

            # 2. 方向信息
            if final_dir:
                mag = final_dir.magnitude()
                log_msg += f"🧭 最终方向: ({final_dir.x:.3f}, {final_dir.y:.3f}, {final_dir.z:.3f})\n"
                log_msg += f"   大小: {mag:.4f} {'✅' if mag > 0.1 else '⚠️ 过小!'}\n"
            else:
                log_msg += f"❌ 最终方向: None\n"

            # 3. 算法分量
            log_msg += f"📊 算法分量:\n"
            if score_dir:
                log_msg += f"   熵方向: ({score_dir.x:.3f}, {score_dir.y:.3f}, {score_dir.z:.3f})\n"
            else:
                log_msg += f"   熵方向: None\n"

            if collide_dir:
                log_msg += f"   排斥力: ({collide_dir.x:.3f}, {collide_dir.y:.3f}, {collide_dir.z:.3f})\n"
            else:
                log_msg += f"   排斥力: None\n"

            if leader_dir:
                log_msg += f"   Leader: ({leader_dir.x:.3f}, {leader_dir.y:.3f}, {leader_dir.z:.3f})\n"
            else:
                log_msg += f"   Leader: None\n"

            # 4. Leader相关信息
            if leader_pos and pos:
                dist_to_leader = (
                    (pos.x - leader_pos.x) ** 2
                    + (pos.y - leader_pos.y) ** 2
                    + (pos.z - leader_pos.z) ** 2
                ) ** 0.5
                log_msg += (
                    f"🎯 Leader距离: {dist_to_leader:.2f}m / {leader_radius:.2f}m\n"
                )
                if dist_to_leader > leader_radius:
                    log_msg += (
                        f"   ⚠️ 出圈! 超出 {dist_to_leader - leader_radius:.2f}m\n"
                    )

            # 5. 其他无人机信息
            log_msg += f"👥 其他无人机: {len(other_positions)}个\n"
            for i, other_pos in enumerate(other_positions):
                if pos and other_pos:
                    dist = (
                        (pos.x - other_pos.x) ** 2
                        + (pos.y - other_pos.y) ** 2
                        + (pos.z - other_pos.z) ** 2
                    ) ** 0.5
                    log_msg += f"   无人机{i + 1}: ({other_pos.x:.2f}, {other_pos.y:.2f}, {other_pos.z:.2f}) 距离{dist:.2f}m\n"

            # 6. 权重信息
            log_msg += f"⚖️  当前权重:\n"
            log_msg += f"   排斥力: {weights.get('repulsionCoefficient', 'N/A'):.3f}\n"
            log_msg += f"   熵值: {weights.get('entropyCoefficient', 'N/A'):.3f}\n"
            log_msg += f"   距离: {weights.get('distanceCoefficient', 'N/A'):.3f}\n"
            log_msg += (
                f"   Leader: {weights.get('leaderRangeCoefficient', 'N/A'):.3f}\n"
            )
            log_msg += f"   方向保持: {weights.get('directionRetentionCoefficient', 'N/A'):.3f}\n"

            # 7. 移动指令（如果记录了）
            if drone_name in self.last_move_commands:
                cmd = self.last_move_commands[drone_name]
                log_msg += f"📤 上次移动指令:\n"
                log_msg += f"   AirSim速度: ({cmd.get('vx', 0):.3f}, {cmd.get('vy', 0):.3f}, {cmd.get('vz', 0):.3f})\n"

            self.logger.debug(log_msg)

        except Exception as e:
            self.logger.error(f"记录{drone_name}状态时出错: {str(e)}")

    def log_move_command(
        self,
        drone_name: str,
        unity_direction,
        unity_velocity,
        airsim_velocity,
        horizontal_speed,
        current_height,
    ):
        """
        记录移动指令

        Args:
            drone_name: 无人机名称
            unity_direction: Unity方向向量
            unity_velocity: Unity速度向量
            airsim_velocity: AirSim速度向量
            horizontal_speed: 水平速度
            current_height: 当前高度
        """
        try:
            self.last_move_commands[drone_name] = {
                "vx": airsim_velocity.x if airsim_velocity else 0,
                "vy": airsim_velocity.y if airsim_velocity else 0,
                "vz": airsim_velocity.z if airsim_velocity else 0,
                "timestamp": datetime.now(),
            }

            log_msg = f"\n{'-' * 60}\n"
            log_msg += f"📤 [{drone_name}] 移动指令 | 帧 #{self.frame_count}\n"
            log_msg += f"{'-' * 60}\n"

            if unity_direction:
                log_msg += f"Unity方向: ({unity_direction.x:.3f}, {unity_direction.y:.3f}, {unity_direction.z:.3f})\n"
            if unity_velocity:
                log_msg += f"Unity速度: ({unity_velocity.x:.3f}, {unity_velocity.y:.3f}, {unity_velocity.z:.3f})\n"
            if airsim_velocity:
                log_msg += f"AirSim速度: ({airsim_velocity.x:.3f}, {airsim_velocity.y:.3f}, {airsim_velocity.z:.3f})\n"

            log_msg += (
                f"水平速度: {horizontal_speed:.3f} m/s | 高度: {current_height:.2f}m\n"
            )

            self.logger.debug(log_msg)

        except Exception as e:
            self.logger.error(f"记录{drone_name}移动指令时出错: {str(e)}")

    def log_summary(self, drone_names: list, server):
        """
        记录汇总信息

        Args:
            drone_names: 无人机名称列表
            server: AlgorithmServer实例
        """
        try:
            log_msg = f"\n{'=' * 80}\n"
            log_msg += f"📊 诊断汇总 | 帧 #{self.frame_count}\n"
            log_msg += f"{'=' * 80}\n"

            with server.data_lock:
                for drone_name in drone_names:
                    runtime_data = server.unity_runtime_data.get(drone_name)
                    if runtime_data and runtime_data.position:
                        pos = runtime_data.position
                        log_msg += f"\n{drone_name}:\n"
                        log_msg += f"  位置: ({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f})\n"

                        if runtime_data.finalMoveDir:
                            mag = runtime_data.finalMoveDir.magnitude()
                            log_msg += f"  方向大小: {mag:.4f}\n"

                        # 检查是否在移动
                        if drone_name in self.last_positions:
                            last_pos = self.last_positions[drone_name]
                            dist = (
                                (pos.x - last_pos["x"]) ** 2
                                + (pos.y - last_pos["y"]) ** 2
                                + (pos.z - last_pos["z"]) ** 2
                            ) ** 0.5
                            status = "✅ 移动中" if dist > 0.5 else "⚠️ 可能卡住"
                            log_msg += f"  状态: {status} (移动了 {dist:.2f}m)\n"

            self.logger.info(log_msg)

        except Exception as e:
            self.logger.error(f"记录汇总时出错: {str(e)}")

    def get_log_file_path(self) -> str:
        """获取日志文件路径"""
        return str(self.log_file)


# 全局诊断日志记录器实例
_diagnostic_logger = None


def get_diagnostic_logger(log_dir: str = None) -> DroneDiagnosticLogger:
    """
    获取诊断日志记录器实例（单例模式）

    Args:
        log_dir: 日志目录

    Returns:
        DroneDiagnosticLogger实例
    """
    global _diagnostic_logger
    if _diagnostic_logger is None:
        _diagnostic_logger = DroneDiagnosticLogger(log_dir)
    return _diagnostic_logger


def reset_diagnostic_logger():
    """重置诊断日志记录器"""
    global _diagnostic_logger
    _diagnostic_logger = None
