from __future__ import annotations

import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from multirotor.Algorithm.Vector3 import Vector3
from multirotor.Visualization.base_visualizer import BaseVisualizer
from multirotor.Visualization.panels.entropy_overview_panel import EntropyOverviewPanel
from multirotor.Visualization.panels.entropy_trend_panel import EntropyTrendPanel
from multirotor.Visualization.panels.training_stats_panel import TrainingStatsPanel
from multirotor.Visualization.panels.reward_curve_panel import RewardCurvePanel
from multirotor.Visualization.panels.weight_panel import WeightPanel
from multirotor.Visualization.panels.weight_history_panel import WeightHistoryPanel
from multirotor.Visualization.panels.battery_panel import BatteryPanel
from multirotor.Visualization.panels.reset_info_panel import ResetInfoPanel
from multirotor.Visualization.training_stats_csv_fallback import (
    load_latest_ddpg_visualization_snapshot,
)
from multirotor.training_stats_schema import normalize_training_stats


class DDPGTrainingVisualizer(BaseVisualizer):
    """DDPG/APF training visualizer."""

    def __init__(self, server=None, env=None):
        super().__init__(server=server, env=env, window_title="DDPG 训练实时可视化")
        self.configure_side_panel_layout(380, 380, min_center_width=440)

        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.episode_count = 0
        self.total_steps = 0
        self.current_episode_reward = 0.0
        self.current_episode_steps = 0

        self.step_timestamps = deque(maxlen=100)
        self.training_start_time = time.time()

        self.reward_history = deque(maxlen=500)
        self.smoothed_rewards = deque(maxlen=500)

        self.weight_history = {
            "repulsionCoefficient": deque(maxlen=5000),
            "entropyCoefficient": deque(maxlen=5000),
            "distanceCoefficient": deque(maxlen=5000),
            "leaderRangeCoefficient": deque(maxlen=5000),
            "directionRetentionCoefficient": deque(maxlen=5000),
            "obstacleRepulsionDistance": deque(maxlen=5000),
            "obstacleRepulsionCoefficient": deque(maxlen=5000),
            "restrictedZoneDistance": deque(maxlen=5000),
            "restrictedZoneCoefficient": deque(maxlen=5000),
        }
        self._last_weight_collection_time = 0.0
        self._weight_collection_interval = 0.1
        self._latest_csv_snapshot: Dict[str, Any] = {}
        self._csv_display_positions: Dict[str, Vector3] = {}
        self._csv_display_leader_position: Optional[Vector3] = None

    def setup_panels(self):
        """Use a fixed dashboard layout so panel size matches content density."""
        side_margin = 10
        row_gap = 10
        left_column_width = self.left_panel_width - 2 * side_margin
        right_column_width = self.right_panel_width - 2 * side_margin
        left_x = side_margin
        right_x = self.SCREEN_WIDTH - self.right_panel_width + side_margin

        left_heights = self._scale_panel_heights(
            [145, 205, 175, 225],
            min_heights=[135, 190, 165, 210],
            row_gap=row_gap,
            outer_margin=10,
        )
        right_heights = self._scale_panel_heights(
            [180, 280, 220],
            min_heights=[170, 265, 210],
            row_gap=row_gap,
            outer_margin=10,
        )

        left_panels = [
            EntropyOverviewPanel(width=left_column_width, height=left_heights[0]),
            TrainingStatsPanel(width=left_column_width, height=left_heights[1]),
            ResetInfoPanel(width=left_column_width, height=left_heights[2]),
            BatteryPanel(width=left_column_width, height=left_heights[3]),
        ]
        right_panels = [
            RewardCurvePanel(width=right_column_width, height=right_heights[0]),
            EntropyTrendPanel(width=right_column_width, height=right_heights[1]),
            WeightPanel(width=right_column_width, height=right_heights[2]),
        ]

        self._register_fixed_column(left_panels, left_x, row_gap)
        self._register_fixed_column(right_panels, right_x, row_gap)

    def _register_fixed_column(self, panels: Iterable, x: int, row_gap: int):
        y = 10
        for panel in panels:
            self.panel_manager.register_panel(panel, position="top_left")
            panel.x = x
            panel.y = y
            y += panel.height + row_gap

    def _compute_steps_per_sec(self) -> float:
        if len(self.step_timestamps) < 2:
            return 0.0
        time_span = self.step_timestamps[-1] - self.step_timestamps[0]
        if time_span <= 0:
            return 0.0
        return len(self.step_timestamps) / time_span

    def _get_current_weights(self) -> Optional[Dict[str, float]]:
        if not self.server:
            return None
        try:
            drone_names = getattr(self.server, "drone_names", None)
            algorithms = getattr(self.server, "algorithms", None)
            if not drone_names or not algorithms:
                return None
            first_drone = drone_names[0]
            if first_drone not in algorithms:
                return None
            return algorithms[first_drone].get_current_coefficients()
        except Exception:
            return None

    def _live_stats_are_empty(self) -> bool:
        current = {}
        try:
            if self.server and hasattr(self.server, "current_training_stats"):
                current = normalize_training_stats(self.server.current_training_stats)
        except Exception:
            return True
        return current.get("total_steps", 0) == 0

    def _load_csv_snapshot_if_needed(self) -> Dict[str, Any]:
        if not self._live_stats_are_empty():
            self._latest_csv_snapshot = {}
            return {}
        snapshot = load_latest_ddpg_visualization_snapshot(
            Path(__file__).resolve().parent.parent / "DDPG_Weight" / "airsim_training_logs",
            now_ts=time.time(),
        )
        self._latest_csv_snapshot = snapshot if snapshot else {}
        return self._latest_csv_snapshot

    def _smooth_csv_position(
        self, key: str, target: Dict[str, float], alpha: float = 0.35
    ) -> Vector3:
        previous = self._csv_display_positions.get(key)
        if previous is None:
            smoothed = Vector3(target["x"], target["y"], target["z"])
        else:
            smoothed = Vector3(
                previous.x + (target["x"] - previous.x) * alpha,
                previous.y + (target["y"] - previous.y) * alpha,
                previous.z + (target["z"] - previous.z) * alpha,
            )
        self._csv_display_positions[key] = smoothed
        return smoothed

    def _smooth_csv_leader_position(
        self, target: Dict[str, float], alpha: float = 0.25
    ) -> Vector3:
        previous = self._csv_display_leader_position
        if previous is None:
            smoothed = Vector3(target["x"], target["y"], target["z"])
        else:
            smoothed = Vector3(
                previous.x + (target["x"] - previous.x) * alpha,
                previous.y + (target["y"] - previous.y) * alpha,
                previous.z + (target["z"] - previous.z) * alpha,
            )
        self._csv_display_leader_position = smoothed
        return smoothed

    def update_data(self):
        grid_data, runtime_data_dict = super().update_data()
        csv_snapshot = self._load_csv_snapshot_if_needed()
        csv_positions = csv_snapshot.get("drone_positions", {})
        leader_position = csv_snapshot.get("leader_position", {})
        if csv_positions and runtime_data_dict:
            smoothed_leader = (
                self._smooth_csv_leader_position(leader_position)
                if leader_position
                else None
            )
            for drone_name, pos in csv_positions.items():
                drone_info = runtime_data_dict.setdefault(
                    drone_name,
                    {
                        "position": None,
                        "finalMoveDir": None,
                        "leaderPosition": None,
                        "leaderScanRadius": 0.0,
                    },
                )
                drone_info["position"] = self._smooth_csv_position(drone_name, pos)
                if smoothed_leader is not None:
                    drone_info["leaderPosition"] = smoothed_leader
        return grid_data, runtime_data_dict

    def get_visualization_data(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}

        fallback_stats = {
            "episode_count": self.episode_count,
            "total_steps": self.total_steps,
            "current_episode_steps": self.current_episode_steps,
            "current_episode_reward": self.current_episode_reward,
            "steps_per_sec": self._compute_steps_per_sec(),
            "reward_history": list(self.reward_history),
            "episode_reward_history": list(self.reward_history),
            "total_training_time": time.time() - self.training_start_time,
        }
        if self.episode_rewards:
            fallback_stats["avg_reward"] = sum(self.episode_rewards) / len(
                self.episode_rewards
            )
            fallback_stats["max_reward"] = max(self.episode_rewards)
            fallback_stats["min_reward"] = min(self.episode_rewards)

        server_stats = None
        training_stats = None
        try:
            if self.server and hasattr(self.server, "current_training_stats"):
                server_stats = self.server.current_training_stats
            if self.server and hasattr(self.server, "training_stats"):
                training_stats = self.server.training_stats
        except Exception:
            server_stats = None
            training_stats = None

        normalized_server_stats = normalize_training_stats(
            stats=server_stats if isinstance(server_stats, dict) else None,
            fallback=fallback_stats,
        )
        server_stats_looks_empty = (
            normalized_server_stats.get("total_steps", 0) == 0
            and normalized_server_stats.get("current_episode_steps", 0) == 0
            and normalized_server_stats.get("current_episode_reward", 0.0) == 0.0
            and normalized_server_stats.get("episode_count", 0) == 0
        )
        if server_stats_looks_empty and isinstance(training_stats, dict):
            normalized_training_stats = normalize_training_stats(
                stats=training_stats,
                fallback=normalized_server_stats,
            )
            if normalized_training_stats.get("total_steps", 0) > 0 or normalized_training_stats.get(
                "episode_count", 0
            ) > 0:
                data["stats_source"] = "training_stats_fallback"
                data.update(normalized_training_stats)
            else:
                data["stats_source"] = "current_training_stats"
                data.update(normalized_server_stats)
        else:
            data["stats_source"] = "current_training_stats"
            data.update(normalized_server_stats)

        if data.get("total_steps", 0) == 0:
            csv_snapshot = self._load_csv_snapshot_if_needed()
            csv_stats = csv_snapshot.get("training_stats", {})
            if csv_stats.get("total_steps", 0) > 0:
                data["stats_source"] = "csv_fallback"
                data.update(csv_stats)
                if csv_snapshot.get("global_scanned_count", 0) > 0:
                    data["csv_global_scanned_count"] = csv_snapshot["global_scanned_count"]
                    data["csv_global_total_count"] = csv_snapshot.get(
                        "global_total_count", 0
                    )
                if csv_snapshot.get("battery_data"):
                    data["battery_data"] = csv_snapshot["battery_data"]

        current_time = time.time()
        if current_time - self._last_weight_collection_time >= self._weight_collection_interval:
            weights = self._get_current_weights()
            if weights:
                self.update_weight_history(weights)
                self._last_weight_collection_time = current_time

        csv_weights = self._latest_csv_snapshot.get("current_weights")
        if csv_weights and self._live_stats_are_empty():
            csv_weights = self._latest_csv_snapshot["current_weights"]
            data["weights"] = csv_weights
            data["use_dqn"] = getattr(self.server, "use_learned_weights", True) if self.server else True
            if current_time - self._last_weight_collection_time >= self._weight_collection_interval:
                self.update_weight_history(csv_weights)
                self._last_weight_collection_time = current_time
        else:
            weights = self._get_current_weights()
            if weights:
                data["weights"] = weights
                data["use_dqn"] = getattr(self.server, "use_learned_weights", True) if self.server else True

        data["weight_history"] = {key: list(values) for key, values in self.weight_history.items()}

        if self.server:
            for local_name, public_name in (
                ("_last_reset_reason", "last_reset_reason"),
                ("_last_reset_time", "last_reset_time"),
                ("_last_collision_object_name", "last_collision_object_name"),
                ("_last_collision_penetration_depth", "last_collision_penetration_depth"),
                ("_reset_history", "reset_history"),
            ):
                if hasattr(self.server, local_name):
                    value = getattr(self.server, local_name)
                    data[public_name] = list(value) if local_name == "_reset_history" else value
                elif hasattr(self.server, public_name):
                    value = getattr(self.server, public_name)
                    data[public_name] = list(value) if public_name == "reset_history" else value

        data.update(self.get_entropy_visualization_data())

        return data

    def update_training_stats(
        self,
        episode_reward: float = None,
        episode_length: int = None,
        current_step_reward: float = None,
        is_episode_done: bool = False,
    ):
        if current_step_reward is not None:
            self.current_episode_reward += current_step_reward
            self.current_episode_steps += 1
            self.total_steps += 1
            self.step_timestamps.append(time.time())

        if is_episode_done and episode_reward is not None:
            self.episode_rewards.append(episode_reward)
            self.reward_history.append(episode_reward)

            recent_rewards = list(self.reward_history)[-10:]
            if recent_rewards:
                self.smoothed_rewards.append(sum(recent_rewards) / len(recent_rewards))

            if episode_length is not None:
                self.episode_lengths.append(episode_length)

            self.episode_count += 1
            self.current_episode_reward = 0.0
            self.current_episode_steps = 0

    def update_weight_history(self, weights: Dict[str, float]):
        for key, value in weights.items():
            if key in self.weight_history:
                self.weight_history[key].append(value)

    def generate_training_charts(
        self, preview_before_save: bool = True, auto_save: bool = False
    ):
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except Exception as exc:
            print(f"[DDPGTrainingVisualizer] 无法导入绘图库: {exc}")
            return None

        if not self.reward_history:
            print("[DDPGTrainingVisualizer] 奖励历史为空，无法生成图表")
            return None

        try:
            import platform

            system = platform.system()
            if system == "Windows":
                plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
            elif system == "Darwin":
                plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
            else:
                plt.rcParams["font.sans-serif"] = ["Droid Sans Fallback", "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
        except Exception:
            pass

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"DDPG 训练统计 (Episodes: {self.episode_count})", fontsize=16, fontweight="bold")

        episodes = list(range(1, len(self.reward_history) + 1))
        rewards = list(self.reward_history)
        ax1 = axes[0, 0]
        ax1.plot(episodes, rewards, "b-", alpha=0.3, linewidth=1, label="原始奖励")
        if self.smoothed_rewards:
            smoothed = list(self.smoothed_rewards)
            smooth_episodes = episodes[-len(smoothed):]
            ax1.plot(smooth_episodes, smoothed, "r-", linewidth=2, label="平滑奖励 (MA-10)")
        avg_reward = float(np.mean(rewards))
        ax1.axhline(y=avg_reward, color="g", linestyle="--", alpha=0.5, label=f"平均值: {avg_reward:.2f}")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward")
        ax1.set_title("Episode 奖励变化")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = axes[0, 1]
        weight_labels = {
            "repulsionCoefficient": "排斥",
            "entropyCoefficient": "熵值",
            "distanceCoefficient": "距离",
            "leaderRangeCoefficient": "Leader",
            "directionRetentionCoefficient": "方向",
        }
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8"]
        has_weight_curve = False
        for idx, (key, label) in enumerate(weight_labels.items()):
            values = list(self.weight_history.get(key, []))
            if not values:
                continue
            has_weight_curve = True
            steps = list(range(1, len(values) + 1))
            ax2.plot(steps, values, color=colors[idx], linewidth=2, label=label, alpha=0.85)
        if has_weight_curve:
            ax2.set_xlabel("Step")
            ax2.set_ylabel("Weight")
            ax2.set_title("核心 APF 权重变化")
            ax2.legend(loc="best", fontsize=8)
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, "暂无权重历史", ha="center", va="center", fontsize=14, color="gray")
            ax2.set_xticks([])
            ax2.set_yticks([])

        ax3 = axes[1, 0]
        if self.episode_lengths:
            lengths = list(self.episode_lengths)
            ep_nums = list(range(1, len(lengths) + 1))
            ax3.bar(ep_nums, lengths, color="skyblue", alpha=0.7)
            avg_length = float(np.mean(lengths))
            ax3.axhline(y=avg_length, color="r", linestyle="--", label=f"平均长度: {avg_length:.1f}")
            ax3.set_xlabel("Episode")
            ax3.set_ylabel("Length")
            ax3.set_title("Episode 长度分布")
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis="y")
        else:
            ax3.text(0.5, 0.5, "暂无 Episode 长度数据", ha="center", va="center", fontsize=14, color="gray")
            ax3.set_xticks([])
            ax3.set_yticks([])

        ax4 = axes[1, 1]
        if len(self.step_timestamps) > 1:
            timestamps = list(self.step_timestamps)
            time_diffs = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
            if time_diffs:
                window_size = min(20, len(time_diffs))
                step_rates = []
                for idx in range(len(time_diffs)):
                    start_idx = max(0, idx - window_size + 1)
                    avg_time = float(np.mean(time_diffs[start_idx: idx + 1]))
                    step_rates.append(1.0 / avg_time if avg_time > 0 else 0.0)
                avg_rate = float(np.mean(step_rates))
                steps = list(range(1, len(step_rates) + 1))
                ax4.plot(steps, step_rates, "g-", linewidth=2)
                ax4.axhline(y=avg_rate, color="r", linestyle="--", label=f"平均速率: {avg_rate:.2f} steps/s")
                ax4.set_xlabel("Recent Step")
                ax4.set_ylabel("Rate")
                ax4.set_title("训练步速")
                ax4.legend()
                ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, "暂无步速数据", ha="center", va="center", fontsize=14, color="gray")
            ax4.set_xticks([])
            ax4.set_yticks([])

        plt.tight_layout()

        saved_files = []
        log_dir = os.path.join(os.path.dirname(__file__), "training_logs")
        os.makedirs(log_dir, exist_ok=True)

        if preview_before_save:
            plt.show()
            if not auto_save:
                response = input("输入 y/yes 保存图表，其他任意键取消: ").strip().lower()
                if response not in {"y", "yes"}:
                    plt.close(fig)
                    return None

        output_path = os.path.join(log_dir, f"training_charts_{time.strftime('%Y%m%d_%H%M%S')}.png")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        saved_files.append(output_path)
        plt.close(fig)
        print(f"[DDPGTrainingVisualizer] 图表已保存: {output_path}")
        return saved_files


TrainingVisualizer = DDPGTrainingVisualizer


if __name__ == "__main__":
    print("DDPG 训练可视化模块")
    visualizer = DDPGTrainingVisualizer(server=None, env=None)
    print(f"初始化完成: {visualizer.window_title}")
