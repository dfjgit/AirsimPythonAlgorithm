"""
运行时可视化器 - 用于系统运行监控

替代原有的SimpleVisualizer,使用新的可视化底座架构
"""
import sys
import os
from typing import Dict, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from multirotor.Visualization.base_visualizer import BaseVisualizer
from multirotor.Visualization.panels.entropy_overview_panel import EntropyOverviewPanel
from multirotor.Visualization.panels.entropy_trend_panel import EntropyTrendPanel
from multirotor.Visualization.panels.training_stats_panel import TrainingStatsPanel
from multirotor.Visualization.panels.reward_curve_panel import RewardCurvePanel
from multirotor.Visualization.panels.weight_panel import WeightPanel
from multirotor.Visualization.panels.battery_panel import BatteryPanel
from multirotor.Visualization.panels.reset_info_panel import ResetInfoPanel
from multirotor.training_stats_schema import normalize_training_stats


class RuntimeVisualizer(BaseVisualizer):
    """
    运行时可视化器
    
    用于在线系统监控,显示:
    - 环境状态(网格、无人机、Leader)
    - 当前APF权重
    - 系统模式(固定权重/DQN预测)
    """
    
    def __init__(self, server=None):
        super().__init__(
            server=server,
            env=None,
            window_title="无人机环境实时可视化"
        )
        self.configure_side_panel_layout(380, 380, min_center_width=440)
    
    def setup_panels(self):
        """注册运行时监控所需面板。"""
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

    def _register_fixed_column(self, panels, x: int, row_gap: int):
        y = 10
        for panel in panels:
            self.panel_manager.register_panel(panel, position="top_left")
            panel.x = x
            panel.y = y
            y += panel.height + row_gap
    
    def get_visualization_data(self) -> Dict[str, Any]:
        """收集可视化数据"""
        data: Dict[str, Any] = {}

        # 获取权重数据
        if self.server and hasattr(self.server, 'drone_names') and self.server.drone_names:
            first_drone = self.server.drone_names[0]
            if first_drone in self.server.algorithms:
                try:
                    weights = self.server.algorithms[first_drone].get_current_coefficients()
                    data['weights'] = weights
                    data['use_dqn'] = getattr(self.server, 'use_learned_weights', False)
                except:
                    pass

        server_stats = {}
        training_stats = {}
        if self.server:
            try:
                if hasattr(self.server, "current_training_stats"):
                    server_stats = getattr(self.server, "current_training_stats") or {}
                if hasattr(self.server, "training_stats"):
                    training_stats = getattr(self.server, "training_stats") or {}
            except Exception:
                server_stats = {}
                training_stats = {}

            data.update(
                normalize_training_stats(
                    stats=server_stats if isinstance(server_stats, dict) else None,
                    fallback=training_stats if isinstance(training_stats, dict) else None,
                )
            )
            data["current_training_stats"] = dict(data)

            for attr_name, public_name in (
                ("_last_reset_reason", "last_reset_reason"),
                ("_last_reset_time", "last_reset_time"),
                ("_last_collision_object_name", "last_collision_object_name"),
                ("_last_collision_penetration_depth", "last_collision_penetration_depth"),
                ("_reset_history", "reset_history"),
            ):
                if hasattr(self.server, attr_name):
                    value = getattr(self.server, attr_name)
                    data[public_name] = list(value) if public_name == "reset_history" else value
                elif hasattr(self.server, public_name):
                    value = getattr(self.server, public_name)
                    data[public_name] = list(value) if public_name == "reset_history" else value

        data.update(self.get_entropy_visualization_data())

        return data


# 兼容性别名
SimpleVisualizer = RuntimeVisualizer


if __name__ == "__main__":
    # 测试代码
    print("运行时可视化器 - 独立测试模式")
    print("提示: 此模式需要AlgorithmServer实例才能完整显示")
    
    visualizer = RuntimeVisualizer(server=None)
    visualizer.start_visualization()
    
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n正在关闭...")
        visualizer.stop_visualization()
