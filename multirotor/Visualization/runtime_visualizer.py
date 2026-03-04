"""
运行时可视化器 - 用于系统运行监控

替代原有的SimpleVisualizer,使用新的可视化底座架构
"""
import sys
import os
from typing import Dict, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from multirotor.Visualization.base_visualizer import BaseVisualizer
from multirotor.Visualization.panels.environment_panel import EnvironmentPanel
from multirotor.Visualization.panels.weight_panel import WeightPanel
from multirotor.Visualization.panels.battery_panel import BatteryPanel


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
    
    def setup_panels(self):
        """注册所需面板"""
        # 环境状态面板
        env_panel = EnvironmentPanel(width=350, height=180)
        self.panel_manager.register_panel(env_panel, position='auto')
        
        # 权重面板
        weight_panel = WeightPanel(width=370, height=280)
        self.panel_manager.register_panel(weight_panel, position='auto')
        
        # 电量面板
        battery_panel = BatteryPanel(width=370, height=260)
        self.panel_manager.register_panel(battery_panel, position='auto')
    
    def get_visualization_data(self) -> Dict[str, Any]:
        """收集可视化数据"""
        data = {}
        
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
