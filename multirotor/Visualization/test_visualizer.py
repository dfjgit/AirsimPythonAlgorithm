"""
统一可视化架构测试脚本

测试基础功能:
1. 面板系统
2. 运行时可视化器
3. 模拟数据渲染
"""
import sys
import os
import time
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from multirotor.Visualization.runtime_visualizer import RuntimeVisualizer
from multirotor.Visualization.panels.training_stats_panel import TrainingStatsPanel
from multirotor.Visualization.panels.reward_curve_panel import RewardCurvePanel
from multirotor.Algorithm.Vector3 import Vector3


class MockServer:
    """模拟AlgorithmServer用于测试"""
    
    def __init__(self):
        self.grid_data = MockGridData()
        self.unity_runtime_data = {
            'Drone1': MockRuntimeData(Vector3(0, 0, 5)),
            'Drone2': MockRuntimeData(Vector3(3, 0, 7)),
        }
        self.drone_names = ['Drone1', 'Drone2']
        self.algorithms = {
            'Drone1': MockAlgorithm(),
            'Drone2': MockAlgorithm()
        }
        self.use_learned_weights = True


class MockGridData:
    """模拟网格数据"""
    
    def __init__(self):
        self.cells = []
        # 创建一些模拟cell
        for x in range(-10, 10, 2):
            for z in range(-10, 10, 2):
                cell = MockCell(Vector3(x, 0, z), entropy=abs(x * z) % 100)
                self.cells.append(cell)


class MockCell:
    """模拟网格单元"""
    
    def __init__(self, center, entropy):
        self.center = center
        self.entropy = entropy


class MockRuntimeData:
    """模拟运行时数据"""
    
    def __init__(self, position):
        self.position = position
        self.finalMoveDir = Vector3(1, 0, 0)
        self.leader_position = Vector3(0, 0, 10)
        self.leader_scan_radius = 30.0


class MockAlgorithm:
    """模拟算法"""
    
    def get_current_coefficients(self):
        return {
            'repulsionCoefficient': 1.5,
            'entropyCoefficient': 2.0,
            'distanceCoefficient': 1.8,
            'leaderRangeCoefficient': 1.2,
            'directionRetentionCoefficient': 1.0
        }


class TestVisualizer(RuntimeVisualizer):
    """测试可视化器 - 添加训练面板"""
    
    def __init__(self, server):
        super().__init__(server)
        self.episode_count = 0
        self.total_steps = 0
        self.current_episode_reward = 0
        self.reward_history = deque(maxlen=100)
        
        # 模拟数据更新
        import threading
        self.data_thread = threading.Thread(target=self._simulate_training, daemon=True)
        self.data_thread.start()
    
    def setup_panels(self):
        """扩展面板:添加训练统计和奖励曲线"""
        super().setup_panels()
        
        # 添加训练统计面板
        training_panel = TrainingStatsPanel()
        self.panel_manager.register_panel(training_panel, position='auto')
        
        # 添加奖励曲线面板
        reward_panel = RewardCurvePanel()
        self.panel_manager.register_panel(reward_panel, position='auto')
    
    def get_visualization_data(self):
        """扩展数据:添加训练数据"""
        data = super().get_visualization_data()
        
        # 添加训练数据
        data['episode_count'] = self.episode_count
        data['total_steps'] = self.total_steps
        data['current_episode_steps'] = self.total_steps % 50
        data['current_episode_reward'] = self.current_episode_reward
        data['steps_per_sec'] = 10.5
        data['reward_history'] = list(self.reward_history)
        
        # 统计数据
        if self.reward_history:
            data['avg_reward'] = sum(self.reward_history) / len(self.reward_history)
            data['max_reward'] = max(self.reward_history)
            data['min_reward'] = min(self.reward_history)
        
        return data
    
    def _simulate_training(self):
        """模拟训练数据更新"""
        import random
        while True:
            time.sleep(0.5)
            
            self.total_steps += 10
            self.current_episode_reward += random.uniform(-5, 10)
            
            # 每50步结束一个episode
            if self.total_steps % 50 == 0:
                self.reward_history.append(self.current_episode_reward)
                self.episode_count += 1
                self.current_episode_reward = 0


def main():
    print("=" * 60)
    print("统一可视化架构测试")
    print("=" * 60)
    print()
    print("测试内容:")
    print("  ✓ 基础可视化器 (BaseVisualizer)")
    print("  ✓ 面板管理系统 (PanelManager)")
    print("  ✓ 预置面板 (Environment, Weight, Training, Reward)")
    print("  ✓ 模拟数据渲染")
    print()
    print("操作提示:")
    print("  - 观察右侧面板是否正常显示")
    print("  - 观察环境中的网格和无人机")
    print("  - 观察训练数据是否实时更新")
    print("  - 按ESC键关闭窗口")
    print("=" * 60)
    print()
    
    # 创建模拟服务器
    mock_server = MockServer()
    
    # 创建测试可视化器
    visualizer = TestVisualizer(mock_server)
    
    # 启动可视化
    print("启动可视化器...")
    visualizer.start_visualization()
    
    # 等待用户关闭
    try:
        print("可视化器已启动,按Ctrl+C停止")
        while visualizer.running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n正在关闭...")
    
    visualizer.stop_visualization()
    print("测试完成!")


if __name__ == "__main__":
    main()
