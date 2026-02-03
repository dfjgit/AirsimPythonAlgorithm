"""
DDPG训练可视化器 - 用于DDPG权重预测训练

替代原有的TrainingVisualizer,使用新的可视化底座架构
支持:
- 训练统计(Episode、步数、奖励)
- 奖励曲线
- 权重变化历史
- 图表生成功能
"""
import sys
import os
import time
from typing import Dict, Any
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from multirotor.Visualization.base_visualizer import BaseVisualizer
from multirotor.Visualization.panels.environment_panel import EnvironmentPanel
from multirotor.Visualization.panels.training_stats_panel import TrainingStatsPanel
from multirotor.Visualization.panels.reward_curve_panel import RewardCurvePanel
from multirotor.Visualization.panels.weight_panel import WeightPanel
from multirotor.Visualization.panels.weight_history_panel import WeightHistoryPanel
from multirotor.Visualization.panels.battery_panel import BatteryPanel


class DDPGTrainingVisualizer(BaseVisualizer):
    """
    DDPG训练可视化器
    
    用于DDPG权重预测训练,显示:
    - 环境状态
    - 训练统计
    - 奖励曲线
    - 当前权重
    - 权重变化历史
    """
    
    def __init__(self, server=None, env=None):
        super().__init__(
            server=server,
            env=env,
            window_title="🎯 DDPG训练实时可视化"
        )
        
        # 训练统计数据
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.episode_count = 0
        self.total_steps = 0
        self.current_episode_reward = 0.0
        self.current_episode_steps = 0
        
        # 步骤速率统计
        self.step_timestamps = deque(maxlen=100)
        self.training_start_time = time.time()
        
        # 奖励历史
        self.reward_history = deque(maxlen=500)
        self.smoothed_rewards = deque(maxlen=500)
        
        # 权重历史（完整记录，保留最近5000步）
        self.weight_history = {
            'repulsionCoefficient': deque(maxlen=5000),
            'entropyCoefficient': deque(maxlen=5000),
            'distanceCoefficient': deque(maxlen=5000),
            'leaderRangeCoefficient': deque(maxlen=5000),
            'directionRetentionCoefficient': deque(maxlen=5000)
        }
    
    def setup_panels(self):
        """注册DDPG训练专用面板"""
        # 环境状态面板
        env_panel = EnvironmentPanel(width=350, height=180)
        self.panel_manager.register_panel(env_panel, position='auto')
        
        # 训练统计面板
        training_panel = TrainingStatsPanel(width=370, height=280)
        self.panel_manager.register_panel(training_panel, position='auto')
        
        # 奖励曲线面板
        reward_panel = RewardCurvePanel(width=370, height=200)
        self.panel_manager.register_panel(reward_panel, position='auto')
        
        # 当前权重面板
        weight_panel = WeightPanel(width=370, height=180)
        self.panel_manager.register_panel(weight_panel, position='auto')
        
        # 权重历史面板
        weight_history_panel = WeightHistoryPanel(width=370, height=250)
        self.panel_manager.register_panel(weight_history_panel, position='auto')
        
        # 电量面板
        battery_panel = BatteryPanel(width=370, height=260)
        self.panel_manager.register_panel(battery_panel, position='auto')
    
    def get_visualization_data(self) -> Dict[str, Any]:
        """收集DDPG训练可视化数据"""
        data = {}
        
        # 训练统计数据
        data['episode_count'] = self.episode_count
        data['total_steps'] = self.total_steps
        data['current_episode_steps'] = self.current_episode_steps
        data['current_episode_reward'] = self.current_episode_reward
        
        # 计算步骤速率
        steps_per_sec = 0.0
        if len(self.step_timestamps) > 1:
            time_span = self.step_timestamps[-1] - self.step_timestamps[0]
            if time_span > 0:
                steps_per_sec = len(self.step_timestamps) / time_span
        data['steps_per_sec'] = steps_per_sec
        
        # 奖励历史
        data['reward_history'] = list(self.reward_history)
        
        # 统计数据
        if self.episode_rewards:
            data['avg_reward'] = sum(self.episode_rewards) / len(self.episode_rewards)
            data['max_reward'] = max(self.episode_rewards)
            data['min_reward'] = min(self.episode_rewards)
        
        # 获取当前权重
        if self.server and hasattr(self.server, 'drone_names') and self.server.drone_names:
            first_drone = self.server.drone_names[0]
            if first_drone in self.server.algorithms:
                try:
                    weights = self.server.algorithms[first_drone].get_current_coefficients()
                    data['weights'] = weights
                    data['use_dqn'] = getattr(self.server, 'use_learned_weights', True)
                except:
                    pass
        
        # 权重历史
        data['weight_history'] = {k: list(v) for k, v in self.weight_history.items()}
        
        return data
    
    def update_training_stats(self, episode_reward: float = None, episode_length: int = None,
                             current_step_reward: float = None, is_episode_done: bool = False):
        """
        更新训练统计信息
        
        Args:
            episode_reward: 当前episode的总奖励
            episode_length: 当前episode的长度
            current_step_reward: 当前步的奖励
            is_episode_done: 是否episode结束
        """
        if current_step_reward is not None:
            self.current_episode_reward += current_step_reward
            self.current_episode_steps += 1
            self.total_steps += 1
            
            # 记录步骤时间戳
            current_time = time.time()
            self.step_timestamps.append(current_time)
            
            # ⭐ 每步都采集权重（完整记录）
            self._collect_current_weights()
        
        if is_episode_done and episode_reward is not None:
            self.episode_rewards.append(episode_reward)
            self.reward_history.append(episode_reward)
            
            # 计算滑动平均值
            window = 10
            recent_rewards = list(self.reward_history)[-window:]
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            self.smoothed_rewards.append(avg_reward)
            
            if episode_length is not None:
                self.episode_lengths.append(episode_length)
            
            self.episode_count += 1
            
            # 重置当前episode统计
            self.current_episode_reward = 0.0
            self.current_episode_steps = 0
    
    def _collect_current_weights(self):
        """从 server 采集当前权重并记录到历史"""
        try:
            if self.server and hasattr(self.server, 'drone_names') and self.server.drone_names:
                first_drone = self.server.drone_names[0]
                if first_drone in self.server.algorithms:
                    weights = self.server.algorithms[first_drone].get_current_coefficients()
                    if weights:
                        self.update_weight_history(weights)
        except Exception as e:
            print(f"[DDPGTrainingVisualizer] 采集权重错误: {e}")
    
    def update_weight_history(self, weights: Dict[str, float]):
        """
        更新权重历史
        
        Args:
            weights: 权重字典
        """
        added = 0
        for key, value in weights.items():
            if key in self.weight_history:
                self.weight_history[key].append(value)
                added += 1
    
    def generate_training_charts(self, preview_before_save: bool = True, auto_save: bool = False):
        """
        生成训练统计图表(使用matplotlib)
        
        Args:
            preview_before_save: 是否在保存前预览
            auto_save: 是否自动保存
            
        Returns:
            保存的文件路径列表,如果未保存则返回None
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            import numpy as np
            
            if not self.reward_history or len(self.reward_history) == 0:
                print("⚠️  没有足够的训练数据生成图表")
                return None
            
            print("\n📈 正在生成训练统计图表...")
            
            # 设置中文字体
            try:
                import platform
                system = platform.system()
                if system == "Windows":
                    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
                elif system == "Darwin":
                    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
                else:
                    plt.rcParams['font.sans-serif'] = ['Droid Sans Fallback', 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
            except:
                pass
            
            # 创建图表: 2行2列
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'🎯 训练统计分析 (Episodes: {self.episode_count})',
                        fontsize=16, fontweight='bold')
            
            # 图1: 奖励曲线
            ax1 = axes[0, 0]
            episodes = list(range(1, len(self.reward_history) + 1))
            rewards = list(self.reward_history)
            
            ax1.plot(episodes, rewards, 'b-', alpha=0.3, linewidth=1, label='原始奖励')
            
            if len(self.smoothed_rewards) > 0:
                smoothed = list(self.smoothed_rewards)
                smooth_episodes = episodes[-len(smoothed):]
                ax1.plot(smooth_episodes, smoothed, 'r-', linewidth=2, label='平滑奖励 (MA-10)')
            
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('奖励')
            ax1.set_title('📈 Episode 奖励曲线')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 显示平均值
            avg_reward = np.mean(rewards)
            ax1.axhline(y=avg_reward, color='g', linestyle='--', alpha=0.5,
                       label=f'平均: {avg_reward:.2f}')
            ax1.legend()
            
            # 图2: 权重变化历史
            ax2 = axes[0, 1]
            if any(len(v) > 0 for v in self.weight_history.values()):
                weight_labels = {
                    'repulsionCoefficient': '排斥系数',
                    'entropyCoefficient': '熵系数',
                    'distanceCoefficient': '距离系数',
                    'leaderRangeCoefficient': '领机系数',
                    'directionRetentionCoefficient': '方向保持系数'
                }
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
                
                for idx, (key, label) in enumerate(weight_labels.items()):
                    if key in self.weight_history and len(self.weight_history[key]) > 0:
                        values = list(self.weight_history[key])
                        steps = list(range(1, len(values) + 1))
                        ax2.plot(steps, values, color=colors[idx],
                                linewidth=2, marker='o', markersize=3,
                                label=label, alpha=0.8)
                
                ax2.set_xlabel('更新次数')
                ax2.set_ylabel('权重值')
                ax2.set_title('🎯 权重系数变化')
                ax2.legend(loc='best', fontsize=8)
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, '暂无权重数据',
                        ha='center', va='center', fontsize=14, color='gray')
                ax2.set_xticks([])
                ax2.set_yticks([])
            
            # 图3: Episode长度统计
            ax3 = axes[1, 0]
            if len(self.episode_lengths) > 0:
                lengths = list(self.episode_lengths)
                ep_nums = list(range(1, len(lengths) + 1))
                ax3.bar(ep_nums, lengths, color='skyblue', alpha=0.7)
                ax3.set_xlabel('Episode')
                ax3.set_ylabel('步数')
                ax3.set_title('👣 Episode 长度分布')
                ax3.grid(True, alpha=0.3, axis='y')
                
                avg_length = np.mean(lengths)
                ax3.axhline(y=avg_length, color='r', linestyle='--',
                           label=f'平均: {avg_length:.1f}')
                ax3.legend()
            else:
                ax3.text(0.5, 0.5, '暂无Episode长度数据',
                        ha='center', va='center', fontsize=14, color='gray')
                ax3.set_xticks([])
                ax3.set_yticks([])
            
            # 图4: 训练速率统计
            ax4 = axes[1, 1]
            if len(self.step_timestamps) > 1:
                timestamps = list(self.step_timestamps)
                time_diffs = [timestamps[i] - timestamps[i-1]
                             for i in range(1, len(timestamps))]
                
                window_size = min(20, len(time_diffs))
                if window_size > 0:
                    step_rates = []
                    for i in range(len(time_diffs)):
                        start_idx = max(0, i - window_size + 1)
                        window_times = time_diffs[start_idx:i+1]
                        avg_time = np.mean(window_times)
                        rate = 1.0 / avg_time if avg_time > 0 else 0
                        step_rates.append(rate)
                    
                    steps = list(range(1, len(step_rates) + 1))
                    ax4.plot(steps, step_rates, 'g-', linewidth=2)
                    ax4.set_xlabel('步数')
                    ax4.set_ylabel('速率 (steps/sec)')
                    ax4.set_title('🚀 训练速率')
                    ax4.grid(True, alpha=0.3)
                    
                    avg_rate = np.mean(step_rates)
                    ax4.axhline(y=avg_rate, color='r', linestyle='--',
                               label=f'平均: {avg_rate:.2f} steps/s')
                    ax4.legend()
            else:
                ax4.text(0.5, 0.5, '暂无训练速率数据',
                        ha='center', va='center', fontsize=14, color='gray')
                ax4.set_xticks([])
                ax4.set_yticks([])
            
            plt.tight_layout()
            
            # 保存逻辑
            saved_files = []
            log_dir = os.path.join(os.path.dirname(__file__), 'training_logs')
            os.makedirs(log_dir, exist_ok=True)
            
            if preview_before_save:
                print("👀 正在显示预览窗口...")
                plt.show()
                
                if not auto_save:
                    print("\n💾 是否保存图表?")
                    response = input("输入 'y' 或 'yes' 保存,其他任意键取消: ").strip().lower()
                    
                    if response in ['y', 'yes', '是', 'Y']:
                        output_path = os.path.join(log_dir,
                                                   f'training_charts_{time.strftime("%Y%m%d_%H%M%S")}.png')
                        fig.savefig(output_path, dpi=150, bbox_inches='tight')
                        saved_files.append(output_path)
                        print(f"✅ 图表已保存: {output_path}")
                    else:
                        print("❌ 已取消保存")
                else:
                    output_path = os.path.join(log_dir,
                                               f'training_charts_{time.strftime("%Y%m%d_%H%M%S")}.png')
                    fig.savefig(output_path, dpi=150, bbox_inches='tight')
                    saved_files.append(output_path)
                    print(f"✅ 图表已自动保存: {output_path}")
            else:
                output_path = os.path.join(log_dir,
                                           f'training_charts_{time.strftime("%Y%m%d_%H%M%S")}.png')
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                saved_files.append(output_path)
                print(f"✅ 图表已保存: {output_path}")
            
            plt.close(fig)
            
            return saved_files if saved_files else None
            
        except Exception as e:
            print(f"❌ 生成图表时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


# 兼容性别名
TrainingVisualizer = DDPGTrainingVisualizer


if __name__ == "__main__":
    # 测试代码
    print("DDPG训练可视化器 - 独立测试模式")
    print("提示: 此模式需要AlgorithmServer和训练环境实例才能完整显示")
    
    visualizer = DDPGTrainingVisualizer(server=None, env=None)
    visualizer.start_visualization()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n正在关闭...")
        visualizer.stop_visualization()
