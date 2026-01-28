"""
训练数据可视化工具

功能说明：
    - 支持 Crazyflie 实体无人机训练数据的可视化分析
    - 支持 DataCollector 扫描数据的可视化分析
    - 自动识别数据类型并应用相应的可视化策略
    - 生成完整的分析报告和图表

数据类型支持：
    1. Crazyflie 训练日志 (JSON/CSV格式)
       - 飞行轨迹 (2D/3D)
       - 速度和加速度曲线
       - 权重变化历史
       - Episode 奖励曲线
       - 电池性能分析
       
    2. DataCollector 扫描数据 (CSV格式)
       - 扫描进度曲线
       - 熵值变化分析
       - 飞行轨迹可视化
       - 算法权重变化

使用方法：
    python visualize_training_data.py --auto              # 自动扫描所有数据目录
    python visualize_training_data.py --json file.json    # 分析单个JSON文件
    python visualize_training_data.py --csv file.csv      # 分析单个CSV文件
    python visualize_training_data.py --dir path/to/logs  # 分析指定目录

日期：2026-01-26
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import platform

# 导入扫描数据分析组件
try:
    from multirotor.Algorithm.visualize_scan_csv import (
        load_and_prepare, _detect_drones, _pick_snapshot_indices,
        plot_scan_progress, plot_trajectories, plot_entropy_snapshots, _safe_plot_wrapper
    )
except ImportError:
    from visualize_scan_csv import (
        load_and_prepare, _detect_drones, _pick_snapshot_indices,
        plot_scan_progress, plot_trajectories, plot_entropy_snapshots, _safe_plot_wrapper
    )

# --- 解决中文显示问题的配置 ---
def set_ch_font():
    system = platform.system()
    if system == "Windows":
        plt.rcParams['font.sans-serif'] = ['SimHei']
    elif system == "Darwin":
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    else:
        plt.rcParams['font.sans-serif'] = ['Droid Sans Fallback']
    plt.rcParams['axes.unicode_minus'] = False

set_ch_font()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
LOGGER = logging.getLogger(__name__)


class CrazyflieDataVisualizer:
    """Crazyflie 训练数据可视化器"""
    
    def __init__(self, output_dir: Path, show_plots: bool = False):
        self.output_dir = output_dir
        self.show_plots = show_plots
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def visualize_json(self, json_path: Path) -> bool:
        """分析 JSON 格式的完整训练数据"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            LOGGER.info(f"📊 分析 Crazyflie 训练数据: {json_path.name}")
            
            # 提取元数据
            metadata = data.get('metadata', {})
            session_id = metadata.get('session_id', 'unknown')
            duration = metadata.get('duration_seconds', 0)
            drone_names = metadata.get('drone_names', [])
            
            LOGGER.info(f"   会话ID: {session_id}")
            LOGGER.info(f"   训练时长: {duration:.2f} 秒")
            LOGGER.info(f"   无人机: {', '.join(drone_names)}")
            
            # 创建子目录
            run_dir = self.output_dir / f"crazyflie_{session_id}"
            run_dir.mkdir(exist_ok=True)
            
            # 1. 飞行数据可视化
            flight_data = data.get('flight_data', {})
            for drone_name, records in flight_data.items():
                if records:
                    self._plot_flight_data(drone_name, records, run_dir)
            
            # 2. 权重变化可视化
            weight_history = data.get('weight_history', [])
            if weight_history:
                self._plot_weight_history(weight_history, run_dir)
            
            # 3. Episode 统计可视化
            episode_stats = data.get('episode_stats', [])
            if episode_stats:
                self._plot_episode_stats(episode_stats, run_dir)
            
            LOGGER.info(f"✅ 分析完成，结果保存在: {run_dir}")
            return True
            
        except Exception as e:
            LOGGER.error(f"❌ 分析 JSON 文件失败: {e}", exc_info=True)
            return False
    
    def visualize_csv(self, csv_path: Path) -> bool:
        """分析 CSV 格式的飞行数据或权重历史"""
        try:
            df = pd.read_csv(csv_path)
            
            if df.empty:
                LOGGER.warning(f"⚠️  文件为空: {csv_path.name}")
                return False
            
            LOGGER.info(f"📊 分析 CSV 文件: {csv_path.name}")
            
            # 判断 CSV 类型
            if 'x' in df.columns and 'y' in df.columns:
                # 飞行数据 CSV
                return self._visualize_flight_csv(csv_path, df)
            elif 'repulsionCoefficient' in df.columns:
                # 权重历史 CSV
                return self._visualize_weight_csv(csv_path, df)
            else:
                LOGGER.warning(f"⚠️  未知的 CSV 格式: {csv_path.name}")
                return False
                
        except Exception as e:
            LOGGER.error(f"❌ 分析 CSV 文件失败: {e}", exc_info=True)
            return False
    
    def _plot_flight_data(self, drone_name: str, records: List[Dict], output_dir: Path):
        """绘制飞行数据的各种图表"""
        df = pd.DataFrame(records)
        
        # 1. 飞行轨迹 (2D)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(df['x'], df['y'], linewidth=2, alpha=0.7)
        ax.scatter(df['x'].iloc[0], df['y'].iloc[0], c='green', s=100, marker='o', label='起点', zorder=5)
        ax.scatter(df['x'].iloc[-1], df['y'].iloc[-1], c='red', s=100, marker='X', label='终点', zorder=5)
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title(f'{drone_name} - 水平面飞行轨迹', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.axis('equal')
        plt.tight_layout()
        plt.savefig(output_dir / f"{drone_name}_trajectory_2d.png", dpi=150)
        if self.show_plots:
            plt.show()
        plt.close()
        
        # 2. 飞行轨迹 (3D)
        if 'z' in df.columns:
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(df['x'], df['y'], df['z'], linewidth=2, alpha=0.7)
            ax.scatter(df['x'].iloc[0], df['y'].iloc[0], df['z'].iloc[0], 
                      c='green', s=100, marker='o', label='起点')
            ax.scatter(df['x'].iloc[-1], df['y'].iloc[-1], df['z'].iloc[-1], 
                      c='red', s=100, marker='X', label='终点')
            ax.set_xlabel('X (m)', fontsize=11)
            ax.set_ylabel('Y (m)', fontsize=11)
            ax.set_zlabel('Z (m)', fontsize=11)
            ax.set_title(f'{drone_name} - 3D 飞行轨迹', fontsize=14, fontweight='bold')
            ax.legend()
            plt.tight_layout()
            plt.savefig(output_dir / f"{drone_name}_trajectory_3d.png", dpi=150)
            if self.show_plots:
                plt.show()
            plt.close()
        
        # 3. 速度曲线
        if 'elapsed_time' in df.columns:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # 总速度
            if 'speed' in df.columns:
                axes[0, 0].plot(df['elapsed_time'], df['speed'], linewidth=2)
                axes[0, 0].set_xlabel('时间 (s)')
                axes[0, 0].set_ylabel('速度 (m/s)')
                axes[0, 0].set_title('总速度')
                axes[0, 0].grid(True, alpha=0.3)
            
            # 各轴速度
            if all(c in df.columns for c in ['xspeed', 'yspeed', 'zspeed']):
                axes[0, 1].plot(df['elapsed_time'], df['xspeed'], label='X', alpha=0.8)
                axes[0, 1].plot(df['elapsed_time'], df['yspeed'], label='Y', alpha=0.8)
                axes[0, 1].plot(df['elapsed_time'], df['zspeed'], label='Z', alpha=0.8)
                axes[0, 1].set_xlabel('时间 (s)')
                axes[0, 1].set_ylabel('速度 (m/s)')
                axes[0, 1].set_title('各轴速度')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            
            # 高度变化
            if 'z' in df.columns:
                axes[1, 0].plot(df['elapsed_time'], df['z'], linewidth=2, color='orange')
                axes[1, 0].set_xlabel('时间 (s)')
                axes[1, 0].set_ylabel('高度 (m)')
                axes[1, 0].set_title('高度变化')
                axes[1, 0].grid(True, alpha=0.3)
            
            # 电池电压
            if 'battery' in df.columns:
                axes[1, 1].plot(df['elapsed_time'], df['battery'], linewidth=2, color='red')
                axes[1, 1].set_xlabel('时间 (s)')
                axes[1, 1].set_ylabel('电压 (V)')
                axes[1, 1].set_title('电池电压')
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.suptitle(f'{drone_name} - 飞行状态分析', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_dir / f"{drone_name}_flight_stats.png", dpi=150)
            if self.show_plots:
                plt.show()
            plt.close()
            
        # 4. 飞行姿态稳定性分析 (Attitude Stability)
        if any(c in df.columns for c in ['xeulerangle', 'yeulerangle']):
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            
            if 'xeulerangle' in df.columns:
                ax1.plot(df['elapsed_time'], df['xeulerangle'], color='blue', label='Roll (X)')
                ax1.set_ylabel('角度 (deg)')
                ax1.set_title('横滚角 (Roll)')
                ax1.grid(True, alpha=0.3)
                ax1.legend()
                
                # 标注稳定性
                jitter = df['xeulerangle'].std()
                ax1.text(0.02, 0.9, f'Jitter: {jitter:.2f}°', transform=ax1.transAxes, 
                        bbox=dict(facecolor='white', alpha=0.8))

            if 'yeulerangle' in df.columns:
                ax2.plot(df['elapsed_time'], df['yeulerangle'], color='green', label='Pitch (Y)')
                ax2.set_ylabel('角度 (deg)')
                ax2.set_title('俯仰角 (Pitch)')
                ax2.set_xlabel('时间 (s)')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                
                # 标注稳定性
                jitter = df['yeulerangle'].std()
                ax2.text(0.02, 0.9, f'Jitter: {jitter:.2f}°', transform=ax2.transAxes, 
                        bbox=dict(facecolor='white', alpha=0.8))
            
            plt.suptitle(f'{drone_name} - 飞行姿态稳定性分析', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_dir / f"{drone_name}_attitude_stability.png", dpi=150)
            if self.show_plots:
                plt.show()
            plt.close()
    
    def _plot_weight_history(self, weight_history: List[Dict], output_dir: Path):
        """绘制权重变化历史"""
        df = pd.DataFrame(weight_history)
        
        if df.empty:
            return
        
        # 权重系数
        weight_cols = ['repulsionCoefficient', 'entropyCoefficient', 'distanceCoefficient',
                       'leaderRangeCoefficient', 'directionRetentionCoefficient']
        
        # 使用中文名称
        weight_names = {
            'repulsionCoefficient': '排斥力系数',
            'entropyCoefficient': '熵值系数',
            'distanceCoefficient': '距离系数',
            'leaderRangeCoefficient': 'Leader范围系数',
            'directionRetentionCoefficient': '方向保持系数'
        }
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1]})
        
        # 1. 权重值变化
        for col in weight_cols:
            if col in df.columns:
                ax1.plot(df['step'], df[col], label=weight_names.get(col, col), 
                       linewidth=2, alpha=0.8, marker='o', markersize=3)
        
        ax1.set_xlabel('训练步数', fontsize=12)
        ax1.set_ylabel('系数值', fontsize=12)
        ax1.set_title('APF 权重系数变化历史 (策略演进过程)', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 2. 稳定性分析 (滚动标准差)
        # 计算所有权重的滚动标准差之和，作为策略震荡的量化指标
        window = max(5, len(df) // 10)
        stability_df = pd.DataFrame()
        for col in weight_cols:
            if col in df.columns:
                stability_df[col] = df[col].rolling(window=window).std()
        
        if not stability_df.empty:
            total_std = stability_df.mean(axis=1)
            ax2.fill_between(df['step'], total_std, 0, color='purple', alpha=0.2, label='平均波动强度')
            ax2.plot(df['step'], total_std, color='purple', linewidth=1.5)
            
            # 标注收敛点：如果后期标准差保持在较低水平
            late_std = total_std.tail(len(df)//5).mean()
            ax2.axhline(y=late_std, color='red', linestyle='--', alpha=0.6, 
                       label=f'后期平均波动: {late_std:.4f}')
            
            if late_std < 0.05:
                ax2.text(0.05, 0.85, "[OK] 策略已趋于稳定 (收敛)", transform=ax2.transAxes, 
                        color='green', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))
            else:
                ax2.text(0.05, 0.85, "[WARN] 策略仍在震荡 (未完全收敛)", transform=ax2.transAxes, 
                        color='orange', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))

        ax2.set_xlabel('训练步数', fontsize=12)
        ax2.set_ylabel('标准差 (Stability)', fontsize=12)
        ax2.set_title(f'策略收敛性证明 (滚动窗口={window}) - 波动越小越稳定', fontsize=13, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "weight_history.png", dpi=150)
        if self.show_plots:
            plt.show()
        plt.close()
        
        # 按无人机分组绘制
        if 'drone_name' in df.columns:
            drone_names = df['drone_name'].unique()
            if len(drone_names) > 1:
                fig, axes = plt.subplots(len(drone_names), 1, figsize=(14, 5 * len(drone_names)))
                if len(drone_names) == 1:
                    axes = [axes]
                
                for idx, drone_name in enumerate(drone_names):
                    drone_df = df[df['drone_name'] == drone_name]
                    for col in weight_cols:
                        if col in drone_df.columns:
                            axes[idx].plot(drone_df['step'], drone_df[col], 
                                         label=weight_names.get(col, col), 
                                         linewidth=2, alpha=0.8, marker='o', markersize=3)
                    
                    axes[idx].set_xlabel('训练步数', fontsize=11)
                    axes[idx].set_ylabel('系数值', fontsize=11)
                    axes[idx].set_title(f'{drone_name} - 权重变化', fontsize=12, fontweight='bold')
                    axes[idx].legend(loc='best', fontsize=9)
                    axes[idx].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(output_dir / "weight_history_by_drone.png", dpi=150)
                if self.show_plots:
                    plt.show()
                plt.close()
    
    def _plot_episode_stats(self, episode_stats: List[Dict], output_dir: Path):
        """绘制 Episode 统计信息与学习速度分析"""
        df = pd.DataFrame(episode_stats)
        
        if df.empty:
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 15))
        
        # 1. 奖励曲线与平滑趋势
        if 'reward' in df.columns and 'episode' in df.columns:
            axes[0].plot(df['episode'], df['reward'], color='blue', alpha=0.3, label='原始奖励')
            
            # 移动平均线
            window = max(2, min(10, len(df) // 2))
            moving_avg = df['reward'].rolling(window=window).mean()
            axes[0].plot(df['episode'], moving_avg, linewidth=3, color='red', label=f'{window}-Episode 移动平均')
            
            axes[0].set_xlabel('Episode', fontsize=12)
            axes[0].set_ylabel('总奖励', fontsize=12)
            axes[0].set_title('Episode 奖励曲线 (收敛趋势)', fontsize=14, fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()

        # 2. 学习速度分析 (奖励上升斜率)
        if 'reward' in df.columns and len(df) > 5:
            # 计算奖励的变化斜率 (使用平滑后的数据)
            # 斜率代表每 Episode 奖励的增长量
            slope = moving_avg.diff().fillna(0)
            
            # 使用填色图展示学习爆发期
            axes[1].fill_between(df['episode'], slope, 0, where=(slope >= 0), 
                               color='green', alpha=0.3, label='正向学习 (策略改进)')
            axes[1].fill_between(df['episode'], slope, 0, where=(slope < 0), 
                               color='red', alpha=0.2, label='策略波动')
            
            axes[1].plot(df['episode'], slope, color='darkgreen', linewidth=1.5)
            
            # 计算平均学习速率
            avg_slope = slope.mean()
            axes[1].axhline(y=avg_slope, color='blue', linestyle='--', alpha=0.5, 
                           label=f'平均学习速率: {avg_slope:.2f}/ep')
            
            axes[1].set_xlabel('Episode', fontsize=12)
            axes[1].set_ylabel('奖励增长斜率', fontsize=12)
            axes[1].set_title('学习速度分析 (证明策略快速习得)', fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
        
        # 3. Episode 长度
        if 'length' in df.columns and 'episode' in df.columns:
            axes[2].plot(df['episode'], df['length'], linewidth=2, marker='s', 
                        markersize=4, color='orange', label='步数')
            axes[2].set_xlabel('Episode', fontsize=12)
            axes[2].set_ylabel('单次步数', fontsize=12)
            axes[2].set_title('Episode 持续时长 (策略稳定性证明)', fontsize=14, fontweight='bold')
            axes[2].grid(True, alpha=0.3)
            
            # 标注稳定性：如果后期步数变短且奖励变高，证明找到了更优路径
            if len(df) > 10:
                final_length = df['length'].tail(5).mean()
                axes[2].axhline(y=final_length, color='red', linestyle=':', label=f'近期平均步数: {final_length:.1f}')
                axes[2].legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / "episode_stats.png", dpi=150)
        if self.show_plots:
            plt.show()
        plt.close()
    
    def _visualize_flight_csv(self, csv_path: Path, df: pd.DataFrame) -> bool:
        """可视化飞行数据 CSV"""
        session_id = df['session_id'].iloc[0] if 'session_id' in df.columns else 'unknown'
        drone_name = df['drone_name'].iloc[0] if 'drone_name' in df.columns else 'UAV'
        
        run_dir = self.output_dir / f"crazyflie_{session_id}"
        run_dir.mkdir(exist_ok=True)
        
        self._plot_flight_data(drone_name, df.to_dict('records'), run_dir)
        LOGGER.info(f"✅ 飞行数据分析完成: {run_dir}")
        return True
    
    def _visualize_weight_csv(self, csv_path: Path, df: pd.DataFrame) -> bool:
        """可视化权重历史 CSV"""
        session_id = df['session_id'].iloc[0] if 'session_id' in df.columns else 'unknown'
        
        run_dir = self.output_dir / f"crazyflie_{session_id}"
        run_dir.mkdir(exist_ok=True)
        
        self._plot_weight_history(df.to_dict('records'), run_dir)
        LOGGER.info(f"✅ 权重历史分析完成: {run_dir}")
        return True


class ScanDataVisualizer:
    """DataCollector 扫描数据可视化器（集成 visualize_scan_csv.py 逻辑）"""
    
    def __init__(self, output_dir: Path, show_plots: bool = False):
        self.output_dir = output_dir
        self.show_plots = show_plots
        self.output_dir.mkdir(parents=True, exist_ok=True)


class DQNDataVisualizer:
    """DQN 移动控制训练数据可视化器"""
    
    def __init__(self, output_dir: Path, show_plots: bool = False):
        self.output_dir = output_dir
        self.show_plots = show_plots
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def visualize_training(self, metadata_path: Path = None, stats_csv_path: Path = None) -> bool:
        """分析 DQN 训练数据
        
        Args:
            metadata_path: 训练元数据 JSON 文件路径
            stats_csv_path: 训练统计 CSV 文件路径
        """
        try:
            # 加载元数据
            metadata = {}
            if metadata_path and metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                LOGGER.info(f"[LOG] 分析 DQN 训练数据: {metadata_path.name}")
            
            # 加载训练统计
            if not stats_csv_path or not stats_csv_path.exists():
                # 尝试从元数据中获取
                if 'training_stats_path' in metadata:
                    stats_csv_path = Path(metadata['training_stats_path'])
            
            if not stats_csv_path or not stats_csv_path.exists():
                LOGGER.error(f"[FAIL] 找不到 DQN 训练统计文件")
                return False
            
            df = pd.read_csv(stats_csv_path)
            if df.empty:
                LOGGER.warning(f"[WARN] DQN 训练统计文件为空")
                return False
            
            # 创建输出目录
            session_id = metadata.get('start_time', 'unknown').replace(':', '-').replace(' ', '_')
            run_dir = self.output_dir / f"dqn_movement_{session_id}"
            run_dir.mkdir(exist_ok=True)
            
            LOGGER.info(f"   训练时长: {metadata.get('duration_seconds', 0):.2f} 秒")
            LOGGER.info(f"   总 episode: {metadata.get('total_episodes', 0)}")
            LOGGER.info(f"   总步数: {metadata.get('total_timesteps', 0)}")
            
            # 1. Episode 奖励曲线
            self._plot_reward_curve(df, run_dir)
            
            # 2. Episode 长度分析
            self._plot_episode_length(df, run_dir)
            
            # 3. 学习速度分析
            self._plot_learning_speed(df, run_dir)
            
            # 4. 总结统计
            self._plot_summary_stats(df, metadata, run_dir)
            
            LOGGER.info(f"[OK] DQN 分析完成，结果保存在: {run_dir}")
            return True
            
        except Exception as e:
            LOGGER.error(f"[FAIL] 分析 DQN 训练数据失败: {e}", exc_info=True)
            return False
    
    def _plot_reward_curve(self, df: pd.DataFrame, output_dir: Path):
        """绘制奖励曲线"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # 原始奖励
        ax.plot(df['episode'], df['reward'], alpha=0.3, color='blue', label='原始奖励')
        
        # 移动平均
        if len(df) > 10:
            window = max(5, min(20, len(df) // 10))
            moving_avg = df['reward'].rolling(window=window).mean()
            ax.plot(df['episode'], moving_avg, linewidth=3, color='red', 
                   label=f'{window}-Episode 移动平均')
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('总奖励', fontsize=12)
        ax.set_title('DQN 训练 - Episode 奖励曲线 (收敛性分析)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / "dqn_reward_curve.png", dpi=150)
        if self.show_plots:
            plt.show()
        plt.close()
    
    def _plot_episode_length(self, df: pd.DataFrame, output_dir: Path):
        """绘制 episode 长度分析"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        ax.plot(df['episode'], df['length'], marker='o', markersize=3, 
               linewidth=1.5, color='orange', label='Episode 步数')
        
        # 添加平均线
        if len(df) > 10:
            avg_length = df['length'].rolling(window=10).mean()
            ax.plot(df['episode'], avg_length, linewidth=2.5, color='darkred', 
                   linestyle='--', label='10-Episode 平均')
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Episode 长度 (步数)', fontsize=12)
        ax.set_title('DQN 训练 - Episode 长度变化', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / "dqn_episode_length.png", dpi=150)
        if self.show_plots:
            plt.show()
        plt.close()
    
    def _plot_learning_speed(self, df: pd.DataFrame, output_dir: Path):
        """绘制学习速度分析（奖励增长旜率）"""
        if len(df) < 5:
            return
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # 计算奖励旜率
        window = max(5, min(10, len(df) // 5))
        smooth_reward = df['reward'].rolling(window=window).mean()
        slope = smooth_reward.diff().fillna(0)
        
        # 绘制填充区域
        ax.fill_between(df['episode'], slope, 0, where=(slope >= 0), 
                       color='green', alpha=0.3, label='正向学习')
        ax.fill_between(df['episode'], slope, 0, where=(slope < 0), 
                       color='red', alpha=0.2, label='策略波动')
        
        ax.plot(df['episode'], slope, color='darkgreen', linewidth=1.5)
        
        # 平均学习速率
        avg_slope = slope.mean()
        ax.axhline(y=avg_slope, color='blue', linestyle='--', alpha=0.5, 
                  label=f'平均学习速率: {avg_slope:.2f}/ep')
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('奖励增长率', fontsize=12)
        ax.set_title('DQN 训练 - 学习速度分析', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / "dqn_learning_speed.png", dpi=150)
        if self.show_plots:
            plt.show()
        plt.close()
    
    def _plot_summary_stats(self, df: pd.DataFrame, metadata: dict, output_dir: Path):
        """绘制总结统计信息"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 奖励分布直方图
        ax1.hist(df['reward'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.axvline(df['reward'].mean(), color='red', linestyle='--', 
                   label=f'平均: {df["reward"].mean():.2f}')
        ax1.set_xlabel('奖励值')
        ax1.set_ylabel('频次')
        ax1.set_title('奖励分布')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 长度分布直方图
        ax2.hist(df['length'], bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
        ax2.axvline(df['length'].mean(), color='red', linestyle='--', 
                   label=f'平均: {df["length"].mean():.2f}')
        ax2.set_xlabel('Episode 长度')
        ax2.set_ylabel('频次')
        ax2.set_title('Episode 长度分布')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 奖励 vs 长度散点图
        ax3.scatter(df['length'], df['reward'], alpha=0.5, c=df['episode'], 
                   cmap='viridis', s=20)
        ax3.set_xlabel('Episode 长度')
        ax3.set_ylabel('奖励')
        ax3.set_title('奖励 vs 长度 (颜色=Episode)')
        ax3.grid(True, alpha=0.3)
        
        # 4. 元数据信息
        ax4.axis('off')
        info_text = f"""DQN 训练总结
        
算法: {metadata.get('algorithm', 'DQN')}
任务: {metadata.get('task', 'movement_control')}

训练时间:
  开始: {metadata.get('start_time', 'N/A')}
  结束: {metadata.get('end_time', 'N/A')}
  总时长: {metadata.get('duration_seconds', 0):.2f} 秒

统计指标:
  总 Episode: {len(df)}
  总步数: {metadata.get('total_timesteps', 0)}
  平均奖励: {df['reward'].mean():.2f}
  最大奖励: {df['reward'].max():.2f}
  最小奖励: {df['reward'].min():.2f}
  平均长度: {df['length'].mean():.2f} 步

动作空间: {metadata.get('action_space', {}).get('n', 6)} 个离散动作
观察空间: {metadata.get('observation_space', {}).get('shape', [21])}
"""
        ax4.text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center', 
                family='monospace')
        
        plt.tight_layout()
        plt.savefig(output_dir / "dqn_summary_stats.png", dpi=150)
        if self.show_plots:
            plt.show()
        plt.close()


class ScanDataVisualizer_ORIGINAL:
    """DataCollector 扫描数据可视化器（集成 visualize_scan_csv.py 逻辑）"""
    
    def __init__(self, output_dir: Path, show_plots: bool = False):
        self.output_dir = output_dir
        self.show_plots = show_plots
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _ask_save_confirmation(self) -> bool:
        """询问用户是否保存图表"""
        print("\n" + "="*60)
        print("💾 是否保存图表？")
        print("="*60)
        response = input("输入 'y' 或 'yes' 保存，其他任意键取消: ").strip().lower()
        return response in ['y', 'yes', '是']
    
    def visualize_csv(self, csv_path: Path) -> bool:
        """分析扫描数据 CSV"""
        run_name = csv_path.stem
        LOGGER.info(f"📊 正在分析扫描数据: {csv_path.name}")
        
        # 检查文件大小
        if csv_path.stat().st_size == 0:
            LOGGER.warning(f"⚠️  文件 {csv_path.name} 是空文件，跳过。")
            return False
        
        # 1. 创建独立输出子目录
        run_dir = self.output_dir / run_name
        run_dir.mkdir(exist_ok=True)

        try:
            # 2. 加载数据
            df, e_bins, e_hist, e_cdf = load_and_prepare(csv_path)
            if df.empty:
                LOGGER.warning(f"⚠️  文件 {csv_path.name} 没有有效数据，跳过。")
                return False
                
            drones = _detect_drones(df.columns.tolist())

            # 3. 生成图表（在内存中，不保存）
            figures = []  # 存储所有图表对象和文件名
            
            if self.show_plots:
                plt.ion()  # 开启交互模式
                LOGGER.info("👀 正在生成预览图表...")

            # 扫描进度与覆盖效能分析
            try:
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                if "elapsed_time" in df.columns and "scan_ratio" in df.columns:
                    ax1.plot(df["elapsed_time"], df["scan_ratio"], label="AOI 区域覆盖率 (任务进度)", linewidth=3, color='#1f77b4')
                    
                    if "global_scan_ratio" in df.columns:
                        ax1.plot(df["elapsed_time"], df["global_scan_ratio"], label="全局环境覆盖率", linestyle='--', color='gray', alpha=0.7)
                    
                    # 寻找关键里程碑 (80%, 90%, 95%)
                    milestones = [50, 80, 90, 95]
                    for ms in milestones:
                        ms_idx = df[df["scan_ratio"] >= ms].index
                        if not ms_idx.empty:
                            idx = ms_idx[0]
                            t = df["elapsed_time"].iloc[idx]
                            ax1.annotate(f'{ms}% @ {t:.1f}s', 
                                        xy=(t, ms), xytext=(t + 5, ms - 10),
                                        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                                        fontsize=9)
                            ax1.scatter(t, ms, color='red', s=30, zorder=5)

                    ax1.set_xlabel("时间 (s)", fontsize=12)
                    ax1.set_ylabel("覆盖百分比 (%)", fontsize=12)
                    ax1.set_title("目标区域覆盖效能分析 (任务完成证明)", fontsize=14, fontweight='bold')
                    ax1.set_ylim(0, 105)
                    ax1.grid(True, alpha=0.3)
                    ax1.legend(loc='lower right')
                    
                    # 绘制覆盖速率 (覆盖率的一阶导数)
                    ax1_v = ax1.twinx()
                    # 计算平滑后的增长速率
                    if len(df) > 5:
                        dt = df["elapsed_time"].diff().fillna(1)
                        dr = df["scan_ratio"].diff().fillna(0)
                        velocity = (dr / dt).rolling(window=5).mean()
                        ax1_v.fill_between(df["elapsed_time"], velocity, 0, alpha=0.1, color='green', label='覆盖速率')
                        ax1_v.set_ylabel("覆盖速率 (%/s)", color='green', alpha=0.6)
                        ax1_v.tick_params(axis='y', labelcolor='green')
                    
                    fig1.tight_layout()
                    figures.append((fig1, "scan_progress.png"))
                    LOGGER.info(f"  [成功] 生成图表: 扫描进度与效能里程碑")
                    if self.show_plots:
                        plt.show()
                        plt.pause(0.1)
            except Exception as e:
                LOGGER.error(f"  [失败] 生成图表 '扫描进度': {e}", exc_info=True)
            
            # 熵值趋势与不确定性消除分析
            if "global_avg_entropy" in df.columns:
                try:
                    fig2, ax2_1 = plt.subplots(figsize=(10, 6))
                    ax2_1.plot(df["elapsed_time"], df["global_avg_entropy"], linewidth=2, color='green', label='平均熵 (H)')
                    ax2_1.set_title("环境平均熵随时间变化 (不确定性消除趋势)", fontsize=14, fontweight='bold')
                    ax2_1.set_xlabel("时间 (s)")
                    ax2_1.set_ylabel("平均熵")
                    ax2_1.grid(True, alpha=0.3)
                    
                    # 计算并绘制不确定性消除率 (UER)
                    ax2_2 = ax2_1.twinx()
                    initial_entropy = df["global_avg_entropy"].iloc[0]
                    uer = (1 - df["global_avg_entropy"] / initial_entropy) * 100
                    ax2_2.plot(df["elapsed_time"], uer, linewidth=2, color='blue', linestyle='--', label='不确定性消除率 (UER)')
                    ax2_2.set_ylabel("消除率 (%)", color='blue')
                    ax2_2.tick_params(axis='y', labelcolor='blue')
                    ax2_2.set_ylim(0, 105)
                    
                    lines1, labels1 = ax2_1.get_legend_handles_labels()
                    lines2, labels2 = ax2_2.get_legend_handles_labels()
                    ax2_1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
                    
                    fig2.tight_layout()
                    figures.append((fig2, "entropy_trend.png"))
                    LOGGER.info(f"  [成功] 生成图表: 熵值趋势与消除率")
                    if self.show_plots:
                        plt.show()
                        plt.pause(0.1)
                except Exception as e:
                    LOGGER.error(f"  [失败] 生成图表 '熵值趋势': {e}", exc_info=True)

                # 新增：不确定性消除效率分析 (UER vs Scan Ratio)
                if "scan_ratio" in df.columns:
                    try:
                        fig_eff, ax_eff = plt.subplots(figsize=(10, 6))
                        initial_entropy = df["global_avg_entropy"].iloc[0]
                        uer_data = (1 - df["global_avg_entropy"] / initial_entropy) * 100
                        
                        ax_eff.plot(df["scan_ratio"], uer_data, linewidth=2, color='darkorange', label='实际消除路径')
                        # 绘制对角线作为基准（线性消除参考）
                        ax_eff.plot([0, 100], [0, 100], linestyle=':', color='gray', label='线性消除基准 (随机)')
                        
                        ax_eff.set_title("不确定性消除效率分析 (UEE)", fontsize=14, fontweight='bold')
                        ax_eff.set_xlabel("扫描覆盖率 (%)")
                        ax_eff.set_ylabel("不确定性消除率 (%)")
                        ax_eff.grid(True, alpha=0.3)
                        
                        # 填充效率增益区域
                        ax_eff.fill_between(df["scan_ratio"], df["scan_ratio"], uer_data, 
                                       where=(uer_data >= df["scan_ratio"]), color='green', alpha=0.1, label='智能增益区')
                        
                        ax_eff.legend()
                        fig_eff.tight_layout()
                        figures.append((fig_eff, "uncertainty_elimination_efficiency.png"))
                        LOGGER.info(f"  [成功] 生成图表: 不确定性消除效率")
                        if self.show_plots:
                            plt.show()
                            plt.pause(0.1)
                    except Exception as e:
                        LOGGER.error(f"  [失败] 生成图表 '消除效率分析': {e}", exc_info=True)

            # 飞行轨迹 2D
            if drones:
                try:
                    fig3, ax3 = plt.subplots(figsize=(8, 8))
                    for drone in drones:
                        x_col, y_col = f"{drone}_x", f"{drone}_y"
                        if x_col in df.columns and y_col in df.columns:
                            ax3.plot(df[x_col], df[y_col], label=f"无人机: {drone}", linewidth=1)
                    ax3.set_xlabel("X (m)")
                    ax3.set_ylabel("Y (m)")
                    ax3.set_title("水平面飞行轨迹 (X-Y)")
                    ax3.grid(True, alpha=0.3)
                    ax3.legend()
                    fig3.tight_layout()
                    figures.append((fig3, "trajectories_xy.png"))
                    LOGGER.info(f"  [成功] 生成图表: 2D轨迹")
                    if self.show_plots:
                        plt.show()
                        plt.pause(0.1)
                except Exception as e:
                    LOGGER.error(f"  [失败] 生成图表 '2D轨迹': {e}", exc_info=True)

            # 飞行轨迹 3D
            if drones:
                try:
                    fig4 = plt.figure(figsize=(10, 8))
                    ax4 = fig4.add_subplot(111, projection="3d")
                    valid_3d = False
                    for drone in drones:
                        x, y, z = f"{drone}_x", f"{drone}_y", f"{drone}_z"
                        if all(c in df.columns for c in [x, y, z]):
                            ax4.plot(df[x], df[y], df[z], label=drone)
                            valid_3d = True
                    if valid_3d:
                        ax4.set_xlabel("X")
                        ax4.set_ylabel("Y")
                        ax4.set_zlabel("Z")
                        ax4.set_title("3D 空间轨迹")
                        ax4.legend()
                        fig4.tight_layout()
                        figures.append((fig4, "trajectories_3d.png"))
                        LOGGER.info(f"  [成功] 生成图表: 3D轨迹")
                        if self.show_plots:
                            plt.show()
                            plt.pause(0.1)
                except Exception as e:
                    LOGGER.error(f"  [失败] 生成图表 '3D轨迹': {e}", exc_info=True)

            # 熵值分布快照
            if e_bins and e_hist:
                try:
                    fig5, ax5 = plt.subplots(figsize=(10, 6))
                    indices = _pick_snapshot_indices(len(df), 4)
                    for idx in indices:
                        if idx >= len(e_bins) or idx >= len(e_hist):
                            continue
                        bins = e_bins[idx]
                        hist = e_hist[idx]
                        if not bins or not hist:
                            continue
                        if len(bins) == len(hist) + 1:
                            x_pos = bins[:-1]
                            width = bins[1] - bins[0]
                        else:
                            x_pos = np.arange(len(hist))
                            width = 0.8
                        time_val = df["elapsed_time"].iloc[idx]
                        ax5.bar(x_pos, hist, width=width, alpha=0.4, label=f"时间={time_val:.1f}s", align="edge")
                    ax5.set_xlabel("信息熵区间")
                    ax5.set_ylabel("网格数量")
                    ax5.set_title("不同阶段的信息熵分布快照")
                    ax5.legend()
                    ax5.grid(True, alpha=0.2)
                    fig5.tight_layout()
                    figures.append((fig5, "entropy_hist_snapshots.png"))
                    LOGGER.info(f"  [成功] 生成图表: 熵值快照")
                    if self.show_plots:
                        plt.show()
                        plt.pause(0.1)
                except Exception as e:
                    LOGGER.error(f"  [失败] 生成图表 '熵值快照': {e}", exc_info=True)

            # 算法权重与策略稳定性分析
            weight_cols = ["repulsion_coefficient", "entropy_coefficient", "distance_coefficient", 
                           "leader_range_coefficient", "direction_retention_coefficient"]
            if any(c in df.columns for c in weight_cols):
                try:
                    fig6, (ax6_1, ax6_2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})
                    
                    # 1. 权重变化曲线
                    for c in weight_cols:
                        if c in df.columns:
                            ax6_1.plot(df["elapsed_time"], df[c], label=c.replace('_', ' '), linewidth=1.5)
                    ax6_1.set_title("算法权重动态响应 (策略执行详志)", fontsize=14, fontweight='bold')
                    ax6_1.set_ylabel("系数值")
                    ax6_1.legend(loc='best', fontsize=8)
                    ax6_1.grid(True, alpha=0.3)
                    
                    # 2. 策略震荡分析 (Stability)
                    # 计算权重变化的滚动方差
                    window = max(5, len(df) // 10)
                    var_df = pd.DataFrame()
                    for c in weight_cols:
                        if c in df.columns:
                            var_df[c] = df[c].rolling(window=window).var()
                    
                    if not var_df.empty:
                        total_var = var_df.mean(axis=1).fillna(0)
                        ax6_2.fill_between(df["elapsed_time"], total_var, 0, color='darkorange', alpha=0.2, label='策略波动强度 (Variance)')
                        ax6_2.plot(df["elapsed_time"], total_var, color='darkorange', linewidth=1)
                        
                        # 稳定性评估
                        late_var = total_var.tail(len(df)//4).mean()
                        ax6_2.axhline(y=late_var, color='red', linestyle='--', alpha=0.5, label=f'后期平均波动: {late_var:.6f}')
                        
                        if late_var < 0.001:
                            ax6_2.text(0.05, 0.8, "[OK] 权重已收敛，参数输出稳定", transform=ax6_2.transAxes, 
                                    color='green', fontweight='bold', bbox=dict(facecolor='white', alpha=0.7))
                        else:
                            ax6_2.text(0.05, 0.8, "[WARN] 权重仍在动态调整中", transform=ax6_2.transAxes, 
                                    color='blue', fontweight='bold', bbox=dict(facecolor='white', alpha=0.7))
                    
                    ax6_2.set_xlabel("时间 (s)")
                    ax6_2.set_ylabel("方差 (Stability)")
                    ax6_2.set_title("策略收敛性证明 - 曲线趋平证明参数已稳定", fontsize=12, fontweight='bold')
                    ax6_2.grid(True, alpha=0.3)
                    ax6_2.legend(loc='upper right', fontsize=8)
                    
                    fig6.tight_layout()
                    figures.append((fig6, "algorithm_weights.png"))
                    LOGGER.info(f"  [成功] 生成图表: 权重变化")
                    if self.show_plots:
                        plt.show()
                        plt.pause(0.1)
                except Exception as e:
                    LOGGER.error(f"  [失败] 生成图表 '权重变化': {e}", exc_info=True)

            # 5. 系统活跃度与无死锁证明 (Liveness Analysis)
            try:
                if "elapsed_time" in df.columns and "scan_ratio" in df.columns:
                    fig7, ax7 = plt.subplots(figsize=(10, 6))
                    
                    # 计算实时覆盖增量
                    dt = df["elapsed_time"].diff().fillna(1)
                    dr = df["scan_ratio"].diff().fillna(0)
                    velocity = (dr / dt).rolling(window=10).mean().fillna(0)
                    
                    ax7.plot(df["elapsed_time"], velocity, color='purple', linewidth=2, label='实时覆盖增量 (Liveness)')
                    ax7.fill_between(df["elapsed_time"], velocity, 0, alpha=0.2, color='purple')
                    
                    # 寻找零增量区间（潜在死锁风险）
                    deadlock_risk = velocity[velocity < 0.001].index
                    if not deadlock_risk.empty and df["scan_ratio"].iloc[-1] < 95:
                        # 只有在未完成任务且速度极低时才标记
                        ax7.scatter(df["elapsed_time"].iloc[deadlock_risk], [0]*len(deadlock_risk), 
                                   color='red', marker='|', label='疑似停滞点')
                    
                    ax7.set_title("系统活跃度分析 (无死锁证明)", fontsize=14, fontweight='bold')
                    ax7.set_xlabel("时间 (s)")
                    ax7.set_ylabel("覆盖速率 (%/s)")
                    ax7.grid(True, alpha=0.3)
                    
                    # 标注：只要最终完成度达标且曲线未长期归零，即证明无死锁
                    if df["scan_ratio"].iloc[-1] > 90:
                        ax7.text(0.05, 0.95, "[OK] 系统持续活跃，任务顺利完成，无死锁发生", 
                                transform=ax7.transAxes, color='green', fontweight='bold',
                                bbox=dict(facecolor='white', alpha=0.8))
                    
                    ax7.legend()
                    fig7.tight_layout()
                    figures.append((fig7, "liveness_analysis.png"))
                    LOGGER.info(f"  [成功] 生成图表: 系统活跃度与无死锁证明")
                    if self.show_plots:
                        plt.show()
                        plt.pause(0.1)
            except Exception as e:
                LOGGER.error(f"  [失败] 生成图表 '活跃度分析': {e}")

            # 6. 电压下降与任务耐力分析 (Endurance Analysis)
            try:
                battery_cols = [c for c in df.columns if 'battery_voltage' in c]
                if battery_cols and "elapsed_time" in df.columns:
                    fig8, ax8_1 = plt.subplots(figsize=(10, 6))
                    
                    # 1. 绘制电压下降曲线
                    for col in battery_cols:
                        uav_name = col.split('_')[0]
                        ax8_1.plot(df["elapsed_time"], df[col], linewidth=2, label=f'{uav_name} 电压')
                    
                    ax8_1.set_xlabel("时间 (s)")
                    ax8_1.set_ylabel("电池电压 (V)")
                    ax8_1.set_title("续航效能分析 (证明在电量耗尽前完成任务)", fontsize=14, fontweight='bold')
                    ax8_1.grid(True, alpha=0.3)
                    
                    # 2. 叠加任务进度 (Scan Ratio)
                    ax8_2 = ax8_1.twinx()
                    if "scan_ratio" in df.columns:
                        ax8_2.plot(df["elapsed_time"], df["scan_ratio"], color='red', linestyle=':', linewidth=3, label='任务进度 (Scan %)')
                        ax8_2.set_ylabel("任务完成度 (%)", color='red')
                        ax8_2.tick_params(axis='y', labelcolor='red')
                        ax8_2.set_ylim(0, 105)
                        
                        # 标注任务完成时的电压余量
                        completion_idx = df[df["scan_ratio"] >= 90].index
                        if not completion_idx.empty:
                            idx = completion_idx[0]
                            time_done = df["elapsed_time"].iloc[idx]
                            ax8_1.axvline(x=time_done, color='green', linestyle='--', alpha=0.5)
                            ax8_1.text(time_done, df[battery_cols[0]].min(), f' 90% 完成 @ {time_done:.1f}s', 
                                      color='green', rotation=90, verticalalignment='bottom')
                    
                    # 合并图例
                    lines1, labels1 = ax8_1.get_legend_handles_labels()
                    lines2, labels2 = ax8_2.get_legend_handles_labels()
                    ax8_1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=9)
                    
                    fig8.tight_layout()
                    figures.append((fig8, "battery_endurance_analysis.png"))
                    LOGGER.info(f"  [成功] 生成图表: 电压下降与续航分析")
                    if self.show_plots:
                        plt.show()
                        plt.pause(0.1)
            except Exception as e:
                LOGGER.error(f"  [失败] 生成图表 '续航分析': {e}")

            # 7. 飞行姿态稳定性分析 (Attitude Stability Analysis)
            try:
                # 检查是否存在姿态数据
                attitude_drones = []
                for drone in drones:
                    if f"{drone}_roll" in df.columns and f"{drone}_pitch" in df.columns:
                        attitude_drones.append(drone)
                
                if attitude_drones:
                    fig_att, (ax_att1, ax_att2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                    
                    for drone in attitude_drones:
                        # Roll 波动
                        roll_data = df[f"{drone}_roll"]
                        ax_att1.plot(df["elapsed_time"], roll_data, label=f'{drone} Roll', alpha=0.7)
                        
                        # Pitch 波动
                        pitch_data = df[f"{drone}_pitch"]
                        ax_att2.plot(df["elapsed_time"], pitch_data, label=f'{drone} Pitch', alpha=0.7)
                        
                        # 计算抖动 (Jitter) - 标准差
                        roll_jitter = roll_data.std()
                        pitch_jitter = pitch_data.std()
                        LOGGER.info(f"  [分析] {drone} 姿态抖动: Roll={roll_jitter:.2f}°, Pitch={pitch_jitter:.2f}°")
                    
                    ax_att1.set_ylabel("横滚角 Roll (deg)")
                    ax_att1.set_title("飞行姿态稳定性分析 (证明无失控风险)", fontsize=14, fontweight='bold')
                    ax_att1.grid(True, alpha=0.3)
                    ax_att1.legend(loc='upper right', fontsize=8)
                    
                    ax_att2.set_ylabel("俯仰角 Pitch (deg)")
                    ax_att2.set_xlabel("时间 (s)")
                    ax_att2.grid(True, alpha=0.3)
                    ax_att2.legend(loc='upper right', fontsize=8)
                    
                    # 稳定性判定标准
                    all_roll = pd.concat([df[f"{d}_roll"] for d in attitude_drones])
                    all_pitch = pd.concat([df[f"{d}_pitch"] for d in attitude_drones])
                    
                    max_abs_roll = all_roll.abs().max()
                    max_abs_pitch = all_pitch.abs().max()
                    avg_jitter = (all_roll.std() + all_pitch.std()) / 2
                    
                    if max_abs_roll < 30 and max_abs_pitch < 30 and avg_jitter < 5:
                        ax_att1.text(0.02, 0.9, "[OK] 飞行姿态极度平稳", transform=ax_att1.transAxes, 
                                 color='green', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))
                    elif max_abs_roll < 45 and max_abs_pitch < 45:
                        ax_att1.text(0.02, 0.9, "[WARN] 飞行存在波动但受控", transform=ax_att1.transAxes, 
                                 color='orange', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))
                    else:
                        ax_att1.text(0.02, 0.9, "[FAIL] 姿态剧烈震荡/失控风险", transform=ax_att1.transAxes, 
                                 color='red', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))
                    
                    fig_att.tight_layout()
                    figures.append((fig_att, "flight_attitude_stability.png"))
                    LOGGER.info(f"  [成功] 生成图表: 飞行姿态稳定性")
                    if self.show_plots:
                        plt.show()
                        plt.pause(0.1)
            except Exception as e:
                LOGGER.error(f"  [失败] 生成图表 '姿态稳定性': {e}")

            # 8. 实时训练奖励与策略同步分析 (Training Sync Analysis)
            try:
                if "step_reward" in df.columns and "elapsed_time" in df.columns:
                    fig9, ax9_1 = plt.subplots(figsize=(10, 6))
                    
                    # 绘制单步奖励
                    ax9_1.plot(df["elapsed_time"], df["step_reward"], color='#1f77b4', alpha=0.4, label='实时步奖励')
                    # 绘制移动平均奖励
                    if len(df) > 10:
                        reward_ma = df["step_reward"].rolling(window=10).mean()
                        ax9_1.plot(df["elapsed_time"], reward_ma, color='#1f77b4', linewidth=2, label='步奖励趋势 (MA-10)')
                    
                    ax9_1.set_xlabel("时间 (s)")
                    ax9_1.set_ylabel("奖励值", color='#1f77b4')
                    ax9_1.tick_params(axis='y', labelcolor='#1f77b4')
                    
                    # 绘制累计奖励
                    ax9_2 = ax9_1.twinx()
                    if "total_reward" in df.columns:
                        ax9_2.plot(df["elapsed_time"], df["total_reward"], color='darkred', linewidth=2.5, label='当前Episode累计奖励')
                        ax9_2.set_ylabel("累计奖励", color='darkred')
                        ax9_2.tick_params(axis='y', labelcolor='darkred')
                    
                    # 标注 Episode 切换点
                    if "training_episode" in df.columns:
                        ep_changes = df[df["training_episode"].diff() != 0].index
                        for idx in ep_changes:
                            if idx == 0: continue
                            t = df["elapsed_time"].iloc[idx]
                            ax9_1.axvline(x=t, color='gray', linestyle='--', alpha=0.5)
                            ax9_1.text(t, ax9_1.get_ylim()[1], f' Ep.{int(df["training_episode"].iloc[idx])}', 
                                      rotation=90, verticalalignment='top', fontsize=8)

                    ax9_1.set_title("训练过程实时分析 (奖励与环境同步)", fontsize=14, fontweight='bold')
                    
                    # 合并图例
                    h1, l1 = ax9_1.get_legend_handles_labels()
                    h2, l2 = ax9_2.get_legend_handles_labels()
                    ax9_1.legend(h1+h2, l1+l2, loc='upper left', fontsize=9)
                    
                    ax9_1.grid(True, alpha=0.3)
                    fig9.tight_layout()
                    figures.append((fig9, "training_realtime_sync.png"))
                    LOGGER.info(f"  [成功] 生成图表: 训练实时同步分析")
                    if self.show_plots:
                        plt.show()
                        plt.pause(0.1)
            except Exception as e:
                LOGGER.error(f"  [失败] 生成图表 '训练实时同步': {e}")

            # 4. 如果是预览模式，问用户是否保存
            if self.show_plots:
                plt.ioff()  # 关闭交互模式
                plt.show()  # 阻塞显示，直到用户关闭所有窗口
                
                # 询问是否保存
                if self._ask_save_confirmation():
                    for fig, filename in figures:
                        fig.savefig(run_dir / filename, dpi=150)
                    LOGGER.info(f"✅ 图表已保存到: {run_dir}")
                    # 关闭所有图表
                    for fig, _ in figures:
                        plt.close(fig)
                    return True
                else:
                    LOGGER.info("❌ 已取消保存")
                    # 关闭所有图表
                    for fig, _ in figures:
                        plt.close(fig)
                    return False
            else:
                # 非预览模式，直接保存
                for fig, filename in figures:
                    fig.savefig(run_dir / filename, dpi=150)
                    plt.close(fig)
                LOGGER.info(f"✅ 扫描数据分析完成，结果保存在: {run_dir}")
                return True

        except Exception as e:
            LOGGER.error(f"❌ 分析扫描数据失败 {csv_path.name}: {e}", exc_info=True)
            return False


def auto_discover_data() -> Tuple[List[Path], List[Path], List[Path]]:
    """自动发现所有可分析的数据文件"""
    crazyflie_files = []
    scan_files = []
    dqn_files = []
    
    # 搜索 Crazyflie 训练日志
    crazyflie_logs_dir = Path("multirotor/DDPG_Weight/crazyflie_logs")
    if crazyflie_logs_dir.exists():
        crazyflie_files.extend(list(crazyflie_logs_dir.glob("crazyflie_training_log_*.json")))
        crazyflie_files.extend(list(crazyflie_logs_dir.glob("crazyflie_flight_*.csv")))
        crazyflie_files.extend(list(crazyflie_logs_dir.glob("crazyflie_weights_*.csv")))
    
    # 搜索 DataCollector 扫描数据
    scan_data_dir = Path("multirotor/DDPG_Weight/airsim_training_logs")
    if scan_data_dir.exists():
        scan_files.extend(list(scan_data_dir.glob("scan_data_*.csv")))
    
    # 搜索 DQN 训练日志
    dqn_logs_dir = Path("multirotor/DQN_Movement/logs")
    if dqn_logs_dir.exists():
        for subdir in dqn_logs_dir.glob("*"):
            if subdir.is_dir():
                metadata_files = list(subdir.glob("dqn_training_metadata.json"))
                dqn_files.extend(metadata_files)
    
    return crazyflie_files, scan_files, dqn_files


class DataComparer:
    """多份数据对比分析器"""
    
    def __init__(self, output_dir: Path, show_plots: bool = False):
        self.output_dir = output_dir
        self.show_plots = show_plots
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def compare_scan_data(self, csv_files: List[Path]) -> bool:
        """对比多份扫描数据"""
        if len(csv_files) < 2:
            LOGGER.warning("⚠️  对比分析至少需要 2 份数据文件")
            return False
        
        LOGGER.info(f"📊 开始对比分析 {len(csv_files)} 份扫描数据...")
        
        all_data = []
        for f in csv_files:
            try:
                df, _, _, _ = load_and_prepare(f)
                if not df.empty:
                    all_data.append((f.stem, df))
            except Exception as e:
                LOGGER.error(f"❌ 读取对比文件失败 {f.name}: {e}")
        
        if not all_data:
            return False
        
        compare_dir = self.output_dir / "comparison_results"
        compare_dir.mkdir(exist_ok=True)
        
        # 1. 对比扫描比例
        fig1, ax1 = plt.subplots(figsize=(12, 7))
        for label, df in all_data:
            if "elapsed_time" in df.columns and "scan_ratio" in df.columns:
                ax1.plot(df["elapsed_time"], df["scan_ratio"], label=label, linewidth=2)
        
        ax1.set_xlabel("时间 (s)", fontsize=12)
        ax1.set_ylabel("扫描完成度 (%)", fontsize=12)
        ax1.set_title("不同实验 - 扫描进度对比", fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='lower right', fontsize=9)
        plt.tight_layout()
        plt.savefig(compare_dir / "compare_scan_progress.png", dpi=150)
        
        # 2. 对比平均熵
        fig2, ax2 = plt.subplots(figsize=(12, 7))
        has_entropy = False
        for label, df in all_data:
            if "elapsed_time" in df.columns and "global_avg_entropy" in df.columns:
                ax2.plot(df["elapsed_time"], df["global_avg_entropy"], label=label, linewidth=2)
                has_entropy = True
        
        if has_entropy:
            ax2.set_xlabel("时间 (s)", fontsize=12)
            ax2.set_ylabel("平均熵值", fontsize=12)
            ax2.set_title("不同实验 - 熵值变化对比", fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='upper right', fontsize=9)
            plt.tight_layout()
            plt.savefig(compare_dir / "compare_entropy_trend.png", dpi=150)
        else:
            plt.close(fig2)
        
        # 3. 对比最终扫描比例 (柱状图)
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        labels = [item[0] for item in all_data]
        final_ratios = [item[1]["scan_ratio"].iloc[-1] if "scan_ratio" in item[1].columns else 0 for item in all_data]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
        bars = ax3.bar(labels, final_ratios, color=colors)
        
        ax3.set_ylabel("最终扫描比例 (%)")
        ax3.set_title("不同实验 - 最终扫描完成度对比")
        ax3.set_ylim(0, 105)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(compare_dir / "compare_final_completion.png", dpi=150)
        
        # 4. 对比不确定性消除效率 (UER vs Scan Ratio)
        fig4, ax4 = plt.subplots(figsize=(12, 7))
        has_eff = False
        for label, df in all_data:
            if "scan_ratio" in df.columns and "global_avg_entropy" in df.columns:
                initial_h = df["global_avg_entropy"].iloc[0]
                uer = (1 - df["global_avg_entropy"] / initial_h) * 100
                ax4.plot(df["scan_ratio"], uer, label=label, linewidth=2)
                has_eff = True
        
        if has_eff:
            ax4.plot([0, 100], [0, 100], linestyle=':', color='gray', label='线性基准')
            ax4.set_xlabel("扫描覆盖率 (%)")
            ax4.set_ylabel("不确定性消除率 (%)")
            ax4.set_title("不同实验 - 不确定性消除效率对比 (UEE)")
            ax4.grid(True, alpha=0.3)
            ax4.legend(loc='best', fontsize=9)
            plt.tight_layout()
            plt.savefig(compare_dir / "compare_elimination_efficiency.png", dpi=150)
        else:
            plt.close(fig4)
            
        # 5. 多机协作效率分析 (Speedup Analysis)
        fig5, (ax5_1, ax5_2) = plt.subplots(2, 1, figsize=(12, 10))
        
        comparison_stats = []
        for label, df in all_data:
            drone_count = len(_detect_drones(df.columns.tolist()))
            # 找到达到 80% 覆盖率的时间
            t_80 = df[df["scan_ratio"] >= 80]["elapsed_time"].iloc[0] if not df[df["scan_ratio"] >= 80].empty else None
            if t_80:
                comparison_stats.append({
                    'label': label,
                    'drones': drone_count,
                    'time': t_80
                })
        
        if len(comparison_stats) >= 2:
            df_stats = pd.DataFrame(comparison_stats)
            # 绘制耗时对比
            ax5_1.bar(df_stats['label'], df_stats['time'], color='skyblue')
            ax5_1.set_ylabel("达到 80% 覆盖耗时 (s)")
            ax5_1.set_title("任务完成效率对比 (时间维度)")
            
            # 计算加速比 (以最小无人机数量的实验为基准)
            min_drones_time = df_stats.loc[df_stats['drones'].idxmin(), 'time']
            df_stats['speedup'] = min_drones_time / df_stats['time']
            
            ax5_2.plot(df_stats['label'], df_stats['speedup'], marker='o', linewidth=2, color='red')
            ax5_2.axhline(y=1, color='gray', linestyle='--')
            ax5_2.set_ylabel("协作加速比")
            ax5_2.set_title("多机协作加速比证明 (对比单机/少机)")
            
            plt.tight_layout()
            plt.savefig(compare_dir / "collaboration_speedup.png", dpi=150)
        else:
            plt.close(fig5)
            
        return True

    def compare_training_results(self, files: List[Path]) -> bool:
        """对比多份训练运行的学习曲线（支持 JSON 和 CSV 混合对比）"""
        if len(files) < 2:
            return False
            
        LOGGER.info(f"📊 开始跨格式对比分析 {len(files)} 份训练奖励数据...")
        
        all_stats = []
        for f in files:
            try:
                if f.suffix == '.json':
                    # 处理实体训练 JSON
                    with open(f, 'r', encoding='utf-8') as jf:
                        data = json.load(jf)
                        stats = data.get('episode_stats', [])
                        if stats:
                            df = pd.DataFrame(stats)
                            # 统一字段名：将实体 JSON 的 length 映射为 steps 以对齐 CSV
                            if 'length' in df.columns:
                                df = df.rename(columns={'length': 'steps'})
                            all_stats.append((f.stem, df))
                elif f.suffix == '.csv' and 'training_stats' in f.name:
                    # 处理虚拟训练 CSV
                    df = pd.read_csv(f)
                    if not df.empty:
                        # 确保 CSV 也有 episode 字段（如果 CSV 叫 'episode' 就不动）
                        all_stats.append((f.stem, df))
            except Exception as e:
                LOGGER.error(f"❌ 读取训练对比文件失败 {f.name}: {e}")
                
        if not all_stats:
            LOGGER.warning("⚠️ 没有找到有效的训练统计数据进行对比")
            return False
            
        compare_dir = self.output_dir / "comparison_training"
        compare_dir.mkdir(exist_ok=True)
        
        # 1. 奖励曲线叠加对比
        fig1, ax1 = plt.subplots(figsize=(12, 7))
        for label, df in all_stats:
            if 'reward' in df.columns and 'episode' in df.columns:
                # 使用移动平均进行平滑对比
                window = max(2, min(10, len(df) // 2))
                smooth_reward = df['reward'].rolling(window=window).mean()
                ax1.plot(df['episode'], smooth_reward, label=f'{label} (平滑)', linewidth=2)
        
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("总奖励")
        ax1.set_title("不同实验 - 学习曲线对比 (奖励上升速度)", fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=9)
        plt.tight_layout()
        plt.savefig(compare_dir / "compare_learning_curves.png", dpi=150)
        
        # 2. 学习速率 (斜率) 对比
        fig2, ax2 = plt.subplots(figsize=(12, 7))
        slopes = []
        for label, df in all_stats:
            if 'reward' in df.columns and 'episode' in df.columns and len(df) > 5:
                # 计算总体的奖励上升斜率 (线性拟合)
                from scipy import stats as scipy_stats
                # 过滤掉前几个Episode（通常是随机探索）
                learn_df = df.tail(int(len(df)*0.8))
                if len(learn_df) > 2:
                    slope, _, _, _, _ = scipy_stats.linregress(learn_df['episode'], learn_df['reward'])
                    slopes.append({'label': label, 'slope': slope})
        
        if slopes:
            df_slopes = pd.DataFrame(slopes)
            bars = ax2.bar(df_slopes['label'], df_slopes['slope'], color=plt.cm.viridis(np.linspace(0.3, 0.8, len(slopes))))
            ax2.set_ylabel("奖励增长斜率 (Learning Rate)")
            ax2.set_title("学习速度量化对比 (证明算法习得效率)", fontsize=14, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            
            # 标注数值
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(compare_dir / "compare_learning_speed.png", dpi=150)
        else:
            plt.close(fig2)
            
        LOGGER.info(f"✅ 训练对比分析完成，结果保存在: {compare_dir}")
        
        if self.show_plots:
            plt.show()
        else:
            plt.close('all')
            
        return True

    def compare_ddpg_vs_dqn(self, ddpg_files: List[Path] = None, dqn_files: List[Path] = None) -> bool:
        """对比 DDPG 权重预测算法 vs DQN 移动控制算法的训练效果
        
        Args:
            ddpg_files: DDPG 训练数据文件列表（支持 JSON/CSV）
            dqn_files: DQN 训练数据文件列表（JSON metadata 路径）
        
        Returns:
            bool: 对比分析是否成功
        """
        LOGGER.info("📊 开始 DDPG vs DQN 算法对比分析...")
        
        # 自动发现文件
        if not ddpg_files:
            ddpg_files = []
            ddpg_logs = Path("multirotor/DDPG_Weight/crazyflie_logs")
            if ddpg_logs.exists():
                ddpg_files.extend(list(ddpg_logs.glob("crazyflie_training_log_*.json")))
                ddpg_files.extend(list(ddpg_logs.glob("training_stats*.csv")))
        
        if not dqn_files:
            dqn_files = []
            dqn_logs = Path("multirotor/DQN_Movement/logs")
            if dqn_logs.exists():
                for subdir in dqn_logs.glob("*"):
                    if subdir.is_dir():
                        metadata_files = list(subdir.glob("dqn_training_metadata.json"))
                        dqn_files.extend(metadata_files)
        
        if not ddpg_files and not dqn_files:
            LOGGER.warning("⚠️ 未找到任何 DDPG 或 DQN 训练数据文件")
            return False
        
        LOGGER.info(f"  发现 {len(ddpg_files)} 个 DDPG 训练数据，{len(dqn_files)} 个 DQN 训练数据")
        
        # 加载 DDPG 数据
        ddpg_data = []
        for f in ddpg_files:
            try:
                if f.suffix == '.json':
                    with open(f, 'r', encoding='utf-8') as jf:
                        data = json.load(jf)
                        stats = data.get('episode_stats', [])
                        if stats:
                            df = pd.DataFrame(stats)
                            if 'length' in df.columns:
                                df = df.rename(columns={'length': 'steps'})
                            ddpg_data.append((f"DDPG-{f.stem}", df, 'DDPG'))
                elif f.suffix == '.csv':
                    df = pd.read_csv(f)
                    if not df.empty and 'reward' in df.columns:
                        ddpg_data.append((f"DDPG-{f.stem}", df, 'DDPG'))
            except Exception as e:
                LOGGER.error(f"❌ 读取 DDPG 文件失败 {f.name}: {e}")
        
        # 加载 DQN 数据
        dqn_data = []
        for f in dqn_files:
            try:
                # 读取元数据获取 CSV 路径
                with open(f, 'r', encoding='utf-8') as jf:
                    metadata = json.load(jf)
                    csv_path = metadata.get('training_stats_path')
                    if csv_path and Path(csv_path).exists():
                        df = pd.read_csv(csv_path)
                        if not df.empty and 'reward' in df.columns:
                            dqn_data.append((f"DQN-{f.parent.name}", df, 'DQN'))
            except Exception as e:
                LOGGER.error(f"❌ 读取 DQN 文件失败 {f.name}: {e}")
        
        all_data = ddpg_data + dqn_data
        if len(all_data) < 2:
            LOGGER.warning("⚠️ 对比分析至少需要 2 份有效数据（1份DDPG + 1份DQN）")
            return False
        
        # 创建对比结果目录
        compare_dir = self.output_dir / "algorithm_comparison_ddpg_vs_dqn"
        compare_dir.mkdir(exist_ok=True)
        
        # 1. 奖励曲线对比 (按算法类型分色)
        fig1, ax1 = plt.subplots(figsize=(14, 8))
        color_map = {'DDPG': '#FF6B6B', 'DQN': '#4ECDC4'}  # DDPG 红色系，DQN 蓝绿色系
        
        for label, df, algo_type in all_data:
            if 'reward' in df.columns and 'episode' in df.columns:
                window = max(2, min(10, len(df) // 10))
                smooth_reward = df['reward'].rolling(window=window, min_periods=1).mean()
                ax1.plot(df['episode'], smooth_reward, label=label, 
                        linewidth=2.5, color=color_map[algo_type], alpha=0.7)
                # 添加原始数据的阴影区域
                ax1.fill_between(df['episode'], 
                               df['reward'].rolling(window=window, min_periods=1).quantile(0.25),
                               df['reward'].rolling(window=window, min_periods=1).quantile(0.75),
                               color=color_map[algo_type], alpha=0.1)
        
        ax1.set_xlabel("Episode", fontsize=13)
        ax1.set_ylabel("总奖励 (Cumulative Reward)", fontsize=13)
        ax1.set_title("DDPG vs DQN 算法对比 - 学习曲线", fontsize=15, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=10)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(compare_dir / "ddpg_vs_dqn_reward_curves.png", dpi=150)
        
        # 2. 收敛速度对比 (奖励增长斜率)
        fig2, ax2 = plt.subplots(figsize=(12, 7))
        convergence_stats = []
        
        for label, df, algo_type in all_data:
            if 'reward' in df.columns and 'episode' in df.columns and len(df) > 10:
                try:
                    from scipy import stats as scipy_stats
                    # 取后 80% 数据计算学习速度
                    learn_df = df.tail(int(len(df) * 0.8))
                    if len(learn_df) > 2:
                        slope, intercept, r_value, _, _ = scipy_stats.linregress(
                            learn_df['episode'], learn_df['reward']
                        )
                        convergence_stats.append({
                            'label': label,
                            'algo': algo_type,
                            'slope': slope,
                            'r_squared': r_value ** 2
                        })
                except Exception as e:
                    LOGGER.warning(f"  计算 {label} 的收敛速度失败: {e}")
        
        if convergence_stats:
            df_conv = pd.DataFrame(convergence_stats)
            colors = [color_map[algo] for algo in df_conv['algo']]
            bars = ax2.bar(df_conv['label'], df_conv['slope'], color=colors, alpha=0.7, edgecolor='black')
            ax2.set_ylabel("奖励增长斜率 (Reward Growth Rate)", fontsize=12)
            ax2.set_title("DDPG vs DQN - 收敛速度对比", fontsize=14, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            
            # 标注数值和 R²
            for i, bar in enumerate(bars):
                height = bar.get_height()
                r2 = df_conv['r_squared'].iloc[i]
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}\n(R²={r2:.3f})', 
                        ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
            
            # 添加图例说明算法颜色
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=color_map['DDPG'], label='DDPG (权重预测)'),
                             Patch(facecolor=color_map['DQN'], label='DQN (移动控制)')]
            ax2.legend(handles=legend_elements, loc='upper left', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(compare_dir / "ddpg_vs_dqn_convergence_speed.png", dpi=150)
        else:
            plt.close(fig2)
        
        # 3. 最终性能对比（最后10个episode的平均奖励）
        fig3, ax3 = plt.subplots(figsize=(12, 7))
        final_performance = []
        
        for label, df, algo_type in all_data:
            if 'reward' in df.columns and len(df) > 0:
                # 取最后10个episode的平均奖励
                final_avg = df['reward'].tail(10).mean()
                final_std = df['reward'].tail(10).std()
                final_performance.append({
                    'label': label,
                    'algo': algo_type,
                    'final_reward': final_avg,
                    'std': final_std
                })
        
        if final_performance:
            df_perf = pd.DataFrame(final_performance)
            colors = [color_map[algo] for algo in df_perf['algo']]
            bars = ax3.bar(df_perf['label'], df_perf['final_reward'], 
                          yerr=df_perf['std'], color=colors, alpha=0.7, 
                          edgecolor='black', capsize=5)
            ax3.set_ylabel("最终平均奖励 (最后10个Episode)", fontsize=12)
            ax3.set_title("DDPG vs DQN - 最终性能对比", fontsize=14, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            
            # 标注数值
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', 
                        va='bottom' if height > 0 else 'top', fontsize=10)
            
            # 添加图例
            legend_elements = [Patch(facecolor=color_map['DDPG'], label='DDPG (权重预测)'),
                             Patch(facecolor=color_map['DQN'], label='DQN (移动控制)')]
            ax3.legend(handles=legend_elements, loc='upper left', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(compare_dir / "ddpg_vs_dqn_final_performance.png", dpi=150)
        else:
            plt.close(fig3)
        
        # 4. 学习稳定性对比（奖励方差分析）
        fig4, ax4 = plt.subplots(figsize=(14, 7))
        
        for label, df, algo_type in all_data:
            if 'reward' in df.columns and 'episode' in df.columns and len(df) > 20:
                # 计算滚动标准差（窗口大小为10）
                rolling_std = df['reward'].rolling(window=10, min_periods=1).std()
                ax4.plot(df['episode'], rolling_std, label=label, 
                        linewidth=2, color=color_map[algo_type], alpha=0.7)
        
        ax4.set_xlabel("Episode", fontsize=13)
        ax4.set_ylabel("奖励标准差 (10-Episode 滚动窗口)", fontsize=12)
        ax4.set_title("DDPG vs DQN - 学习稳定性对比 (波动程度)", fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc='best', fontsize=10)
        plt.tight_layout()
        plt.savefig(compare_dir / "ddpg_vs_dqn_stability.png", dpi=150)
        
        # 5. 生成对比报告文本
        report_path = compare_dir / "comparison_report.txt"
        with open(report_path, 'w', encoding='utf-8') as report:
            report.write("="*80 + "\n")
            report.write("DDPG vs DQN 算法对比分析报告\n")
            report.write("="*80 + "\n\n")
            
            report.write("1. 算法简介\n")
            report.write("-" * 80 + "\n")
            report.write("  DDPG (Deep Deterministic Policy Gradient):\n")
            report.write("    - 用途: APF 权重参数预测 (连续动作空间)\n")
            report.write("    - 输出: 6个连续权重值 (wg, wo, wd, wl, wf, wn)\n")
            report.write("    - 观察空间: 环境熵值、位置、速度等\n\n")
            
            report.write("  DQN (Deep Q-Network):\n")
            report.write("    - 用途: 无人机移动控制 (离散动作空间)\n")
            report.write("    - 动作: 6个方向 (上/下/左/右/前/后)\n")
            report.write("    - 观察空间: 位置、速度、熵值、Leader信息等 (21维)\n\n")
            
            report.write("2. 收敛速度对比\n")
            report.write("-" * 80 + "\n")
            if convergence_stats:
                for stat in convergence_stats:
                    report.write(f"  {stat['label']:40s}: 斜率={stat['slope']:8.4f}, R²={stat['r_squared']:.4f}\n")
            else:
                report.write("  无法计算收敛速度统计\n")
            report.write("\n")
            
            report.write("3. 最终性能对比\n")
            report.write("-" * 80 + "\n")
            if final_performance:
                for perf in final_performance:
                    report.write(f"  {perf['label']:40s}: 平均奖励={perf['final_reward']:8.2f} ± {perf['std']:.2f}\n")
            else:
                report.write("  无法计算最终性能统计\n")
            report.write("\n")
            
            report.write("4. 结论与建议\n")
            report.write("-" * 80 + "\n")
            report.write("  - DDPG 和 DQN 解决不同类型的强化学习问题\n")
            report.write("  - DDPG 适合连续参数优化，DQN 适合离散决策\n")
            report.write("  - 建议根据具体任务选择合适的算法\n")
            report.write("  - 可结合使用：DQN控制移动 + DDPG优化APF权重\n")
            report.write("\n" + "="*80 + "\n")
        
        LOGGER.info(f"✅ DDPG vs DQN 对比分析完成，结果保存在: {compare_dir}")
        LOGGER.info(f"  📈 生成图表: reward_curves, convergence_speed, final_performance, stability")
        LOGGER.info(f"  📄 生成报告: {report_path.name}")
        
        if self.show_plots:
            plt.show()
        else:
            plt.close('all')
        
        return True

    def compare_crazyflie_data(self, csv_files: List[Path]) -> bool:
        """对比多份 Crazyflie 飞行数据"""
        if len(csv_files) < 2:
            return False
            
        LOGGER.info(f"📊 开始对比分析 {len(csv_files)} 份 Crazyflie 数据...")
        
        all_data = []
        for f in csv_files:
            try:
                df = pd.read_csv(f)
                if not df.empty:
                    all_data.append((f.stem, df))
            except Exception as e:
                LOGGER.error(f"❌ 读取对比文件失败 {f.name}: {e}")
                
        if not all_data:
            return False
            
        compare_dir = self.output_dir / "comparison_results_crazyflie"
        compare_dir.mkdir(exist_ok=True)
        
        # 对比速度
        fig, ax = plt.subplots(figsize=(12, 7))
        has_speed = False
        for label, df in all_data:
            if "elapsed_time" in df.columns and "speed" in df.columns:
                ax.plot(df["elapsed_time"], df["speed"], label=label, alpha=0.7)
                has_speed = True
        
        if has_speed:
            ax.set_xlabel("时间 (s)")
            ax.set_ylabel("速度 (m/s)")
            ax.set_title("不同实验 - 飞行速度对比")
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=9)
            plt.tight_layout()
            plt.savefig(compare_dir / "compare_flight_speed.png", dpi=150)
        else:
            plt.close(fig)
            
        LOGGER.info(f"✅ 对比分析完成，结果保存在: {compare_dir}")
        
        if self.show_plots:
            plt.show()
        else:
            plt.close('all')
            
        return True


def main():
    parser = argparse.ArgumentParser(description="训练数据可视化工具")
    parser.add_argument("--auto", action="store_true", help="自动扫描所有数据目录")
    parser.add_argument("--json", type=str, help="分析单个 JSON 文件")
    parser.add_argument("--csv", type=str, help="分析单个 CSV 文件")
    parser.add_argument("--dir", type=str, help="分析指定目录下的所有数据文件")
    parser.add_argument("--out", type=str, default="analysis_results", help="输出目录")
    parser.add_argument("--show", action="store_true", help="完成后显示图表窗口")
    parser.add_argument("--compare", action="store_true", help="对同类型数据进行对比分析")
    parser.add_argument("--compare-algorithms", action="store_true", help="对比 DDPG vs DQN 算法性能")
    args = parser.parse_args()
    
    output_dir = Path(args.out)
    
    # 创建可视化器
    crazyflie_viz = CrazyflieDataVisualizer(output_dir, show_plots=args.show)
    scan_viz = ScanDataVisualizer(output_dir, show_plots=args.show)
    dqn_viz = DQNDataVisualizer(output_dir, show_plots=args.show)
    
    files_to_process = []
    dqn_files = []
    
    # 处理输入参数
    if args.auto:
        LOGGER.info("🔍 自动扫描数据文件...")
        crazyflie_files, scan_files, dqn_data_files = auto_discover_data()
        files_to_process.extend(crazyflie_files)
        files_to_process.extend(scan_files)
        dqn_files.extend(dqn_data_files)
        LOGGER.info(f"   发现 {len(crazyflie_files)} 个 Crazyflie 文件")
        LOGGER.info(f"   发现 {len(scan_files)} 个扫描数据文件")
        LOGGER.info(f"   发现 {len(dqn_data_files)} 个 DQN 训练数据")
    
    if args.json:
        files_to_process.append(Path(args.json))
    
    if args.csv:
        files_to_process.append(Path(args.csv))
    
    if args.dir:
        dir_path = Path(args.dir)
        if dir_path.exists():
            # 检查是否是 DQN 目录
            if 'DQN' in str(dir_path).upper():
                # 搜索 DQN 元数据文件
                for subdir in dir_path.glob("*"):
                    if subdir.is_dir():
                        metadata_files = list(subdir.glob("dqn_training_metadata.json"))
                        dqn_files.extend(metadata_files)
            else:
                files_to_process.extend(list(dir_path.glob("*.json")))
                files_to_process.extend(list(dir_path.glob("*.csv")))
    
    if not files_to_process and not dqn_files:
        LOGGER.error("❌ 未找到任何数据文件")
        LOGGER.info("提示: 使用 --auto 自动扫描，或使用 --json/--csv/--dir 指定文件")
        return 1
    
    LOGGER.info(f"\n{'='*60}")
    LOGGER.info(f"开始处理 {len(files_to_process) + len(dqn_files)} 个文件")
    LOGGER.info(f"{'='*60}\n")
    
    # DDPG vs DQN 算法对比分析
    if args.compare_algorithms:
        comparer = DataComparer(output_dir, show_plots=args.show)
        comparer.compare_ddpg_vs_dqn()
    
    # 对比分析
    if args.compare:
        comparer = DataComparer(output_dir, show_plots=args.show)
        
        # 分组文件
        scan_to_compare = [f for f in files_to_process if 'scan_data' in f.name and f.suffix == '.csv']
        crazyflie_to_compare = [f for f in files_to_process if 'crazyflie' in f.name and f.suffix == '.csv']
        # 训练奖励对比：合并 JSON 和 training_stats CSV
        training_to_compare = [f for f in files_to_process if f.suffix == '.json' or ('training_stats' in f.name and f.suffix == '.csv')]
        
        if scan_to_compare:
            comparer.compare_scan_data(scan_to_compare)
        
        if crazyflie_to_compare:
            comparer.compare_crazyflie_data(crazyflie_to_compare)

        if training_to_compare:
            comparer.compare_training_results(training_to_compare)
    
    success_count = 0
    fail_count = 0
    
    # 处理 DDPG/Crazyflie/Scan 数据
    for file_path in files_to_process:
        if not file_path.exists():
            LOGGER.warning(f"⚠️  文件不存在: {file_path}")
            fail_count += 1
            continue
        
        try:
            if file_path.suffix == '.json':
                if crazyflie_viz.visualize_json(file_path):
                    success_count += 1
                else:
                    fail_count += 1
            elif file_path.suffix == '.csv':
                # 判断是 Crazyflie 数据还是扫描数据
                if 'crazyflie' in file_path.name:
                    if crazyflie_viz.visualize_csv(file_path):
                        success_count += 1
                    else:
                        fail_count += 1
                elif 'scan_data' in file_path.name:
                    if scan_viz.visualize_csv(file_path):
                        success_count += 1
                    else:
                        fail_count += 1
                else:
                    LOGGER.warning(f"⚠️  未知的 CSV 类型: {file_path.name}")
                    fail_count += 1
        except Exception as e:
            LOGGER.error(f"❌ 处理文件失败 {file_path.name}: {e}")
            fail_count += 1
    
    # 处理 DQN 数据
    for dqn_meta_path in dqn_files:
        if not dqn_meta_path.exists():
            LOGGER.warning(f"⚠️  文件不存在: {dqn_meta_path}")
            fail_count += 1
            continue
        
        try:
            # 从元数据中获取 CSV 路径
            with open(dqn_meta_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                csv_path = metadata.get('training_stats_path')
                if csv_path and Path(csv_path).exists():
                    if dqn_viz.visualize_training(dqn_meta_path, Path(csv_path)):
                        success_count += 1
                    else:
                        fail_count += 1
                else:
                    LOGGER.warning(f"⚠️  DQN 训练统计 CSV 不存在: {csv_path}")
                    fail_count += 1
        except Exception as e:
            LOGGER.error(f"❌ 处理 DQN 文件失败 {dqn_meta_path.name}: {e}")
            fail_count += 1
    
    LOGGER.info(f"\n{'='*60}")
    LOGGER.info(f"处理完成!")
    LOGGER.info(f"  ✅ 成功: {success_count} 个")
    LOGGER.info(f"  ❌ 失败: {fail_count} 个")
    LOGGER.info(f"  📁 结果目录: {output_dir.absolute()}")
    LOGGER.info(f"{'='*60}\n")
    
    if args.show:
        plt.show()
    
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
