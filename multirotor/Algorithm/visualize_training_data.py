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
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        for col in weight_cols:
            if col in df.columns:
                ax.plot(df['step'], df[col], label=weight_names.get(col, col), 
                       linewidth=2, alpha=0.8, marker='o', markersize=3)
        
        ax.set_xlabel('训练步数', fontsize=12)
        ax.set_ylabel('系数值', fontsize=12)
        ax.set_title('APF 权重系数变化历史', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
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
        """绘制 Episode 统计信息"""
        df = pd.DataFrame(episode_stats)
        
        if df.empty:
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # 奖励曲线
        if 'reward' in df.columns and 'episode' in df.columns:
            axes[0].plot(df['episode'], df['reward'], linewidth=2, marker='o', markersize=4)
            axes[0].set_xlabel('Episode', fontsize=12)
            axes[0].set_ylabel('总奖励', fontsize=12)
            axes[0].set_title('Episode 奖励曲线', fontsize=14, fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            
            # 添加移动平均线
            if len(df) > 5:
                window = min(10, len(df) // 2)
                moving_avg = df['reward'].rolling(window=window).mean()
                axes[0].plot(df['episode'], moving_avg, linewidth=3, alpha=0.6, 
                           label=f'{window}-Episode 移动平均', color='red')
                axes[0].legend()
        
        # Episode 长度
        if 'length' in df.columns and 'episode' in df.columns:
            axes[1].plot(df['episode'], df['length'], linewidth=2, marker='s', 
                        markersize=4, color='orange')
            axes[1].set_xlabel('Episode', fontsize=12)
            axes[1].set_ylabel('步数', fontsize=12)
            axes[1].set_title('Episode 长度变化', fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
        
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

            # 扫描进度
            try:
                fig1, ax1 = plt.subplots(figsize=(10, 5))
                if "elapsed_time" in df.columns and "scan_ratio" in df.columns:
                    ax1.plot(df["elapsed_time"], df["scan_ratio"], label="AOI 区域扫描比例", linewidth=2)
                    if "global_scan_ratio" in df.columns:
                        ax1.plot(df["elapsed_time"], df["global_scan_ratio"], label="全局扫描比例", linestyle="--")
                    ax1.set_xlabel("时间 (s)")
                    ax1.set_ylabel("完成度 (%)")
                    ax1.set_title("扫描进度曲线")
                    ax1.grid(True, alpha=0.3)
                    ax1.legend()
                    fig1.tight_layout()
                    figures.append((fig1, "scan_progress.png"))
                    LOGGER.info(f"  [成功] 生成图表: 扫描进度")
                    if self.show_plots:
                        plt.show()
                        plt.pause(0.1)
            except Exception as e:
                LOGGER.error(f"  [失败] 生成图表 '扫描进度': {e}", exc_info=True)
            
            # 熵值趋势
            if "global_avg_entropy" in df.columns:
                try:
                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                    ax2.plot(df["elapsed_time"], df["global_avg_entropy"], linewidth=2, color='green')
                    ax2.set_title("AOI 平均熵随时间变化", fontsize=14, fontweight='bold')
                    ax2.set_xlabel("时间 (s)")
                    ax2.set_ylabel("平均熵")
                    ax2.grid(True, alpha=0.3)
                    fig2.tight_layout()
                    figures.append((fig2, "entropy_trend.png"))
                    LOGGER.info(f"  [成功] 生成图表: 熵值趋势")
                    if self.show_plots:
                        plt.show()
                        plt.pause(0.1)
                except Exception as e:
                    LOGGER.error(f"  [失败] 生成图表 '熵值趋势': {e}", exc_info=True)

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

            # 算法权重
            weight_cols = ["repulsion_coefficient", "entropy_coefficient", "distance_coefficient"]
            if any(c in df.columns for c in weight_cols):
                try:
                    fig6, ax6 = plt.subplots(figsize=(10, 5))
                    for c in weight_cols:
                        if c in df.columns:
                            ax6.plot(df["elapsed_time"], df[c], label=c.replace('_', ' '))
                    ax6.set_title("算法自适应权重变化", fontsize=14, fontweight='bold')
                    ax6.set_xlabel("时间 (s)")
                    ax6.set_ylabel("系数值")
                    ax6.legend()
                    ax6.grid(True, alpha=0.3)
                    fig6.tight_layout()
                    figures.append((fig6, "algorithm_weights.png"))
                    LOGGER.info(f"  [成功] 生成图表: 权重变化")
                    if self.show_plots:
                        plt.show()
                        plt.pause(0.1)
                except Exception as e:
                    LOGGER.error(f"  [失败] 生成图表 '权重变化': {e}", exc_info=True)

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


def auto_discover_data() -> Tuple[List[Path], List[Path]]:
    """自动发现所有可分析的数据文件"""
    crazyflie_files = []
    scan_files = []
    
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
    
    return crazyflie_files, scan_files


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
            
        LOGGER.info(f"✅ 对比分析完成，结果保存在: {compare_dir}")
        
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
    args = parser.parse_args()
    
    output_dir = Path(args.out)
    
    # 创建可视化器
    crazyflie_viz = CrazyflieDataVisualizer(output_dir, show_plots=args.show)
    scan_viz = ScanDataVisualizer(output_dir, show_plots=args.show)
    
    files_to_process = []
    
    # 处理输入参数
    if args.auto:
        LOGGER.info("🔍 自动扫描数据文件...")
        crazyflie_files, scan_files = auto_discover_data()
        files_to_process.extend(crazyflie_files)
        files_to_process.extend(scan_files)
        LOGGER.info(f"   发现 {len(crazyflie_files)} 个 Crazyflie 文件")
        LOGGER.info(f"   发现 {len(scan_files)} 个扫描数据文件")
    
    if args.json:
        files_to_process.append(Path(args.json))
    
    if args.csv:
        files_to_process.append(Path(args.csv))
    
    if args.dir:
        dir_path = Path(args.dir)
        if dir_path.exists():
            files_to_process.extend(list(dir_path.glob("*.json")))
            files_to_process.extend(list(dir_path.glob("*.csv")))
    
    if not files_to_process:
        LOGGER.error("❌ 未找到任何数据文件")
        LOGGER.info("提示: 使用 --auto 自动扫描，或使用 --json/--csv/--dir 指定文件")
        return 1
    
    LOGGER.info(f"\n{'='*60}")
    LOGGER.info(f"开始处理 {len(files_to_process)} 个文件")
    LOGGER.info(f"{'='*60}\n")
    
    # 对比分析
    if args.compare:
        comparer = DataComparer(output_dir, show_plots=args.show)
        
        # 分组文件
        scan_to_compare = [f for f in files_to_process if 'scan_data' in f.name and f.suffix == '.csv']
        crazyflie_to_compare = [f for f in files_to_process if 'crazyflie' in f.name and f.suffix == '.csv']
        
        if scan_to_compare:
            comparer.compare_scan_data(scan_to_compare)
        
        if crazyflie_to_compare:
            comparer.compare_crazyflie_data(crazyflie_to_compare)
            
    success_count = 0
    fail_count = 0
    
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
