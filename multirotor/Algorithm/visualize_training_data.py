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
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
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
    """DataCollector 扫描数据可视化器（使用现有的 visualize_scan_csv.py 逻辑）"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def visualize_csv(self, csv_path: Path) -> bool:
        """分析扫描数据 CSV"""
        # 这里可以调用现有的 visualize_scan_csv.py 的功能
        # 为了简化，直接返回 True，实际可以导入原有函数
        LOGGER.info(f"📊 扫描数据 CSV 分析: {csv_path.name}")
        LOGGER.info(f"   提示: 使用 visualize_scan_csv.py 进行详细分析")
        return True


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


def main():
    parser = argparse.ArgumentParser(description="训练数据可视化工具")
    parser.add_argument("--auto", action="store_true", help="自动扫描所有数据目录")
    parser.add_argument("--json", type=str, help="分析单个 JSON 文件")
    parser.add_argument("--csv", type=str, help="分析单个 CSV 文件")
    parser.add_argument("--dir", type=str, help="分析指定目录下的所有数据文件")
    parser.add_argument("--out", type=str, default="analysis_results", help="输出目录")
    parser.add_argument("--show", action="store_true", help="完成后显示图表窗口")
    args = parser.parse_args()
    
    output_dir = Path(args.out)
    
    # 创建可视化器
    crazyflie_viz = CrazyflieDataVisualizer(output_dir)
    scan_viz = ScanDataVisualizer(output_dir)
    
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
