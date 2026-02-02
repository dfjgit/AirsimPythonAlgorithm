"""
多算法训练数据统一对比分析工具 (Unified Training Analyzer)
设计目标：基于 DataCollector 产出的标准化 CSV 数据，实现跨算法、跨场景的自动对比。
无需为新算法编写新代码，只需在训练脚本中通过 set_experiment_meta 设置标签即可。
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TrainingAnalyzer")

class UnifiedTrainingAnalyzer:
    # 算法 ID 到友好名称的映射
    ALGO_NAME_MAP = {
        'hrl_dqn_apf': '双层融合训练 (HRL+APF)',
        'pure_dqn': '纯 DQN 移动控制',
        'ddpg_apf': 'DDPG 权重自适应 (APF)',
        'unknown': '未标记算法 (历史数据)'
    }

    # 指标 ID 到友好名称的映射
    METRIC_NAME_MAP = {
        'reward': '累计奖励',
        'scan_efficiency': '扫描效率 (Cell/Step)',
        'scan_ratio': '扫描完成度 (%)',
        'global_avg_entropy': '全局平均熵',
        'episode': '训练轮次 (Episode)',
        'elapsed_time': '运行时间 (秒)'
    }

    def __init__(self, output_dir: str = "analysis_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.runs = []  # 存储所有加载的实验数据
        
        # 统一字体配置 (解决中文显示)
        self._setup_plotting_style()

    def _setup_plotting_style(self):
        sns.set_theme(style="whitegrid")
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False

    def load_data(self, log_dirs: List[str]):
        """
        扫描多个目录，加载所有符合格式的 CSV 文件。
        支持 training CSV 和 scan_data CSV。
        """
        for d in log_dirs:
            p = Path(d)
            if not p.exists():
                logger.warning(f"目录不存在: {d}")
                continue
            
            # 查找所有 CSV
            for csv_file in p.glob("*.csv"):
                try:
                    df = pd.read_csv(csv_file)
                    if df.empty: continue
                    
                    # 尝试从数据中识别算法标签
                    algo = df['algorithm_type'].iloc[0] if 'algorithm_type' in df.columns else None
                    
                    # 如果数据中没有标签，尝试从路径猜测
                    if not algo or pd.isna(algo):
                        path_str = str(csv_file).lower()
                        if 'hrl' in path_str or 'hierarchical' in path_str:
                            algo = 'hrl_dqn_apf'
                        elif 'dqn' in path_str:
                            algo = 'pure_dqn'
                        elif 'ddpg' in path_str:
                            algo = 'ddpg_apf'
                        else:
                            algo = 'unknown'
                    
                    env = df['env_type'].iloc[0] if 'env_type' in df.columns else "unknown"
                    
                    # 记录该次运行的元数据
                    run_info = {
                        'file': csv_file,
                        'name': csv_file.stem,
                        'algorithm': algo,
                        'env': env,
                        'data': df,
                        'type': 'training' if 'training' in csv_file.name else 'scan'
                    }
                    self.runs.append(run_info)
                    logger.info(f"已加载: {csv_file.name} (算法: {algo}, 类型: {run_info['type']})")
                except Exception as e:
                    logger.error(f"加载失败 {csv_file.name}: {e}")

    def plot_comparison(self, metric: str, data_type: str = 'training', x_axis: str = 'episode'):
        """
        对比不同算法在特定指标上的表现。
        """
        target_runs = [r for r in self.runs if r['type'] == data_type]
        if not target_runs:
            logger.warning(f"没有找到类型为 {data_type} 的数据")
            return

        plt.figure(figsize=(14, 8))
        
        # 获取中文友好名称
        metric_zh = self.METRIC_NAME_MAP.get(metric, metric)
        x_axis_zh = self.METRIC_NAME_MAP.get(x_axis, x_axis)
        
        # 按算法分组绘图
        unique_algos = sorted(set(r['algorithm'] for r in target_runs))
        
        for algo_id in unique_algos:
            algo_dfs = [r['data'] for r in target_runs if r['algorithm'] == algo_id]
            
            # 合并该算法的所有运行数据
            all_data = pd.concat(algo_dfs)
            
            # 获取显示名称
            display_name = self.ALGO_NAME_MAP.get(algo_id, algo_id)
            
            if x_axis in all_data.columns and metric in all_data.columns:
                # 绘制均值线和标准差填充区域
                sns.lineplot(
                    data=all_data, 
                    x=x_axis, 
                    y=metric, 
                    label=display_name, 
                    errorbar='sd',
                    linewidth=2.5
                )

        plt.title(f"多算法对比分析: {metric_zh} 随 {x_axis_zh} 变化趋势", fontsize=16, pad=20)
        plt.xlabel(x_axis_zh, fontsize=12)
        plt.ylabel(metric_zh, fontsize=12)
        
        # 优化图例：放在图外右侧，避免遮挡曲线
        plt.legend(title="算法类型", title_fontsize='13', fontsize='11', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        filename = self.output_dir / f"comparison_{data_type}_{metric}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        logger.info(f"对比图表已保存: {filename} (标签: {unique_algos})")
        plt.close()

    def generate_summary_report(self):
        """
        生成综合对比报告
        """
        summary_data = []
        for r in self.runs:
            df = r['data']
            algo_display = self.ALGO_NAME_MAP.get(r['algorithm'], r['algorithm'])
            
            if r['type'] == 'training':
                summary_data.append({
                    '算法名称': algo_display,
                    '运行记录': r['name'],
                    '平均奖励': df['reward'].mean() if 'reward' in df.columns else 0,
                    '最高奖励': df['reward'].max() if 'reward' in df.columns else 0,
                    '训练轮次': len(df),
                    '最终效率': df['scan_efficiency'].iloc[-1] if 'scan_efficiency' in df.columns else 0
                })
            elif r['type'] == 'scan':
                summary_data.append({
                    '算法名称': algo_display,
                    '运行记录': r['name'],
                    '最终扫描率(%)': df['scan_ratio'].iloc[-1] if 'scan_ratio' in df.columns else 0,
                    '最低熵值': df['global_avg_entropy'].min() if 'global_avg_entropy' in df.columns else 0,
                    '总耗时(s)': df['elapsed_time'].iloc[-1] if 'elapsed_time' in df.columns else 0
                })

        summary_df = pd.DataFrame(summary_data)
        if not summary_df.empty:
            # 按算法聚合看均值
            algo_comparison = summary_df.groupby('算法名称').mean(numeric_only=True)
            print("\n" + "="*70)
            print("🚀 多算法平均性能量化对比报告 (Averaged Performance Report)")
            print("="*70)
            print(algo_comparison)
            print("="*70)
            
            report_file = self.output_dir / "algorithm_comparison_report.csv"
            # 导出带中文表头的 CSV，并使用 UTF-8 SIG 确保 Excel 打开不乱码
            algo_comparison.to_csv(report_file, encoding='utf-8-sig')
            logger.info(f"对比报告已导出: {report_file}")
        else:
            logger.warning("没有可用于生成报告的数据")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="多算法对比分析工具")
    parser.add_argument("--dirs", nargs='+', help="数据目录列表", default=[
        "multirotor/DDPG_Weight/airsim_training_logs",
        "multirotor/DQN_Movement/logs/dqn_scan_data"
    ])
    parser.add_argument("--out", default="multirotor/Algorithm/analysis_results", help="结果保存目录")
    args = parser.parse_args()

    analyzer = UnifiedTrainingAnalyzer(output_dir=args.out)
    analyzer.load_data(args.dirs)
    
    # 绘制关键指标对比
    # 1. 训练奖励对比 (DQN vs DDPG vs HRL)
    analyzer.plot_comparison(metric='reward', data_type='training', x_axis='episode')
    
    # 2. 扫描效率对比
    analyzer.plot_comparison(metric='scan_efficiency', data_type='training', x_axis='episode')
    
    # 3. 实时扫描比例对比 (随时间变化)
    analyzer.plot_comparison(metric='scan_ratio', data_type='scan', x_axis='elapsed_time')
    
    # 4. 熵下降速度对比
    analyzer.plot_comparison(metric='global_avg_entropy', data_type='scan', x_axis='elapsed_time')
    
    # 生成报告
    analyzer.generate_summary_report()
