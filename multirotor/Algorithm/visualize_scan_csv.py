"""
离线 CSV 可视化工具（DataCollector 输出数据的深度分析）
功能：
  1. 支持单个或多个 CSV 文件的扫描进度、熵值变化、权重曲线可视化。
  2. 支持 2D/3D 轨迹绘制。
  3. 支持熵值分布直方图与累计分布函数（CDF）的快照分析。
  4. 自动生成多实验对比表和对比图。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import platform

# --- 解决中文显示问题的配置 ---
def set_ch_font():
    # 根据操作系统自动选择常见的中文字体
    system = platform.system()
    if system == "Windows":
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial']
    elif system == "Darwin":  # macOS
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'STHeiti', 'sans-serif']
    else:  # Linux (如 Ubuntu)
        plt.rcParams['font.sans-serif'] = ['Droid Sans Fallback', 'Ubuntu Micro Hei', 'WenQuanYi Micro Hei', 'sans-serif']
    
    # 解决负号 '-' 显示为方块的问题
    plt.rcParams['axes.unicode_minus'] = False 

set_ch_font()

# 配置日志系统，确保错误能输出到控制台
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
LOGGER = logging.getLogger(__name__)

def _parse_percent_series(series: pd.Series) -> pd.Series:
    """
    处理百分比字符串数据。
    例如将 "85.32%" 转换为浮点数 85.32。
    """
    try:
        return (
            series.astype(str)
            .str.replace("%", "", regex=False)
            .replace("nan", np.nan)
            .astype(float)
        )
    except Exception as e:
        LOGGER.error(f"转换百分比字段失败: {e}")
        return series

def _safe_json_list(value: str) -> List[float]:
    """
    安全解析存储在 CSV 单元格中的 JSON 列表（如 [1.2, 0.5, ...]）。
    """
    if pd.isna(value) or value.lower() == "nan" or not value.strip():
        return []
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [float(x) for x in parsed]
    except Exception:
        # 如果解析失败，可能是数据截断或格式错误，返回空列表
        pass
    return []

def _pick_snapshot_indices(n_rows: int, max_snapshots: int) -> List[int]:
    """
    根据总行数，均匀选取指定数量的快照索引点，用于展示不同时间点的状态。
    """
    if n_rows <= 0:
        return []
    if max_snapshots <= 1:
        return [n_rows - 1]
    # 使用 linspace 确保覆盖起点和终点
    return np.linspace(0, n_rows - 1, num=max_snapshots, dtype=int).tolist()

def _detect_drones(columns: List[str]) -> List[str]:
    """
    自动识别 CSV 中的无人机。规则：寻找以 '_x' 结尾的列名。
    """
    drones = set()
    for col in columns:
        if col.endswith("_x"):
            drones.add(col[:-2])
    return sorted(list(drones))

def load_and_prepare(csv_path: Path) -> Tuple[pd.DataFrame, List[List[float]], List[List[float]], List[List[float]]]:
    """
    核心读取函数：加载 CSV 并进行数据清洗。
    返回：清洗后的 DataFrame 以及 熵值相关的三组序列数据。
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到文件: {csv_path}")
    
    # 检查文件是否为空
    if csv_path.stat().st_size == 0:
        LOGGER.warning(f"{csv_path.name} 是空文件，跳过。")
        return pd.DataFrame(), [], [], []
    
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        LOGGER.warning(f"{csv_path.name} 没有数据，跳过。")
        return pd.DataFrame(), [], [], []
    except Exception as e:
        LOGGER.warning(f"读取 CSV 失败: {csv_path.name}, 原因: {e}")
        return pd.DataFrame(), [], [], []

    if df.empty:
        LOGGER.warning(f"{csv_path.name} 数据为空，跳过。")
        return df, [], [], []

    # 1. 转换时间字段。如果缺失，则按行索引生成模拟时间。
    if "elapsed_time" in df.columns:
        df["elapsed_time"] = pd.to_numeric(df["elapsed_time"], errors="coerce")
    else:
        LOGGER.warning(f"{csv_path.name} 缺少 elapsed_time 列，将使用索引作为时间。")
        df["elapsed_time"] = np.arange(len(df), dtype=float)

    # 2. 转换扫描比例字段（处理百分比符号）
    for col in ["scan_ratio", "global_scan_ratio"]:
        if col in df.columns:
            df[col] = _parse_percent_series(df[col])

    # 3. 解析复杂的 JSON 列表字段（熵值分布统计）
    entropy_bins = []
    entropy_hist = []
    entropy_cdf = []
    
    # 提取这些列是为了避免在循环中重复解析
    if "entropy_bins" in df.columns:
        entropy_bins = [_safe_json_list(str(v)) for v in df["entropy_bins"]]
    if "entropy_hist" in df.columns:
        entropy_hist = [_safe_json_list(str(v)) for v in df["entropy_hist"]]
    if "entropy_cdf" in df.columns:
        entropy_cdf = [_safe_json_list(str(v)) for v in df["entropy_cdf"]]

    return df, entropy_bins, entropy_hist, entropy_cdf

# --- 绘图函数部分 ---

def plot_scan_progress(df: pd.DataFrame, out_dir: Path, close_fig: bool = True) -> None:
    """绘制扫描完成度随时间变化的曲线。"""
    if "elapsed_time" not in df.columns or "scan_ratio" not in df.columns:
        LOGGER.warning("缺失扫描比例相关列，跳过 plot_scan_progress")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["elapsed_time"], df["scan_ratio"], label="AOI 区域扫描比例", linewidth=2)
    if "global_scan_ratio" in df.columns:
        ax.plot(df["elapsed_time"], df["global_scan_ratio"], label="全局扫描比例", linestyle="--")
    
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("完成度 (%)")
    ax.set_title("扫描进度曲线")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "scan_progress.png", dpi=150)
    if close_fig:
        plt.close(fig)

def plot_trajectories(df: pd.DataFrame, drones: List[str], out_dir: Path, close_fig: bool = True) -> None:
    """绘制无人机的 2D 和 3D 飞行轨迹。"""
    if not drones:
        return

    # 2D 轨迹绘制
    fig, ax = plt.subplots(figsize=(8, 8))
    for drone in drones:
        x_col, y_col = f"{drone}_x", f"{drone}_y"
        if x_col in df.columns and y_col in df.columns:
            ax.plot(df[x_col], df[y_col], label=f"无人机: {drone}", linewidth=1)
    
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("水平面飞行轨迹 (X-Y)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(out_dir / "trajectories_xy.png", dpi=150)
    if close_fig:
        plt.close(fig)

    # 3D 轨迹绘制
    try:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        valid_3d = False
        for drone in drones:
            x, y, z = f"{drone}_x", f"{drone}_y", f"{drone}_z"
            if all(c in df.columns for c in [x, y, z]):
                ax.plot(df[x], df[y], df[z], label=drone)
                valid_3d = True
        
        if valid_3d:
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title("3D 空间轨迹")
            ax.legend()
            fig.savefig(out_dir / "trajectories_3d.png", dpi=150)
    except Exception as e:
        LOGGER.error(f"3D 轨迹绘图失败: {e}")
    finally:
        if close_fig:
            plt.close(fig)

def plot_entropy_snapshots(
    df: pd.DataFrame, entropy_bins: List[List[float]], entropy_hist: List[List[float]], 
    out_dir: Path, max_snapshots: int, close_fig: bool = True
) -> None:
    """绘制信息熵分布直方图的快照。"""
    if not entropy_bins or not entropy_hist:
        return

    n_rows = len(df)
    indices = _pick_snapshot_indices(n_rows, max_snapshots)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx in indices:
        if idx >= len(entropy_bins) or idx >= len(entropy_hist):
            continue
        bins = entropy_bins[idx]
        hist = entropy_hist[idx]
        if not bins or not hist:
            continue

        # 处理分箱边缘：通常 bins 比 hist 多一个元素
        if len(bins) == len(hist) + 1:
            x_pos = bins[:-1]
            width = bins[1] - bins[0]
        else:
            x_pos = np.arange(len(hist))
            width = 0.8

        time_val = df["elapsed_time"].iloc[idx]
        ax.bar(x_pos, hist, width=width, alpha=0.4, label=f"时间={time_val:.1f}s", align="edge")

    ax.set_xlabel("信息熵区间")
    ax.set_ylabel("网格数量")
    ax.set_title("不同阶段的信息熵分布快照")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.savefig(out_dir / "entropy_hist_snapshots.png", dpi=150)
    if close_fig:
        plt.close(fig)

# --- 工具与包装器 ---

def _safe_plot_wrapper(plot_fn, *args, plot_name: str, run_name: str):
    """
    绘图函数的安全包装器。
    如果某个绘图函数崩溃，它会记录错误但允许程序继续处理其他图表或文件。
    """
    try:
        plot_fn(*args)
        LOGGER.info(f"  [成功] 绘制图表: {plot_name}")
    except Exception as e:
        LOGGER.error(f"  [失败] 绘制图表 '{plot_name}' (运行: {run_name}): {e}", exc_info=True)

def main():
    parser = argparse.ArgumentParser(description="AirSim 扫描数据 CSV 可视化工具")
    parser.add_argument("--csv", help="单个 CSV 文件路径")
    parser.add_argument("--csv-dir", help="包含多个 CSV 的目录")
    parser.add_argument("--out", default="analysis_results", help="结果输出目录")
    parser.add_argument("--snapshots", type=int, default=4, help="熵值快照数量")
    parser.add_argument("--show", action="store_true", help="绘图完成后是否弹出窗口显示（注意：批量处理时不建议开启）")
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # 解析输入路径
    input_files = []
    if args.csv:
        input_files.append(Path(args.csv))
    if args.csv_dir:
        dir_path = Path(args.csv_dir)
        if dir_path.exists():
            # 匹配 scan_data_ 开头的 CSV 文件
            input_files.extend(list(dir_path.glob("scan_data_*.csv")))
    
    if not input_files:
        LOGGER.error("未找到有效的 CSV 输入文件。请检查 --csv 或 --csv-dir 参数。")
        return

    LOGGER.info(f"开始处理，共发现 {len(input_files)} 个文件。")

    for csv_p in input_files:
        run_name = csv_p.stem
        LOGGER.info(f"正在处理文件: {csv_p.name}")
        
        # 1. 创建该文件的独立输出子目录
        run_dir = out_root / run_name
        run_dir.mkdir(exist_ok=True)

        try:
            # 2. 加载数据
            df, e_bins, e_hist, e_cdf = load_and_prepare(csv_p)
            if df.empty:
                LOGGER.warning(f"文件 {csv_p.name} 为空，跳过。")
                continue
                
            drones = _detect_drones(df.columns.tolist())

            # 3. 逐个执行绘图任务（带异常保护）
            _safe_plot_wrapper(plot_scan_progress, df, run_dir, plot_name="扫描进度", run_name=run_name)
            
            if "global_avg_entropy" in df.columns:
                # 绘制熵值变化曲线
                def plot_entropy_curve(d, r):
                    fig, ax = plt.subplots()
                    ax.plot(d["elapsed_time"], d["global_avg_entropy"])
                    ax.set_title("平均熵随时间变化")
                    fig.savefig(r / "entropy_trend.png")
                    plt.close(fig)
                _safe_plot_wrapper(plot_entropy_curve, df, run_dir, plot_name="熵值趋势", run_name=run_name)

            _safe_plot_wrapper(plot_trajectories, df, drones, run_dir, plot_name="飞行轨迹", run_name=run_name)
            
            _safe_plot_wrapper(plot_entropy_snapshots, df, e_bins, e_hist, run_dir, args.snapshots, 
                               plot_name="熵值分布快照", run_name=run_name)

            # 绘制算法权重（如果有这些列）
            weight_cols = ["repulsion_coefficient", "entropy_coefficient", "distance_coefficient"]
            if any(c in df.columns for c in weight_cols):
                def plot_weights_local(d, r):
                    fig, ax = plt.subplots(figsize=(10, 5))
                    for c in weight_cols:
                        if c in d.columns:
                            ax.plot(d["elapsed_time"], d[c], label=c.split('_')[0])
                    ax.set_title("自适应权重变化")
                    ax.legend()
                    fig.savefig(r / "algorithm_weights.png")
                    plt.close(fig)
                _safe_plot_wrapper(plot_weights_local, df, run_dir, plot_name="权重变化", run_name=run_name)

        except Exception as e:
            LOGGER.error(f"处理文件 {csv_p.name} 时发生严重错误: {e}", exc_info=True)

    LOGGER.info(f"所有任务处理完成。结果保存在: {out_root.absolute()}")
    
    if args.show:
        plt.show()

if __name__ == "__main__":
    main()