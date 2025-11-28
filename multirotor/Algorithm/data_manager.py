import os
import time
import csv
import json
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import platform
from typing import Optional
from .HexGridDataModel import HexGridDataModel

class DataCollector:
    """数据采集逻辑核心"""
    def __init__(self, output_dir="data_logs"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        self.is_recording = False
        self.start_time = 0
        self.records = []
        self.lock = threading.Lock()
        
        # 实时统计数据（用于UI显示）
        self.current_stats = {
            "time": 0.0,
            "unscanned": 0, # 剩余工作量 (Sum of 1s)
            "scanned": 0,   # 已完成 (Sum of 0s)
            "coverage": 0.0,
            "total": 0
        }

    def start_recording(self):
        self.records = []
        self.start_time = time.time()
        self.is_recording = True
        return True

    def stop_recording(self):
        if not self.is_recording:
            return None
        self.is_recording = False
        filepath = self.save_to_file()
        return filepath

    def process_step(self, grid_data: Optional[HexGridDataModel]):
        """处理单步数据"""
        # 即使不记录，也计算统计数据用于UI实时显示
        if not grid_data or not hasattr(grid_data, 'cells'):
            return

        total_cells = len(grid_data.cells)
        if total_cells == 0:
            return

        # 逻辑：未被侦察(entropy >= 30) = 1, 被侦察(entropy < 30) = 0
        # 统计数值 = 所有栅格的值之和 (即未被侦察的数量)
        unscanned_count = 0
        scanned_count = 0
        
        for cell in grid_data.cells:
            try:
                if cell.entropy < 30:
                    scanned_count += 1
                else:
                    unscanned_count += 1
            except Exception:
                # 兼容性：如果cell没有entropy属性，跳过
                continue
        
        # AOI区域统计数值 (未侦察的总和)
        aoi_sum_value = unscanned_count 
        coverage = (scanned_count / total_cells * 100) if total_cells > 0 else 0

        # 更新实时状态
        current_time = time.time()
        elapsed = current_time - self.start_time if self.is_recording else 0
        
        with self.lock:
            self.current_stats = {
                "time": elapsed,
                "unscanned": unscanned_count,
                "scanned": scanned_count,
                "coverage": coverage,
                "total": total_cells
            }
            
            # 如果正在记录，存入内存
            if self.is_recording:
                self.records.append({
                    "timestamp": round(elapsed, 2),
                    "aoi_sum_value": aoi_sum_value, # 核心指标：剩余未侦察量
                    "scanned_count": scanned_count,
                    "total_cells": total_cells,
                    "coverage_ratio": round(coverage, 2)
                })

    def save_to_file(self):
        """保存数据到CSV"""
        if not self.records:
            return None

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"mission_data_{timestamp}.csv"
        filepath = os.path.join(self.output_dir, filename)

        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # 写入表头
                writer.writerow(["Time(s)", "AOI Value Sum (Unscanned)", "Scanned Cells", "Total Cells", "Coverage(%)"])
                # 写入内容
                for r in self.records:
                    writer.writerow([
                        r["timestamp"],
                        r["aoi_sum_value"],
                        r["scanned_count"],
                        r["total_cells"],
                        r["coverage_ratio"]
                    ])
            return filepath
        except Exception as e:
            print(f"保存失败: {e}")
            return None

class DataManagerWindow:
    """独立的Tkinter管理窗口"""
    def __init__(self, collector: DataCollector):
        self.collector = collector
        self.root = None
        self.is_running = False

    def start(self):
        """在独立线程启动GUI"""
        self.thread = threading.Thread(target=self._run_gui, daemon=True)
        self.thread.start()

    def _run_gui(self):
        self.root = tk.Tk()
        self.root.title("数据采集管理中心")
        self.root.geometry("500x600")
        try:
            # Keep window on top for convenience
            self.root.attributes("-topmost", 1)
        except Exception:
            pass
        
        # 1. 状态面板
        status_frame = tk.LabelFrame(self.root, text="实时监控状态", padx=10, pady=10)
        status_frame.pack(fill="x", padx=10, pady=5)
        
        self.lbl_status = tk.Label(status_frame, text="状态: 待机", fg="gray", font=("Arial", 12, "bold"))
        self.lbl_status.pack(anchor="w")
        
        self.lbl_time = tk.Label(status_frame, text="记录时长: 0.0s", font=("Arial", 10))
        self.lbl_time.pack(anchor="w")
        
        # 核心数据显示
        data_grid = tk.Frame(status_frame)
        data_grid.pack(fill="x", pady=5)
        
        tk.Label(data_grid, text="AOI统计数值(未侦察):", font=("Arial", 11)).grid(row=0, column=0, sticky="w")
        self.lbl_val_sum = tk.Label(data_grid, text="0", font=("Arial", 14, "bold"), fg="red")
        self.lbl_val_sum.grid(row=0, column=1, sticky="e", padx=20)
        
        tk.Label(data_grid, text="覆盖率:", font=("Arial", 11)).grid(row=1, column=0, sticky="w")
        self.lbl_coverage = tk.Label(data_grid, text="0.0%", font=("Arial", 14, "bold"), fg="green")
        self.lbl_coverage.grid(row=1, column=1, sticky="e", padx=20)

        # 2. 控制按钮
        ctrl_frame = tk.Frame(self.root, pady=10)
        ctrl_frame.pack(fill="x", padx=10)
        
        self.btn_start = tk.Button(ctrl_frame, text="开始采集", bg="#90EE90", command=self.toggle_recording, height=2)
        self.btn_start.pack(fill="x")
        
        # 3. 文件列表管理
        file_frame = tk.LabelFrame(self.root, text="历史数据文件 (data_logs/)", padx=10, pady=10)
        file_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # 滚动条和列表
        scrollbar = tk.Scrollbar(file_frame)
        scrollbar.pack(side="right", fill="y")
        
        self.file_list = tk.Listbox(file_frame, yscrollcommand=scrollbar.set, font=("Courier", 9))
        self.file_list.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.file_list.yview)
        
        # 双击打开文件
        self.file_list.bind('<Double-1>', self._on_file_double_click)
        
        # 文件操作按钮
        btn_frame = tk.Frame(file_frame)
        btn_frame.pack(fill="x", side="bottom")
        tk.Button(btn_frame, text="刷新列表", command=self.refresh_file_list).pack(side="left", padx=2)
        tk.Button(btn_frame, text="打开文件夹", command=self.open_folder).pack(side="right", padx=2)

        # 初始化列表
        self.refresh_file_list()
        
        # 启动定时刷新循环
        self.is_running = True
        self.root.after(500, self.update_ui_loop)
        self.root.mainloop()

    def _on_file_double_click(self, event):
        selection = self.file_list.curselection()
        if not selection:
            return
        filename = self.file_list.get(selection[0])
        path = os.path.abspath(os.path.join(self.collector.output_dir, filename))
        try:
            if platform.system() == "Windows":
                os.startfile(path)
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as e:
            messagebox.showerror("打开失败", f"无法打开文件:\n{e}")

    def toggle_recording(self):
        if not self.collector.is_recording:
            self.collector.start_recording()
            self.btn_start.config(text="停止采集并保存", bg="#FFB6C1") # 变粉色
            self.lbl_status.config(text="状态: 正在采集 REC ●", fg="red")
        else:
            path = self.collector.stop_recording()
            self.btn_start.config(text="开始采集", bg="#90EE90") # 变绿色
            self.lbl_status.config(text="状态: 已保存", fg="blue")
            if path:
                messagebox.showinfo("保存成功", f"文件已保存至:\n{os.path.basename(path)}")
                self.refresh_file_list()

    def update_ui_loop(self):
        """定时从Collector获取数据更新界面"""
        if not self.is_running:
            return
            
        # 读取Collector的实时数据
        with self.collector.lock:
            stats = self.collector.current_stats
            
        self.lbl_time.config(text=f"记录时长: {stats['time']:.1f}s")
        self.lbl_val_sum.config(text=str(stats['unscanned']))
        self.lbl_coverage.config(text=f"{stats['coverage']:.1f}%")
        
        # 继续循环
        self.root.after(500, self.update_ui_loop)

    def refresh_file_list(self):
        """刷新文件列表"""
        self.file_list.delete(0, tk.END)
        if os.path.exists(self.collector.output_dir):
            files = sorted(os.listdir(self.collector.output_dir), reverse=True)
            for f in files:
                if f.endswith(".csv") or f.endswith(".json"):
                    self.file_list.insert(tk.END, f)

    def open_folder(self):
        """打开数据文件夹"""
        path = os.path.abspath(self.collector.output_dir)
        try:
            if platform.system() == "Windows":
                os.startfile(path)
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as e:
            print(f"无法打开文件夹: {e}")

    def stop(self):
        self.is_running = False
        if self.root:
            try:
                self.root.quit()
            except Exception:
                pass
