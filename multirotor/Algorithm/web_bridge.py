import threading
import time
import os
import csv
from typing import Optional
from .HexGridDataModel import HexGridDataModel

# 尝试导入 socketio，如果失败则设置为 None
try:
    import socketio
    HAS_SOCKETIO = True
except ImportError as e:
    HAS_SOCKETIO = False
    socketio = None
    print(f"[WebBridge] 警告: python-socketio 未安装: {e}")
    print("[WebBridge] 提示: 请运行 'pip install python-socketio[client]' 安装依赖")

class WebBridge:
    """
    负责 Python 与 Node.js Web服务器的通信
    替代原来的 Tkinter GUI，实现完全解耦的 Web 控制台
    """
    def __init__(self, output_dir="data_logs"):
        if not HAS_SOCKETIO:
            raise ImportError("python-socketio 未安装，无法使用 WebBridge。请运行: pip install python-socketio[client]")
        
        self.sio = socketio.Client()
        self.output_dir = output_dir
        self.is_connected = False
        self.is_recording = False
        
        # 数据存储
        self.start_time = 0
        self.records = []
        self.lock = threading.Lock()

        # 确保输出目录存在
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # 注册事件
        self.sio.on('connect', self._on_connect)
        self.sio.on('disconnect', self._on_disconnect)
        self.sio.on('py_command', self._on_command)

        # 启动自动连接线程
        self.conn_thread = threading.Thread(target=self._connect_loop, daemon=True)
        self.conn_thread.start()

    def _connect_loop(self):
        """后台线程：持续尝试连接到 Node.js 服务器"""
        while True:
            if not self.is_connected:
                try:
                    # 连接到本地的 Node.js 服务器
                    self.sio.connect('http://localhost:3000', wait_timeout=2)
                except Exception as e:
                    # 连接失败，等待后重试
                    time.sleep(2)
            else:
                # 已连接，定期检查连接状态
                time.sleep(5)

    def _on_connect(self):
        """连接成功回调"""
        print("[WebBridge] 已连接到 Web 控制台 (http://localhost:3000)")
        self.is_connected = True

    def _on_disconnect(self):
        """断开连接回调"""
        print("[WebBridge] 与 Web 控制台断开连接")
        self.is_connected = False

    def _on_command(self, data):
        """处理来自网页的指令"""
        action = data.get('action')
        if action == 'start':
            self.start_recording()
        elif action == 'stop':
            self.stop_recording()

    def start_recording(self):
        """开始记录数据"""
        with self.lock:
            self.records = []
            self.start_time = time.time()
            self.is_recording = True
            print("[WebBridge] 开始记录数据...")

    def stop_recording(self):
        """停止记录并保存文件"""
        with self.lock:
            if not self.is_recording:
                return None
            self.is_recording = False
            filename = self._save_to_file()
            if filename:
                print(f"[WebBridge] 停止记录，保存至 {filename}")
                
                # 通知 Web 端文件已保存
                if self.is_connected:
                    try:
                        self.sio.emit('py_file_saved', filename)
                    except Exception as e:
                        print(f"[WebBridge] 发送文件保存通知失败: {e}")
            return filename

    def process_step(self, grid_data: Optional[HexGridDataModel]):
        """
        处理单步数据并推送到 Web
        
        Args:
            grid_data: HexGridDataModel 实例，包含网格数据
        """
        if not grid_data or not hasattr(grid_data, 'cells'):
            return

        total_cells = len(grid_data.cells)
        if total_cells == 0:
            return

        # 统计逻辑：未被侦察(entropy >= 30) = 未侦察, 被侦察(entropy < 30) = 已侦察
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
        
        # 时间计算
        elapsed = 0
        if self.is_recording:
            elapsed = time.time() - self.start_time
        
        # 1. 记录数据 (如果在录制中)
        if self.is_recording:
            with self.lock:
                self.records.append({
                    "timestamp": round(elapsed, 2),
                    "aoi_sum_value": aoi_sum_value,
                    "scanned_count": scanned_count,
                    "total_cells": total_cells,
                    "coverage_ratio": round(coverage, 2)
                })

        # 2. 推送实时数据给 Web (无论是否录制，只要连接就推送，用于监控)
        if self.is_connected:
            telemetry = {
                "recording": self.is_recording,
                "time": elapsed,
                "unscanned": unscanned_count,
                "scanned": scanned_count,
                "coverage": coverage,
                "total": total_cells
            }
            try:
                self.sio.emit('py_telemetry', telemetry)
            except Exception as e:
                # 发送失败不影响主程序
                pass

    def _save_to_file(self):
        """保存数据到CSV文件"""
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
            return filename
        except Exception as e:
            print(f"[WebBridge] 保存CSV失败: {e}")
            return None

    def get_current_stats(self):
        """获取当前统计数据（用于兼容性）"""
        with self.lock:
            elapsed = time.time() - self.start_time if self.is_recording else 0
            return {
                "time": elapsed,
                "is_recording": self.is_recording
            }

