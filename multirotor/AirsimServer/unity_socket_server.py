import socket
import json
import threading
import logging
import time
from typing import Dict, Any, Optional, Callable
from Algorithm.scanner_runtime_data import ScannerRuntimeData
from Algorithm.scanner_config_data import ScannerConfigData

# 配置日志（简化输出）
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("UnitySocketServer")


class UnitySocketServer:
    """与Unity通信的Socket服务器核心类"""
    
    def __init__(self, host='localhost', port=5000, buffer_size=4096):
        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        self.socket = None
        self.connection = None  # 当前连接
        self.running = False    # 运行状态标志
        self.server_thread = None  # 服务器主线程
        
        # 数据缓冲区
        self.receive_buffer = ""  # 接收缓存
        self.pending_config = None  # 待发送的配置数据
        self.pending_runtime = None  # 待发送的运行时数据
        
        # 接收数据存储与回调
        self.received_grid = None
        self.received_runtime = None
        self.data_callback = None  # 数据接收回调函数

    def start(self) -> bool:
        """启动Socket服务器并监听连接"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen(1)
            self.running = True
            
            # 启动服务器线程
            self.server_thread = threading.Thread(target=self._server_loop, daemon=True)
            self.server_thread.start()
            logger.info(f"服务器启动成功 {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"启动失败: {str(e)}")
            return False

    def stop(self) -> None:
        """停止服务器并释放资源"""
        self.running = False
        
        # 关闭连接
        if self.connection:
            try:
                self.connection.close()
                logger.info("连接已关闭")
            except Exception as e:
                logger.error(f"关闭连接出错: {str(e)}")
            self.connection = None
        
        # 关闭socket
        if self.socket:
            try:
                self.socket.close()
                logger.info("服务器已停止")
            except Exception as e:
                logger.error(f"关闭socket出错: {str(e)}")
            self.socket = None

    def is_connected(self) -> bool:
        """检查是否与Unity建立连接"""
        return self.connection is not None

    def send_config(self, config: ScannerConfigData) -> None:
        """发送配置数据到Unity"""
        try:
            self.pending_config = {
                "type": "config_data",
                "timestamp": time.time(),
                "data": config.to_dict()
            }
        except Exception as e:
            logger.error(f"配置数据准备失败: {str(e)}")

    def send_runtime(self, runtime: ScannerRuntimeData) -> None:
        """发送运行时数据到Unity"""
        try:
            data = {
                "type": "runtime_data",
                "timestamp": time.time(),
                "data": runtime.to_dict()
            }
            # 添加无人机标识
            if hasattr(runtime, 'drone_name'):
                data['uav_name'] = runtime.drone_name
            self.pending_runtime = data
        except Exception as e:
            logger.error(f"运行时数据准备失败: {str(e)}")

    def set_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """设置数据接收回调函数"""
        self.data_callback = callback

    def _server_loop(self) -> None:
        """服务器主循环：等待并处理Unity连接"""
        while self.running:
            try:
                self.socket.settimeout(1.0)
                conn, addr = self.socket.accept()
                self.connection = conn
                logger.info(f"Unity已连接: {addr}")
                self._handle_conn(conn)  # 处理连接
            except socket.timeout:
                continue  # 超时继续等待
            except Exception as e:
                if self.running:
                    logger.error(f"服务器循环错误: {str(e)}")
                break

    def _handle_conn(self, conn: socket.socket) -> None:
        """处理与Unity的单连接通信"""
        conn.settimeout(1.0)
        self.receive_buffer = ""  # 重置缓冲区

        while self.running:
            try:
                # 接收数据并解析
                self._recv_and_parse(conn)
                
                # 发送待处理数据
                self._send_pending_data(conn)
                
                time.sleep(0.01)  # 降低CPU占用
            except Exception as e:
                logger.error(f"连接处理错误: {str(e)}")
                break

        # 清理连接
        conn.close()
        self.connection = None
        logger.info("连接已断开")

    def _recv_and_parse(self, conn: socket.socket) -> None:
        """接收数据并解析JSON"""
        try:
            data = conn.recv(self.buffer_size).decode('utf-8')
            if data:
                self.receive_buffer += data
                self._parse_buffer()  # 解析缓冲区
        except socket.timeout:
            pass  # 超时忽略
        except ConnectionResetError:
            logger.warning("Unity强制断开连接")
            raise
        except Exception as e:
            logger.error(f"接收数据错误: {str(e)}")

    def _parse_buffer(self) -> None:
        """解析缓冲区中的JSON数据（支持多对象连续传输，修复嵌套花括号问题）"""
        if '{' not in self.receive_buffer:
            return  # 没有起始符，直接返回

        try:
            # 尝试解析整个缓冲区（完整数据场景）
            parsed = json.loads(self.receive_buffer)
            self._handle_parsed(parsed)
            self.receive_buffer = ""
            return
        except json.JSONDecodeError:
            pass  # 不完整，进入部分解析逻辑
        except Exception as e:
            logger.error(f"完整解析失败: {str(e)}")
            self.receive_buffer = ""
            return

        # 部分解析：通过花括号匹配找到第一个完整的JSON对象
        buffer = self.receive_buffer
        start_idx = buffer.find('{')
        if start_idx == -1:
            self.receive_buffer = ""  # 无有效起始，清空
            return

        stack = []
        end_idx = -1
        # 从第一个'{'开始遍历，跟踪花括号匹配
        for i in range(start_idx, len(buffer)):
            char = buffer[i]
            if char == '{':
                stack.append(char)
            elif char == '}':
                if stack:
                    stack.pop()
                    # 栈为空时，找到完整对象的结束位置
                    if not stack:
                        end_idx = i
                        break

        if end_idx == -1:
            # 未找到完整对象，保留缓冲区等待后续数据
            return
        else:
            # 提取并解析第一个完整对象
            partial = buffer[start_idx:end_idx + 1]
            remaining = buffer[end_idx + 1:].lstrip()  # 剩余数据
            try:
                parsed = json.loads(partial)
                self._handle_parsed(parsed)
                self.receive_buffer = remaining  # 更新缓冲区为剩余数据
                if self.receive_buffer:
                    self._parse_buffer()  # 递归处理剩余数据
            except json.JSONDecodeError as e:
                logger.warning(f"部分数据解析失败（内容: {partial[:100]}...），错误: {str(e)}")
                # 跳过当前错误片段，从下一个'{'开始
                next_start = buffer.find('{', start_idx + 1)
                self.receive_buffer = buffer[next_start:] if next_start != -1 else ""
                if self.receive_buffer:
                    self._parse_buffer()
            except Exception as e:
                logger.error(f"处理部分数据时出错: {str(e)}")
                self.receive_buffer = remaining

    def _handle_parsed(self, data: Dict[str, Any]) -> None:
        """处理解析后的JSON数据并触发回调"""
        if not data or 'type' not in data:
            logger.warning("忽略无效数据（缺少type字段）")
            return

        callback_data = {}
        data_type = data['type']
        
        # 提取数据内容
        if 'data' in data:
            if data_type == 'grid_data':
                self.received_grid = data['data']
                callback_data['grid_data'] = data['data']
            elif data_type == 'runtime_data':
                self.received_runtime = data['data']
                callback_data['runtime_data'] = data['data']
        
        # 附加无人机标识
        if 'uav_name' in data:
            callback_data['uav_name'] = data['uav_name']
        
        # 触发回调
        if callback_data and self.data_callback:
            try:
                self.data_callback(callback_data)
            except Exception as e:
                logger.error(f"回调函数执行失败: {str(e)}")

    def _send_pending_data(self, conn: socket.socket) -> None:
        """发送待处理的配置/运行时数据"""
        # 发送配置数据
        if self.pending_config:
            self._send(conn, self.pending_config)
            self.pending_config = None
        
        # 发送运行时数据
        if self.pending_runtime:
            self._send(conn, self.pending_runtime)
            self.pending_runtime = None

    def _send(self, conn: socket.socket, data: Dict[str, Any]) -> None:
        """向Unity发送JSON数据"""
        try:
            json_str = json.dumps(data, ensure_ascii=False)
            conn.sendall(json_str.encode('utf-8'))
            logger.debug(f"发送数据: {data['type']}")
        except Exception as e:
            logger.error(f"发送数据失败: {str(e)}")