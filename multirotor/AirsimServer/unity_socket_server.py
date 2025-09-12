import socket
import json
import threading
import logging
import time
from typing import Dict, Any, Optional, Callable, Iterable
from Algorithm.scanner_runtime_data import ScannerRuntimeData
from Algorithm.scanner_config_data import ScannerConfigData
# 导入新的数据包结构
from AirsimServer.data_pack import DataPacks, PackType

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
        self.pending_packs = []  # 待发送的数据包列表（使用DataPacks结构）
        
        # 接收数据存储与回调
        self.received_grid = None
        self.received_runtimes = []  # 改为列表存储多个运行时数据
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
        """发送配置数据到Unity（使用新的DataPacks结构）"""
        try:
            # 创建数据包
            pack = DataPacks()
            pack.type = PackType.config_data
            pack.time_span = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            pack.pack_data_list = [config.to_dict()]  # 放入列表中
            
            self.pending_packs.append(pack)
        except Exception as e:
            logger.error(f"配置数据准备失败: {str(e)}")

    def send_runtime(self, runtimes: Iterable[ScannerRuntimeData]) -> None:
        """发送多个运行时数据到Unity（修改为接收可迭代的多个数据）"""
        try:
            # 创建数据包
            pack = DataPacks()
            pack.type = PackType.runtime_data
            pack.time_span = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            
            # 将多个ScannerRuntimeData转换为字典并添加到列表
            pack.pack_data_list = [runtime.to_dict() for runtime in runtimes]
            
            # 如果有无人机标识，使用第一个runtime的标识（或根据实际需求调整）
            first_runtime = next(iter(runtimes), None)
            if first_runtime and hasattr(first_runtime, 'drone_name'):
                setattr(pack, 'uav_name', first_runtime.drone_name)
                
            self.pending_packs.append(pack)
            logger.debug(f"添加了包含{len(pack.pack_data_list)}个运行时数据的数据包")
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
        """解析缓冲区中的JSON数据（适配新的DataPacks结构）"""
        if '{' not in self.receive_buffer:
            return  # 没有起始符，直接返回

        try:
            # 尝试解析整个缓冲区作为DataPacks对象
            parsed = json.loads(self.receive_buffer)
            
            # 验证是否是有效的DataPacks结构
            if not isinstance(parsed, dict) or 'type' not in parsed:
                raise ValueError("数据格式错误：必须是包含'type'字段的DataPacks对象")
            
            self._handle_parsed(parsed)
            self.receive_buffer = ""  # 成功解析后清空缓冲区
        except json.JSONDecodeError as e:
            logger.error(f"JSON格式错误: {str(e)}")
            self.receive_buffer = ""  # 清空缓冲区
        except ValueError as e:
            logger.error(str(e))
            self.receive_buffer = ""
        except Exception as e:
            logger.error(f"解析数据时出错: {str(e)}")
            self.receive_buffer = ""

    def _handle_parsed(self, data: Dict[str, Any]) -> None:
        """处理解析后的DataPacks数据并触发回调"""
        if not data or 'type' not in data:
            logger.warning("忽略无效数据（缺少type字段）")
            return

        callback_data = {}
        data_type = data['type']
        pack_data_list = data.get('pack_data_list', [])
        
        # 处理数据内容
        if pack_data_list and isinstance(pack_data_list, list):
            if data_type == PackType.grid_data.value:
                # grid_data保持只取第一个元素（根据需求）
                self.received_grid = pack_data_list[0] if pack_data_list else None
                callback_data['grid_data'] = self.received_grid
            elif data_type == PackType.runtime_data.value:
                # runtime_data改为接收所有元素
                self.received_runtimes = pack_data_list
                callback_data['runtime_data'] = self.received_runtimes
        
        # 附加无人机标识
        if 'uav_name' in data:
            callback_data['uav_name'] = data['uav_name']
        
        # 附加时间跨度
        if 'time_span' in data:
            callback_data['time_span'] = data['time_span']
        
        # 触发回调
        if callback_data and self.data_callback:
            try:
                self.data_callback(callback_data)
            except Exception as e:
                logger.error(f"回调函数执行失败: {str(e)}")

    def _send_pending_data(self, conn: socket.socket) -> None:
        """发送待处理的数据包"""
        while self.pending_packs and self.connection:
            pack = self.pending_packs.pop(0)
            self._send(conn, pack)

    def _send(self, conn: socket.socket, pack: DataPacks) -> None:
        """向Unity发送DataPacks结构的JSON数据"""
        try:
            # 转换为可序列化的字典
            pack_dict = {
                "type": pack.type.value,
                "time_span": pack.time_span,
                "pack_data_list": pack.pack_data_list
            }
            # 添加无人机标识（如果有）
            if hasattr(pack, 'uav_name'):
                pack_dict['uav_name'] = pack.uav_name
                
            json_str = json.dumps(pack_dict, ensure_ascii=False)
            conn.sendall(json_str.encode('utf-8'))
            logger.debug(f"发送数据包: {pack.type.value}，包含{len(pack.pack_data_list)}个数据项")
        except Exception as e:
            logger.error(f"发送数据失败: {str(e)}")
