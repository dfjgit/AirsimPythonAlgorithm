import socket
import json
import threading
import logging
import time
from typing import Dict, Any, Optional, Callable, Iterable
from Algorithm.scanner_runtime_data import ScannerRuntimeData
from Algorithm.scanner_config_data import ScannerConfigData
from Crazyswarm.crazyflie_operate import CrazyflieOperate
from AirsimServer.data_pack import DataPacks, PackType

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("UnitySocketServer")


class UnitySocketServer:
    """与Unity通信的Socket服务器核心类"""

    def __init__(self, host='localhost', port=5000, buffer_size=8192):  # 优化：增大缓冲区
        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        self.socket = None
        self.connection = None  # 当前连接
        self.running = False  # 运行状态标志
        self.server_thread = None  # 服务器主线程

        # 数据缓冲区
        self.receive_buffer = ""  # 接收缓存
        self.pending_packs = []  # 待发送的数据包列表（使用DataPacks结构）

        # 添加发送锁，解决多线程同时发送导致的粘包问题
        self.send_lock = threading.Lock()
        # 接收数据存储与回调
        self.received_grid = None
        self.received_runtimes = []  # 存储多个运行时数据
        self.received_crazyflie_logging = [] #存储所有Crazyflie无人机的当前日志
        self.data_callback = None  # 数据接收回调函数
        
        # 性能统计
        self.stats_grid_updates = 0
        self.stats_runtime_updates = 0
        self.stats_crazyflie_logging_updates = 0

    def start(self) -> bool:
        """启动Socket服务器并监听连接"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen(1)
            self.running = True

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

        if self.connection:
            try:
                self.connection.close()
                logger.info("连接已关闭")
            except Exception as e:
                logger.error(f"关闭连接出错: {str(e)}")
            self.connection = None

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
        """发送配置数据到Unity（适配字典类型的pack_data_list）"""
        try:
            pack = DataPacks()
            pack.type = PackType.config_data
            pack.time_span = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            pack.pack_data_list = config.to_dict()  # 字典结构（匹配config.json）
            logging.info("发送配置数据")
            self.pending_packs.append(pack)
        except Exception as e:
            logger.error(f"配置数据准备失败: {str(e)}")

    def send_runtime(self, runtimes: Iterable[ScannerRuntimeData]) -> None:
        """发送多个运行时数据到Unity（列表类型的pack_data_list）"""
        try:
            pack = DataPacks()
            pack.type = PackType.runtime_data
            pack.time_span = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            # 每个runtime数据已包含uavname，无需顶层字段
            pack.pack_data_list = [runtime.to_dict() for runtime in runtimes]
            self.pending_packs.append(pack)
            # logger.debug(f"添加了包含{len(pack.pack_data_list)}个运行时数据的数据包")
        except Exception as e:
            logger.error(f"运行时数据准备失败: {str(e)}")

    def send_crazyflie_operate(self, operateDatas: list[CrazyflieOperate]):
        """发送实体无人机操作指令数据到Unity"""
        try:
            pack = DataPacks()
            pack.type = PackType.crazyflie_operate_data
            pack.time_span = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            pack.pack_data_list = [operateData.to_dict() for operateData in operateDatas]
            self.pending_packs.append(pack)
            logger.info(f"实体无人机操作指令数据包数据：{pack.pack_data_list}")
        except Exception as e:
            logger.error(f"实体无人机Crazyflie指令数据准备失败: {str(e)}")

    
    def send_reset_command(self) -> None:
        """发送环境重置命令到Unity"""
        try:
            pack = DataPacks()
            pack.type = PackType.reset_env
            pack.time_span = str(time.time())
            pack.pack_data_list = {}  # 重置命令不需要额外数据
            self.pending_packs.append(pack)
            logger.info("[重置] 已发送环境重置命令到Unity")
        except Exception as e:
            logger.error(f"发送重置命令失败: {str(e)}")
    
    def get_stats(self) -> dict:
        """获取通信统计信息"""
        return {
            'grid_updates': self.stats_grid_updates,
            'runtime_updates': self.stats_runtime_updates,
            'is_connected': self.is_connected()
        }

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
                self._handle_conn(conn)
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    logger.error(f"服务器循环错误: {str(e)}")
                break

    def _handle_conn(self, conn: socket.socket) -> None:
        """处理与Unity的单连接通信"""
        conn.settimeout(1.0)
        self.receive_buffer = ""

        while self.running:
            try:
                self._recv_and_parse(conn)
                self._send_pending_data(conn)
                time.sleep(0.01)
            except Exception as e:
                logger.error(f"连接处理错误: {str(e)}")
                break

        conn.close()
        self.connection = None
        logger.info("连接已断开")

    def _recv_and_parse(self, conn: socket.socket) -> None:
        """接收数据并解析JSON"""
        try:
            data = conn.recv(self.buffer_size).decode('utf-8')
            if not data:
                # 收到空数据可能意味着连接关闭
                raise ConnectionResetError("收到空数据，连接可能已关闭")
            self.receive_buffer += data
            self._parse_buffer()
        except socket.timeout:
            pass  # 超时是正常的，继续等待
        except ConnectionResetError:
            logger.warning("Unity断开连接")
            raise  # 重新抛出，让上层处理
        except OSError as e:
            # Windows错误 10038: 在一个非套接字上尝试了一个操作
            if e.winerror == 10038:
                logger.warning("Socket已关闭，等待重新连接")
                raise ConnectionResetError("Socket已关闭")
            logger.error(f"接收数据OS错误: {str(e)}")
            raise  # 重新抛出严重错误
        except Exception as e:
            # 其他错误记录但不断开连接
            logger.warning(f"接收数据时出现异常: {str(e)}")


    def _parse_buffer(self) -> None:
        """解析缓冲区中的JSON数据（适配DataPacks结构）"""

        if not self.receive_buffer:
            return

        # 使用换行符作为数据包分隔符
        while '\n' in self.receive_buffer:
            # 找到第一个换行符的位置
            newline_pos = self.receive_buffer.index('\n')
            # 提取一个完整的数据包
            packet = self.receive_buffer[:newline_pos]
            # 更新缓冲区，保留剩余数据
            self.receive_buffer = self.receive_buffer[newline_pos + 1:]

            # 处理提取出的数据包
            if packet.strip():
                try:
                    # logger.debug(f"解析数据包: {packet}")
                    parsed = json.loads(packet)
                    if isinstance(parsed, dict) and 'type' in parsed:
                        self._handle_parsed(parsed)
                    else:
                        logger.warning(f"无效的数据包格式（缺少type字段）: {packet}")
                except json.JSONDecodeError as e:
                    logger.error(f"JSON解析错误: {str(e)}, 数据包: {packet}")
                except Exception as e:
                    logger.error(f"解析数据包时出错: {str(e)}, 数据包: {packet}")

        # 对于没有换行符的情况，我们不做处理，等待更多数据到达


    def _handle_parsed(self, data: Dict[str, Any]) -> None: ##这里data应该时DataPack格式
        """处理解析后的DataPacks数据并触发回调"""
        if not data or 'type' not in data:
            logger.warning("忽略无效数据（缺少type字段）")
            return

        callback_data = {}
        data_type = data['type']
        pack_data_list = data.get('pack_data_list', {})  # 默认为空字典

        # 处理不同类型的数据
        if data_type == PackType.grid_data.value:
            # grid_data的pack_data_list是字典（包含cells字段）
            self.received_grid = pack_data_list if isinstance(pack_data_list, dict) else None
            callback_data['grid_data'] = self.received_grid
            self.stats_grid_updates += 1
            # logger.debug(f"收到网格数据，cells数量: {len(self.received_grid.get('cells', [])) if self.received_grid else 0}")
        elif data_type == PackType.runtime_data.value:
            # runtime_data的pack_data_list是列表，每个元素包含uavname
            self.received_runtimes = pack_data_list if isinstance(pack_data_list, list) else []
            callback_data['runtime_data'] = self.received_runtimes
            self.stats_runtime_updates += 1
        elif data_type == PackType.crazyflie_logging_data.value:
            self.received_crazyflie_logging = pack_data_list
            callback_data['crazyflie_logging'] = self.received_crazyflie_logging
            self.stats_crazyflie_logging_updates += 1
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
        """向Unity发送完整的DataPacks结构（不含顶层uav_name）"""
        try:
            # 构建数据包字典（移除uav_name字段）
            pack_dict = {
                "type": pack.type.value,
                "time_span": pack.time_span,
                "pack_data_list": pack.pack_data_list  # 核心数据（列表或字典）
            }
            # 序列化为JSON并发送
            json_data = json.dumps(pack_dict, ensure_ascii=False) + "\n"  # 加换行符作为分隔符
            # 使用锁确保发送操作的原子性
            with self.send_lock:
                conn.sendall(json_data.encode('utf-8'))
            # logger.debug(f"发送数据: {pack_dict['type']} (长度: {len(json_data)})")
        except Exception as e:
            logger.error(f"发送数据失败: {str(e)}")