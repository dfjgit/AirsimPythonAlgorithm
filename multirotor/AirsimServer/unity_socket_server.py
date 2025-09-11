import socket
import json
import threading
import logging
import time
from typing import Dict, Any, Optional, Callable
from Algorithm.scanner_runtime_data import ScannerRuntimeData
from Algorithm.HexGridDataModel import HexGridDataModel
from Algorithm.scanner_config_data import ScannerConfigData

logger = logging.getLogger("UnitySocketServer")

class UnitySocketServer:
    """与Unity通信的Socket服务器类"""
    def __init__(self, host='localhost', port=5000, buffer_size=4096):
        """初始化Socket服务器"""
        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        self.socket = None
        self.connection = None
        self.running = False
        self.server_thread = None
        
        # 待发送数据缓冲区（分离存储）
        self.pending_config_data = None
        self.pending_runtime_data = None
        
        # 接收数据存储
        self.received_grid_data = None
        self.received_runtime_data = None
        
        self.data_received_callback = None
        
        # 消息缓存，用于处理分段接收的JSON数据
        self.receive_buffer = ""
        
    def start(self) -> bool:
        """启动Socket服务器"""
        logger.info("调用start()启动Socket服务器")
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen(1)
            self.running = True
            
            self.server_thread = threading.Thread(target=self._server_loop)
            self.server_thread.daemon = True
            self.server_thread.start()
            
            logger.info(f"Unity Socket服务器已启动在 {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"启动Unity Socket服务器失败: {str(e)}")
            return False
    
    def stop(self) -> None:
        """停止Socket服务器"""
        logger.info("调用stop()停止Socket服务器")
        self.running = False
        
        if self.connection:
            try:
                self.connection.close()
                logger.info("已关闭与Unity的连接")
            except Exception as e:
                logger.error(f"关闭连接时出错: {str(e)}")
            self.connection = None
        
        if self.socket:
            try:
                self.socket.close()
                logger.info("已关闭Socket服务器")
            except Exception as e:
                logger.error(f"关闭Socket时出错: {str(e)}")
            self.socket = None
        
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(2.0)
    
    def is_connected(self) -> bool:
        """检查与Unity的连接状态"""
        logger.info("调用is_connected()检查连接状态")
        return self.connection is not None
    
    def send_config_data(self, config_data: ScannerConfigData) -> None:
        """发送配置数据到Unity"""
        logger.info("调用send_config_data()发送配置数据")
        try:
            data = {
                "type": "config_data",
                "timestamp": time.time(),
                "data": config_data.to_dict()
            }
            self.pending_config_data = data
        except Exception as e:
            logger.error(f"准备发送配置数据时出错: {str(e)}")
    
    def send_runtime_data(self, runtime_data: ScannerRuntimeData) -> None:
        """发送运行时数据到Unity"""
        logger.info("调用send_runtime_data()发送运行时数据")
        try:
            data = {
                "type": "runtime_data",
                "timestamp": time.time(),
                "data": runtime_data.to_dict()
            }
            
            # 如果运行时数据中包含无人机名称，添加到顶级字段
            if hasattr(runtime_data, 'drone_name') and runtime_data.drone_name:
                data['uav_name'] = runtime_data.drone_name
                logger.debug(f"添加无人机标识: {runtime_data.drone_name}")
            
            self.pending_runtime_data = data
            logger.debug(f"已准备运行时数据，等待发送: {data['type']}, timestamp: {data['timestamp']}")
            
            # 如果有连接，尝试立即发送（而不仅仅依赖主循环）
            if self.connection:
                logger.debug("检测到有活跃连接，尝试立即发送运行时数据")
        except Exception as e:
            logger.error(f"准备发送运行时数据时出错: {str(e)}")
    
    def get_grid_data(self) -> Optional[Dict[str, Any]]:
        """获取最近接收到的网格数据"""
        logger.info("调用get_grid_data()获取网格数据")
        return self.received_grid_data
    
    def get_runtime_data(self) -> Optional[Dict[str, Any]]:
        """获取最近接收到的运行时数据"""
        logger.info("调用get_runtime_data()获取运行时数据")
        return self.received_runtime_data
    
    def set_data_received_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """设置数据接收回调函数"""
        logger.info("调用set_data_received_callback()设置数据接收回调")
        self.data_received_callback = callback
    
    def _server_loop(self) -> None:
        """服务器主循环，等待Unity连接"""
        while self.running:
            try:
                self.socket.settimeout(1.0)
                logger.info("等待Unity连接...")
                conn, addr = self.socket.accept()
                self.connection = conn
                logger.info(f"Unity已连接，地址: {addr}")
                self._handle_connection(conn)
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    logger.error(f"服务器循环出错: {str(e)}")
                break

    def _handle_connection(self, conn: socket.socket) -> None:
        """处理与Unity的连接"""
        conn.settimeout(1.0)

        # 重置接收缓冲区
        self.receive_buffer = ""

        while self.running:
            try:
                # 接收Unity发送的数据
                try:
                    data = conn.recv(self.buffer_size).decode('utf-8')
                    if data:
                        # 将新接收的数据添加到缓冲区
                        self.receive_buffer += data
                        # 尝试解析缓冲区中的完整JSON消息
                        self._process_buffered_data()
                except socket.timeout:
                    # 超时异常不需要处理，继续循环
                    pass
                except ConnectionResetError:
                    logger.warning("Unity客户端强制断开连接")
                    break
                except Exception as e:
                    logger.error(f"接收数据时出错: {str(e)}")
                    # 不要中断循环，继续尝试接收

                # 发送待发送的配置数据
                if self.pending_config_data:
                    try:
                        self._send_data(conn, self.pending_config_data)
                        self.pending_config_data = None
                    except Exception as e:
                        logger.error(f"发送配置数据时出错: {str(e)}")
                        break

                # 发送待发送的运行时数据
                if self.pending_runtime_data:
                    try:
                        logger.debug(f"正在发送运行时数据到Unity，数据类型: {self.pending_runtime_data['type']}")
                        if 'uav_name' in self.pending_runtime_data:
                            logger.debug(f"发送的运行时数据包含无人机标识: {self.pending_runtime_data['uav_name']}")
                        self._send_data(conn, self.pending_runtime_data)
                        logger.debug("运行时数据发送完成，清空待发送缓冲区")
                        self.pending_runtime_data = None
                    except Exception as e:
                        logger.error(f"发送运行时数据时出错: {str(e)}")
                        break

                time.sleep(0.01)  # 短暂休眠避免CPU占用过高
            except Exception as e:
                logger.error(f"处理Unity连接时出错: {str(e)}")
                break

        try:
            conn.close()
            self.connection = None
            logger.info("已断开与Unity的连接")
        except Exception as e:
            logger.error(f"关闭Unity连接时出错: {str(e)}")

    def _process_parsed_data(self, received_data: Dict[str, Any]) -> None:
        """处理已成功解析的Unity数据（仅支持新版格式）"""
        logger.debug(f"接收到Unity数据: {received_data}")

        # 创建一个用于回调的数据字典
        callback_data = {}

        # 仅处理新版数据格式（包含type字段）
        if 'type' in received_data:
            data_type = received_data['type']
            logger.info(f"接收到Unity数据类型: {data_type}")

            # 如果包含data字段，将其内容提取并添加到回调数据中
            if 'data' in received_data:
                if data_type == 'runtime_data':
                    self.received_runtime_data = received_data['data']
                    callback_data['runtime_data'] = received_data['data']
                    logger.debug("成功接收并解析Unity的runtime_data")
                elif data_type == 'grid_data':
                    self.received_grid_data = received_data['data']
                    callback_data['grid_data'] = received_data['data']
                    logger.debug("成功接收并解析Unity的grid_data")
                elif data_type == 'request':
                    # 处理请求类型的数据
                    if 'request' in received_data:
                        callback_data['request'] = received_data['request']

            # 保留额外的字段
            if 'uav_name' in received_data:
                callback_data['uav_name'] = received_data['uav_name']
        else:
            # 不支持旧版格式，记录警告
            logger.warning("接收到不支持的数据格式（缺少type字段），已忽略")

        # 如果有数据需要传递给回调函数，才调用回调
        if callback_data and self.data_received_callback and callable(self.data_received_callback):
            try:
                self.data_received_callback(callback_data)
            except Exception as e:
                logger.error(f"执行数据接收回调函数时出错: {str(e)}")
                # 回调函数出错不应影响主循环

    def _process_buffered_data(self) -> None:
        """处理缓冲区中的数据，尝试解析完整的JSON消息"""
        # 至少需要包含一对花括号
        if '{' not in self.receive_buffer or '}' not in self.receive_buffer:
            return

        try:
            # 尝试解析整个缓冲区
            received_data = json.loads(self.receive_buffer)
            # 解析成功，处理数据
            self._process_parsed_data(received_data)
            # 清空缓冲区
            self.receive_buffer = ""
        except json.JSONDecodeError as e:
            # 记录详细的错误信息，帮助定位问题
            error_pos = e.pos if hasattr(e, 'pos') else -1
            logger.warning(f"JSON解析错误在位置 {error_pos}: {str(e)}")

            # 检查是否是网格数据（通常体积较大）
            if '"type":"grid_data"' in self.receive_buffer[:100]:
                logger.info("检测到可能的网格数据，尝试特殊处理")
                # 尝试寻找完整的网格数据结构
                try:
                    # 查找grid_data的开始和结束位置
                    grid_start = self.receive_buffer.find('{"type":"grid_data"')
                    if grid_start >= 0:
                        # 寻找匹配的结束花括号
                        brace_count = 0
                        for i in range(grid_start, len(self.receive_buffer)):
                            if self.receive_buffer[i] == '{':
                                brace_count += 1
                            elif self.receive_buffer[i] == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    # 尝试解析完整的网格数据
                                    grid_data_str = self.receive_buffer[grid_start:i + 1]
                                    try:
                                        grid_data = json.loads(grid_data_str)
                                        logger.info("成功解析网格数据")
                                        self._process_parsed_data(grid_data)
                                        # 更新缓冲区，保留未处理的数据
                                        self.receive_buffer = self.receive_buffer[i + 1:].lstrip()
                                        # 递归处理剩余数据
                                        if self.receive_buffer:
                                            self._process_buffered_data()
                                    except json.JSONDecodeError:
                                        logger.warning("特殊处理网格数据失败")
                                        # 继续使用标准方法处理
                                        break
                except Exception as grid_e:
                    logger.error(f"网格数据特殊处理时出错: {str(grid_e)}")

            # 标准的部分解析逻辑
            last_brace_pos = self.receive_buffer.rfind('}')
            if last_brace_pos > 0:
                try:
                    partial_data = self.receive_buffer[:last_brace_pos + 1]
                    received_data = json.loads(partial_data)
                    self._process_parsed_data(received_data)
                    self.receive_buffer = self.receive_buffer[last_brace_pos + 1:].lstrip()
                    if self.receive_buffer:
                        self._process_buffered_data()
                except json.JSONDecodeError:
                    logger.warning(f"无法解析部分数据: {self.receive_buffer[:1000]}...")
                    first_brace_pos = self.receive_buffer.find('{')
                    if first_brace_pos >= 0:
                        self.receive_buffer = self.receive_buffer[first_brace_pos:]
                    else:
                        self.receive_buffer = ""
            else:
                logger.warning(f"缓冲区数据格式错误，无法解析: {self.receive_buffer[:1000]}...")
                self.receive_buffer = ""
        except Exception as e:
            logger.error(f"处理缓冲区数据时出错: {str(e)}")
            self.receive_buffer = ""
    
    def _send_data(self, conn: socket.socket, data: Dict[str, Any]) -> None:
        """向Unity发送数据"""
        try:
            json_data = json.dumps(data, ensure_ascii=False)
            conn.sendall(json_data.encode('utf-8'))
            logger.debug(f"已发送数据到Unity: {json_data}")
        except Exception as e:
            logger.error(f"向Unity发送数据时出错: {str(e)}")
    