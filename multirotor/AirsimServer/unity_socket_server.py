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
            self.pending_runtime_data = data
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
                    pass
                
                # 发送待发送的配置数据
                if self.pending_config_data:
                    self._send_data(conn, self.pending_config_data)
                    self.pending_config_data = None
                
                # 发送待发送的运行时数据
                if self.pending_runtime_data:
                    self._send_data(conn, self.pending_runtime_data)
                    self.pending_runtime_data = None
                
                time.sleep(0.01)
            except Exception as e:
                logger.error(f"处理Unity连接时出错: {str(e)}")
                break
        
        try:
            conn.close()
            self.connection = None
            logger.info("已断开与Unity的连接")
        except Exception as e:
            logger.error(f"关闭Unity连接时出错: {str(e)}")
    
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
        except json.JSONDecodeError:
            # 尝试寻找最后一个花括号，可能包含多个消息或消息不完整
            last_brace_pos = self.receive_buffer.rfind('}')
            if last_brace_pos > 0:
                # 尝试解析最后一个花括号之前的内容
                try:
                    partial_data = self.receive_buffer[:last_brace_pos+1]
                    received_data = json.loads(partial_data)
                    # 解析成功，处理数据
                    self._process_parsed_data(received_data)
                    # 保留未解析的数据在缓冲区
                    self.receive_buffer = self.receive_buffer[last_brace_pos+1:].lstrip()
                except json.JSONDecodeError:
                    # 仍然无法解析，可能是消息不完整，等待更多数据
                    pass
            # 否则，继续积累数据
        except Exception as e:
            logger.error(f"处理缓冲区数据时出错: {str(e)}")
    
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
        if callback_data and self.data_received_callback and isinstance(self.data_received_callback, Callable):
            self.data_received_callback(callback_data)
    
    def _send_data(self, conn: socket.socket, data: Dict[str, Any]) -> None:
        """向Unity发送数据"""
        try:
            json_data = json.dumps(data, ensure_ascii=False)
            conn.sendall(json_data.encode('utf-8'))
            logger.debug(f"已发送数据到Unity: {json_data}")
        except Exception as e:
            logger.error(f"向Unity发送数据时出错: {str(e)}")
    