import socket
import threading
import json
import logging
import socket
from typing import List, Tuple, Optional

# 配置日志
logger = logging.getLogger("SocketServer")


class BaseSocketServer:
    """
    基础Socket服务器类
    处理客户端连接和通信
    """
    def __init__(self, host: str = '0.0.0.0', port: int = 65432):
        self.host = host
        self.port = port
        self.server_socket: Optional[socket.socket] = None
        self.running = False
        self.client_handlers: List[threading.Thread] = []
        self.lock = threading.Lock()

    def start(self) -> None:
        """启动Socket服务器"""
        self.running = True
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            logger.info(f"服务器已启动，监听 {self.host}:{self.port}")

            while self.running:
                try:
                    self.server_socket.settimeout(1.0)
                    client_socket, addr = self.server_socket.accept()
                    logger.info(f"新连接: {addr}")
                    
                    handler = threading.Thread(target=self.handle_client, args=(client_socket, addr))
                    handler.daemon = True
                    handler.start()
                    
                    with self.lock:
                        self.client_handlers.append(handler)
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        logger.error(f"接受连接时出错: {e}")

        except Exception as e:
            logger.error(f"服务器错误: {e}")
        finally:
            self.stop()

    def handle_client(self, client_socket: socket.socket, addr: Tuple[str, int]) -> None:
        """
        处理客户端连接
        子类需要实现具体的命令处理逻辑
        """
        handler_thread = threading.current_thread()
        try:
            with client_socket:
                client_socket.settimeout(60.0)
                logger.info(f"开始处理客户端: {addr}")
                
                while True:
                    try:
                        data = client_socket.recv(1024).decode('utf-8')
                        if not data:
                            logger.info(f"客户端断开连接: {addr}")
                            break

                        try:
                            command = json.loads(data)
                            # 只记录命令类型，不记录完整数据内容
                            cmd_type = command.get("command", "unknown")
                            logger.debug(f"收到命令 from {addr}: 命令类型 = {cmd_type}")
                        except json.JSONDecodeError as e:
                            response = {"status": "error", "message": f"无效的JSON格式: {str(e)}"}
                            client_socket.sendall(json.dumps(response).encode('utf-8'))
                            continue

                        response = self.process_command(command)
                        client_socket.sendall(json.dumps(response).encode('utf-8'))

                    except UnicodeDecodeError:
                        response = {"status": "error", "message": "无法解码UTF-8数据"}
                        client_socket.sendall(json.dumps(response).encode('utf-8'))
                    except socket.timeout:
                        logger.warning(f"客户端{addr}超时未发送数据")
                        response = {"status": "error", "message": "连接超时"}
                        client_socket.sendall(json.dumps(response).encode('utf-8'))
                        break
                    except ConnectionResetError:
                        logger.info(f"客户端{addr}强制断开连接")
                        break
                    except Exception as e:
                        logger.error(f"处理客户端{addr}请求时出错: {str(e)}")
                        response = {"status": "error", "message": f"服务器内部错误: {str(e)}"}
                        try:
                            client_socket.sendall(json.dumps(response).encode('utf-8'))
                        except:
                            pass
                        break
        except Exception as e:
            logger.error(f"客户端{addr}处理线程出错: {str(e)}")
        finally:
            logger.info(f"客户端{addr}连接已关闭")
            with self.lock:
                if handler_thread in self.client_handlers:
                    self.client_handlers.remove(handler_thread)
            
    def process_command(self, command: dict) -> dict:
        """
        处理命令的抽象方法
        子类必须实现这个方法
        """
        raise NotImplementedError("子类必须实现process_command方法")

    def stop(self) -> None:
        """停止服务器"""
        self.running = False
        if self.server_socket:
            try:
                self.server_socket.close()
                logger.info("服务器套接字已关闭")
            except Exception as e:
                logger.error(f"关闭服务器套接字时出错: {e}")
        
        with self.lock:
            for handler in self.client_handlers:
                try:
                    handler.join(timeout=5.0)
                    if handler.is_alive():
                        logger.warning("客户端处理线程未能及时终止")
                except Exception as e:
                    logger.error(f"等待客户端处理线程结束时出错: {e}")
        
        logger.info("服务器已完全停止")


class DroneSocketServer(BaseSocketServer):
    """
    无人机Socket服务器
    集成命令处理器处理无人机相关命令
    """
    def __init__(self, host: str = '0.0.0.0', port: int = 65432, command_processor=None):
        super().__init__(host, port)
        self.command_processor = command_processor

    def process_command(self, command: dict) -> dict:
        """
        使用命令处理器处理命令
        """
        if self.command_processor:
            return self.command_processor.process_command(command)
        else:
            return {"status": "error", "message": "命令处理器未初始化"}
