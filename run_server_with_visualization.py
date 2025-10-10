import os
import sys
import time

# 设置项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入multirotor目录下的setup_path，确保airsim模块能被正确导入
from multirotor import setup_path

# 设置multirotor目录到Python路径，确保AirsimServer等模块能被正确导入
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'multirotor'))

if __name__ == "__main__":
    try:
        print("正在启动AlgorithmServer...")
        # 导入AlgorithmServer模块
        from AlgorithmServer import MultiDroneAlgorithmServer
        
        # 创建并启动服务器实例
        server = MultiDroneAlgorithmServer()
        
        try:
            server.start()
            print("AlgorithmServer已启动。可视化窗口应该会自动打开...")
            print("按Ctrl+C停止服务器...")
            
            # 保持程序运行
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("正在停止AlgorithmServer...")
        finally:
            server.stop()
            print("AlgorithmServer已停止")
            
    except ImportError as e:
        print(f"导入模块时出错: {str(e)}")
        print("请确保所有依赖已安装，并且项目结构正确。")
        sys.exit(1)
    except Exception as e:
        print(f"启动AlgorithmServer时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)