import sys
import os
import argparse
import logging
import time

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RunAirsimClient")

def main():
    """启动Airsim客户端的主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='运行Airsim客户端算法')
    parser.add_argument('--grid-data', type=str, 
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "hex_grid_data.json"),
                        help='蜂窝网格数据文件路径')
    parser.add_argument('--scanner-data', type=str, 
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "scanner_data.json"),
                        help='扫描器配置数据文件路径')
    parser.add_argument('--mission-duration', type=int, default=30,
                        help='任务持续时间（秒），默认为30秒')
    parser.add_argument('--debug', action='store_true', 
                        help='启用调试日志')
    args = parser.parse_args()
    
    # 如果启用了调试模式，设置日志级别为DEBUG
    if args.debug:
        for handler in logging.root.handlers:
            handler.setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    try:
        # 确保所需的文件存在
        if not os.path.exists(args.grid_data):
            logger.error(f"网格数据文件不存在: {args.grid_data}")
            logger.info("请确保已生成网格数据文件")
            sys.exit(1)
        
        if not os.path.exists(args.scanner_data):
            logger.error(f"扫描器数据文件不存在: {args.scanner_data}")
            logger.info("请确保已生成扫描器数据文件")
            sys.exit(1)
        
        # 导入AirsimClient类
        from airsim_client import AirsimClient
        
        # 创建并初始化客户端
        logger.info(f"正在初始化Airsim客户端...")
        logger.info(f"网格数据文件: {args.grid_data}")
        logger.info(f"扫描器数据文件: {args.scanner_data}")
        client = AirsimClient(args.grid_data, args.scanner_data)
        
        # 连接到Airsim
        logger.info("正在连接到Airsim模拟器...")
        if client.connect():
            # 启动任务
            client.start_mission()
            
            # 运行指定的时间
            logger.info(f"任务已开始，将运行{args.mission_duration}秒...")
            time.sleep(args.mission_duration)
            
            # 停止任务
            logger.info("任务时间结束，正在停止...")
            client.stop_mission()
            
        else:
            logger.error("无法连接到Airsim模拟器，任务无法启动")
            logger.info("请确保Airsim模拟器正在运行")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("用户中断程序")
    except ImportError as e:
        logger.error(f"导入模块时出错: {str(e)}")
        logger.info("请检查Python环境和模块安装")
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保断开连接
        if 'client' in locals():
            logger.info("正在断开与Airsim模拟器的连接...")
            client.disconnect()
        
        logger.info("程序已退出")

if __name__ == "__main__":
    main()