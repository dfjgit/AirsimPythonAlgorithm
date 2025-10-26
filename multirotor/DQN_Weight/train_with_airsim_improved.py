"""
改进版AirSim环境训练脚本
解决Unity卡死问题
支持Ctrl+C强制退出
"""
import os
import sys
import time
import signal
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 全局标志，用于Ctrl+C处理
training_interrupted = False

def signal_handler(sig, frame):
    """处理Ctrl+C信号"""
    global training_interrupted
    if not training_interrupted:
        print("\n\n" + "=" * 60)
        print("[中断] 检测到Ctrl+C，正在停止训练...")
        print("=" * 60)
        training_interrupted = True
    else:
        print("\n[强制退出] 再次按Ctrl+C将强制退出程序")
        sys.exit(1)

# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)

print("=" * 60)
print("DQN训练 - 改进版（防止Unity卡死）")
print("=" * 60)

# 检查依赖
print("\n检查依赖...")
try:
    import torch
    from stable_baselines3 import DDPG
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.callbacks import BaseCallback
    print("[OK] 依赖检查通过")
except ImportError as e:
    print(f"[错误] 缺少依赖: {e}")
    input("按Enter退出...")
    sys.exit(1)

# 导入项目模块
from simple_weight_env import SimpleWeightEnv

# 导入AlgorithmServer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AlgorithmServer import MultiDroneAlgorithmServer


class ImprovedTrainingCallback(BaseCallback):
    """改进的训练回调，减少输出频率，支持Ctrl+C中断"""
    
    def __init__(self, total_timesteps, check_freq=5000, save_path='./models/', verbose=1):
        super(ImprovedTrainingCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf
        self.last_print_step = 0
        self.print_interval = max(total_timesteps // 20, 1000)  # 只显示20次
        
        os.makedirs(save_path, exist_ok=True)
        
    def _on_step(self) -> bool:
        # 检查是否被中断
        global training_interrupted
        if training_interrupted:
            print("\n[回调] 检测到中断信号，停止训练...")
            return False  # 返回False停止训练
        
        # 减少打印频率，避免阻塞
        if self.num_timesteps - self.last_print_step >= self.print_interval:
            progress = (self.num_timesteps / self.total_timesteps) * 100
            
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
            else:
                mean_reward = 0
            
            print(f"[训练] {progress:.1f}% | 步数: {self.num_timesteps} | 奖励: {mean_reward:.2f}")
            self.last_print_step = self.num_timesteps
            
            # 保存最佳模型
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                print(f"[最佳] 奖励: {mean_reward:.2f}")
                self.model.save(os.path.join(self.save_path, 'best_model'))
        
        # 定期保存检查点
        if self.num_timesteps % self.check_freq == 0:
            checkpoint_path = os.path.join(self.save_path, f'checkpoint_{self.num_timesteps}')
            self.model.save(checkpoint_path)
            print(f"[检查点] 已保存: checkpoint_{self.num_timesteps}")
        
        return True  # 继续训练


def main():
    """主训练流程"""
    
    DRONE_NAMES = ["UAV1"]
    TOTAL_TIMESTEPS = 50000  # 减少步数，避免长时间运行
    
    # 全局变量，用于清理
    server = None
    
    print("\n" + "=" * 60)
    print("步骤1: 启动AirSim环境")
    print("=" * 60)
    print(f"训练步数: {TOTAL_TIMESTEPS}")
    print("\n[重要] 请确保Unity AirSim仿真已经运行！")
    
    confirm = input("Unity已运行？(Y/N): ").strip().upper()
    if confirm != 'Y':
        print("请先启动Unity")
        return
    
    try:
        print("\n[1/5] 启动AlgorithmServer...")
        
        # 创建服务器（不使用可视化，减少负载）
        server = MultiDroneAlgorithmServer(
            drone_names=DRONE_NAMES,
            use_learned_weights=False
        )
    
        # 启动服务器
        if not server.start():
            print("[错误] AlgorithmServer启动失败")
            return
        
        print("[OK] AlgorithmServer已连接")
        
        # 只让无人机起飞，不启动算法线程（避免冲突）
        print("\n[2/5] 让无人机起飞...")
        print("[重要] 训练模式：不启动算法线程，避免与训练环境冲突")
        
        # 手动起飞，不调用start_mission()
        for drone_name in DRONE_NAMES:
            print(f"  起飞 {drone_name}...")
            if not server.drone_controller.takeoff(drone_name):
                print(f"[错误] {drone_name}起飞失败")
                server.stop()
                return
        
        print("[OK] 无人机已起飞（未启动算法线程）")
        
        # 等待系统稳定
        print("\n[3/5] 等待系统稳定...")
        time.sleep(5)
        
        # 创建训练环境
        print("\n[4/5] 创建训练环境...")
        env = SimpleWeightEnv(
            server=server,
            drone_name=DRONE_NAMES[0]
        )
        print(f"[OK] 环境创建成功")
        
        # 创建DDPG模型（降低复杂度）
        print("\n[5/5] 创建DDPG模型...")
        
        n_actions = env.action_space.shape[0]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.2 * np.ones(n_actions)  # 降低噪声
        )
        
        model = DDPG(
            "MlpPolicy",
            env,
            action_noise=action_noise,
            learning_rate=1e-4,
            buffer_size=10000,      # 减小缓冲区
            learning_starts=500,     # 更早开始学习
            batch_size=64,          # 减小批次
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "episode"),
            gradient_steps=-1,
            verbose=0,              # 减少日志输出
            device='cpu'
        )
        
        print("[OK] DDPG模型创建成功")
        
        # 开始训练
        print("\n" + "=" * 60)
        print("开始训练")
        print("=" * 60)
        print(f"训练步数: {TOTAL_TIMESTEPS}")
        print("按 Ctrl+C 可以随时停止训练")
        print("如果Ctrl+C无效，请关闭此窗口\n")
        
        model_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        training_callback = ImprovedTrainingCallback(
            total_timesteps=TOTAL_TIMESTEPS,
            check_freq=5000,
            save_path=model_dir,
            verbose=1
        )
        
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            log_interval=None,  # 关闭日志
            callback=training_callback
        )
        print("\n[OK] 训练完成！")
        
        # 保存最终模型
        print("\n保存模型...")
        final_model_path = os.path.join(model_dir, 'weight_predictor_airsim')
        model.save(final_model_path)
        print(f"[OK] 模型已保存: {final_model_path}.zip")
        
        print("\n训练流程完成！")
        print("\n下一步:")
        print("  1. 测试模型: python test_trained_model.py")
        print("  2. 使用模型: python ../AlgorithmServer.py --use-learned-weights")
        
    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("[中断] 正在停止训练...")
        print("=" * 60)
        print("\n请稍候，正在清理资源...")
        
    except Exception as e:
        print(f"\n\n[错误] 训练出错: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 确保清理资源
        if server:
            print("\n停止AlgorithmServer...")
            try:
                # 降落无人机
                for drone_name in DRONE_NAMES:
                    try:
                        print(f"  降落 {drone_name}...")
                        server.drone_controller.land(drone_name)
                    except:
                        pass
                
                # 停止服务器（由于没启动算法线程，这里只是断开连接）
                server.unity_socket.stop()
                print("[OK] AlgorithmServer已停止")
            except Exception as e:
                print(f"[警告] 清理资源时出现错误: {e}")
        
        print("\n训练已结束")
        print("按Enter键退出...")
        try:
            input()
        except:
            pass


if __name__ == "__main__":
    main()

