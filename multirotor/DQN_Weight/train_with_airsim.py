"""
使用真实AirSim环境训练DQN模型
连接到AlgorithmServer，使用实际仿真数据进行训练
"""
import os
import sys
import time
import threading
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

print("=" * 60)
print("DQN训练 - 使用真实AirSim环境")
print("=" * 60)

# 检查依赖
print("\n检查依赖...")
try:
    import torch
    print(f"[OK] PyTorch: {torch.__version__}")
except ImportError:
    print("[X] PyTorch未安装")
    print("  安装: pip install torch")
    sys.exit(1)

try:
    from stable_baselines3 import DDPG
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    print(f"[OK] Stable-Baselines3已安装")
except ImportError:
    print("[X] Stable-Baselines3未安装")
    print("  安装: pip install stable-baselines3")
    sys.exit(1)

# 导入项目模块
from simple_weight_env import SimpleWeightEnv

# 导入AlgorithmServer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AlgorithmServer import MultiDroneAlgorithmServer


class TrainingCallback(BaseCallback):
    """训练回调：显示进度和保存最佳模型"""
    
    def __init__(self, total_timesteps, check_freq=1000, save_path='./models/', verbose=1):
        super(TrainingCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf
        self.last_print_step = 0
        self.print_interval = max(total_timesteps // 50, 1000)  # 显示50次进度
        
        # 创建保存目录
        os.makedirs(save_path, exist_ok=True)
        
    def _on_step(self) -> bool:
        # 显示进度
        if self.num_timesteps - self.last_print_step >= self.print_interval:
            progress = (self.num_timesteps / self.total_timesteps) * 100
            bar_length = 50
            filled = int(bar_length * self.num_timesteps / self.total_timesteps)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            # 获取最近的平均奖励
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
                mean_length = np.mean([ep_info['l'] for ep_info in self.model.ep_info_buffer])
            else:
                mean_reward = 0
                mean_length = 0
            
            print(f"\r进度: [{bar}] {progress:.1f}% | "
                  f"步数: {self.num_timesteps}/{self.total_timesteps} | "
                  f"奖励: {mean_reward:.2f} | "
                  f"长度: {mean_length:.0f}", end='', flush=True)
            
            self.last_print_step = self.num_timesteps
            
            # 保存最佳模型
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.verbose > 0:
                    print(f"\n[BEST] 新最佳模型！奖励: {mean_reward:.2f}")
                self.model.save(os.path.join(self.save_path, 'best_model'))
        
        # 定期保存检查点
        if self.num_timesteps % self.check_freq == 0:
            checkpoint_path = os.path.join(self.save_path, f'checkpoint_{self.num_timesteps}')
            self.model.save(checkpoint_path)
            if self.verbose > 1:
                print(f"\n[SAVE] 检查点已保存: {checkpoint_path}")
        
        return True
    
    def _on_training_end(self) -> None:
        print()  # 换行


def start_algorithm_server(drone_names):
    """启动算法服务器（在单独线程中）"""
    print("\n[启动] AlgorithmServer...")
    
    # 创建服务器实例
    server = MultiDroneAlgorithmServer(
        drone_names=drone_names,
        use_learned_weights=False  # 训练时不使用已有模型
    )
    
    # 启动服务器（在单独线程）
    def run_server():
        try:
            # 启动服务器（可视化会自动启动）
            if not server.start():
                print("\n[错误] AlgorithmServer启动失败")
                return
            
            print("[OK] AlgorithmServer已连接")
            
            # 启动任务（让无人机起飞并开始处理数据）
            print("[启动] 开始任务...")
            if not server.start_mission():
                print("\n[错误] 任务启动失败")
                return
            
            print("[运行] 任务已启动，训练可以开始...")
            
            # 保持服务器运行
            while server.running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n[停止] AlgorithmServer收到中断信号")
            server.stop()
        except Exception as e:
            print(f"\n[错误] AlgorithmServer异常: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 启动服务器线程
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # 等待服务器初始化并启动任务
    print("[等待] AlgorithmServer初始化和任务启动...")
    time.sleep(10)  # 增加等待时间，确保连接成功且无人机起飞
    
    return server, server_thread


def main():
    """主训练流程"""
    
    # 配置参数
    DRONE_NAMES = ["UAV1"]  # 训练时使用1个无人机
    TOTAL_TIMESTEPS = 100000  # 训练步数（可调整）
    
    print("\n" + "=" * 60)
    print("步骤1: 启动AirSim环境")
    print("=" * 60)
    print(f"无人机数量: {len(DRONE_NAMES)}")
    print(f"训练步数: {TOTAL_TIMESTEPS}")
    print(f"可视化: 自动启用")
    print("\n[重要] 请确保Unity AirSim仿真已经运行！")
    print("如果Unity未运行，请先启动Unity场景")
    input("准备好后按Enter继续...")
    
    # 启动算法服务器（可视化会自动启动）
    server, server_thread = start_algorithm_server(DRONE_NAMES)
    
    print("\n" + "=" * 60)
    print("步骤2: 创建训练环境")
    print("=" * 60)
    
    # 创建训练环境（连接到真实的server）
    env = SimpleWeightEnv(
        server=server,
        drone_name=DRONE_NAMES[0]  # 使用第一个无人机
    )
    
    print(f"[OK] 环境创建成功")
    print(f"  观察空间: {env.observation_space.shape}")
    print(f"  动作空间: {env.action_space.shape}")
    print(f"  连接到Server: {server is not None}")
    
    print("\n" + "=" * 60)
    print("步骤3: 创建DDPG模型")
    print("=" * 60)
    
    # 添加动作噪声（用于探索）
    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.3 * np.ones(n_actions)  # 降低噪声，因为使用真实环境
    )
    
    # 创建DDPG模型
    model = DDPG(
        "MlpPolicy",
        env,
        action_noise=action_noise,
        learning_rate=1e-4,  # 降低学习率，更稳定
        buffer_size=50000,
        learning_starts=1000,
        batch_size=128,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "episode"),
        gradient_steps=-1,
        verbose=1,
        device='cpu'  # 使用CPU（或改为'cuda'如果有GPU）
    )
    
    print(f"[OK] DDPG模型创建成功")
    print(f"  学习率: {model.learning_rate}")
    print(f"  批次大小: {model.batch_size}")
    print(f"  缓冲区大小: {model.buffer_size}")
    print(f"  设备: {model.device}")
    
    print("\n" + "=" * 60)
    print("步骤4: 开始训练")
    print("=" * 60)
    
    print(f"训练步数: {TOTAL_TIMESTEPS}")
    print(f"预计时间: 约{TOTAL_TIMESTEPS // 1000}分钟")
    print("提示: 按 Ctrl+C 可以随时停止训练")
    print("\n开始训练...\n")
    
    # 创建模型保存目录
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    # 创建回调
    training_callback = TrainingCallback(
        total_timesteps=TOTAL_TIMESTEPS,
        check_freq=5000,  # 每5000步保存一次检查点
        save_path=model_dir,
        verbose=1
    )
    
    # 开始训练
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            log_interval=10,
            callback=training_callback
        )
        print("\n\n[OK] 训练完成！")
        
    except KeyboardInterrupt:
        print("\n\n[中断] 训练被用户中断")
        print("正在保存当前模型...")
    except Exception as e:
        print(f"\n\n[错误] 训练出错: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("步骤5: 保存最终模型")
    print("=" * 60)
    
    # 保存最终模型
    final_model_path = os.path.join(model_dir, 'weight_predictor_airsim')
    model.save(final_model_path)
    print(f"[OK] 最终模型已保存: {final_model_path}.zip")
    
    # 保存最佳模型副本
    best_model_src = os.path.join(model_dir, 'best_model.zip')
    if os.path.exists(best_model_src):
        best_model_dst = os.path.join(model_dir, 'weight_predictor_airsim_best.zip')
        import shutil
        shutil.copy(best_model_src, best_model_dst)
        print(f"[OK] 最佳模型已保存: {best_model_dst}")
    
    print("\n" + "=" * 60)
    print("步骤6: 测试模型")
    print("=" * 60)
    
    # 测试模型
    print("进行5次测试...")
    obs = env.reset()
    
    for i in range(5):
        action, _states = model.predict(obs, deterministic=True)
        print(f"\n测试 {i+1}:")
        print(f"  预测权重: α1={action[0]:.2f}, α2={action[1]:.2f}, "
              f"α3={action[2]:.2f}, α4={action[3]:.2f}, α5={action[4]:.2f}")
        
        obs, reward, done, info = env.step(action)
        print(f"  奖励: {reward:.2f}")
        print(f"  扫描单元格: {info.get('scanned_cells', 0)}")
        
        if done:
            obs = env.reset()
    
    print("\n" + "=" * 60)
    print("训练完成总结")
    print("=" * 60)
    print(f"[OK] 最终模型: {final_model_path}.zip")
    print(f"[OK] 最佳模型: weight_predictor_airsim_best.zip")
    print(f"[OK] 检查点目录: {model_dir}")
    print("\n下一步:")
    print("  1. 测试模型: python test_trained_model.py")
    print("  2. 使用模型: python ../AlgorithmServer.py --use-learned-weights")
    print("  3. 将模型复制为: weight_predictor_simple.zip (系统默认使用)")
    print("=" * 60)
    
    # 停止服务器
    print("\n[停止] AlgorithmServer...")
    server.stop()
    time.sleep(2)
    
    print("\n训练流程全部完成！")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被中断")
    except Exception as e:
        print(f"\n\n发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n按任意键退出...")
        try:
            input()
        except:
            pass

