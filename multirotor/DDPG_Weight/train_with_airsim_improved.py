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
from training_visualizer import TrainingVisualizer

# 导入AlgorithmServer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AlgorithmServer import MultiDroneAlgorithmServer


class ImprovedTrainingCallback(BaseCallback):
    """改进的训练回调，突出显示模型和奖励，并更新可视化"""
    
    def __init__(self, total_timesteps, check_freq=1000, save_path='./models/', 
                 training_visualizer=None, verbose=1):
        super(ImprovedTrainingCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.check_freq = check_freq
        self.save_path = save_path
        self.training_visualizer = training_visualizer  # 训练可视化器
        self.best_mean_reward = -np.inf
        self.last_print_step = 0
        self.print_interval = max(total_timesteps // 10, 100)  # 只显示10次
        self.episode_count = 0
        self.episode_rewards = []
        
        os.makedirs(save_path, exist_ok=True)
        
    def _on_step(self) -> bool:
        # 检查是否被中断
        global training_interrupted
        if training_interrupted:
            print("\n[中断] 停止训练...")
            return False  # 返回False停止训练
        
        # 记录episode奖励
        if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer) > self.episode_count:
            ep_reward = self.model.ep_info_buffer[-1]['r']
            ep_length = self.model.ep_info_buffer[-1]['l']
            self.episode_rewards.append(ep_reward)
            self.episode_count = len(self.model.ep_info_buffer)
            
            # 更新训练可视化
            if self.training_visualizer:
                self.training_visualizer.update_training_stats(
                    episode_reward=ep_reward,
                    episode_length=ep_length,
                    is_episode_done=True
                )
            
            print(f"\n{'╔'+'═'*58+'╗'}")
            print(f"║  🎉 Episode #{self.episode_count} 完成！{' '*(45-len(str(self.episode_count)))}║")
            print(f"{'╠'+'═'*58+'╣'}")
            print(f"║  📈 本次奖励: {ep_reward:+8.2f}{' '*40}║")
            print(f"║  📏 Episode长度: {ep_length:4.0f} 步{' '*36}║")
            
            if len(self.episode_rewards) > 1:
                avg_reward = np.mean(self.episode_rewards)
                best_reward = max(self.episode_rewards)
                worst_reward = min(self.episode_rewards)
                print(f"║{' '*58}║")
                print(f"║  📊 统计信息:{' '*43}║")
                print(f"║    • 平均奖励: {avg_reward:+8.2f}{' '*35}║")
                print(f"║    • 最佳奖励: {best_reward:+8.2f}{' '*35}║")
                print(f"║    • 最差奖励: {worst_reward:+8.2f}{' '*35}║")
                
                # 奖励趋势
                if len(self.episode_rewards) >= 3:
                    recent_avg = np.mean(self.episode_rewards[-3:])
                    trend = "📈 上升" if recent_avg > avg_reward else "📉 下降"
                    print(f"║    • 最近趋势: {trend}{' '*35}║")
            
            print(f"║{' '*58}║")
            remaining_steps = self.total_timesteps - self.num_timesteps
            progress = self.num_timesteps / self.total_timesteps * 100
            print(f"║  🎯 训练进度: {self.num_timesteps}/{self.total_timesteps} ({progress:.1f}%){' '*(24-len(str(self.total_timesteps))*2-len(f'{progress:.1f}'))}║")
            print(f"║  ⏳ 剩余步数: {remaining_steps}{' '*(43-len(str(remaining_steps)))}║")
            print(f"{'╚'+'═'*58+'╝'}\n")
            
            # 如果训练还没结束，提示即将开始下一个Episode
            if self.num_timesteps < self.total_timesteps:
                print(f"{'─'*60}")
                print(f"🔄 准备下一个Episode（#{self.episode_count + 1}）...")
                print(f"   环境将自动重置...")
                print(f"{'─'*60}\n")
        
        # 减少打印频率，避免阻塞
        if self.num_timesteps - self.last_print_step >= self.print_interval:
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
            else:
                mean_reward = 0
            
            # 保存最佳模型
            if mean_reward > self.best_mean_reward and mean_reward > 0:
                self.best_mean_reward = mean_reward
                model_path = os.path.join(self.save_path, 'best_model')
                self.model.save(model_path)
                print(f"\n🏆 新最佳模型！奖励: {mean_reward:.2f}")
                print(f"💾 已保存: {model_path}.zip\n")
            
            self.last_print_step = self.num_timesteps
        
        # 定期保存检查点
        if self.num_timesteps % self.check_freq == 0 and self.num_timesteps > 0:
            checkpoint_path = os.path.join(self.save_path, f'checkpoint_{self.num_timesteps}')
            self.model.save(checkpoint_path)
            print(f"💾 检查点: checkpoint_{self.num_timesteps}.zip")
        
        return True  # 继续训练


def main():
    """主训练流程"""
    
    # ==================== 训练参数配置 ====================
    DRONE_NAMES = ["UAV1", "UAV2", "UAV3"]  # 使用4台无人机协同训练
    TOTAL_TIMESTEPS = 100            # 总训练步数（快速训练）
    STEP_DURATION = 20.0             # 每步飞行时长（秒） 提高飞行时长
    CHECKPOINT_FREQ = 1000           # 检查点保存频率
    ENABLE_VISUALIZATION = True      # 是否启用可视化（训练专用可视化）
    # =====================================================
    
    # 全局变量，用于清理
    server = None
    training_visualizer = None
    
    print("\n" + "=" * 60)
    print("🚀 DQN权重训练 - 多无人机协同模式")
    print("=" * 60)
    print(f"🚁 无人机数量: {len(DRONE_NAMES)} 台 ({', '.join(DRONE_NAMES)})")
    print(f"📊 训练步数: {TOTAL_TIMESTEPS} 步")
    print(f"⏱️  每步时长: {STEP_DURATION} 秒")
    print(f"💾 检查点: 每 {CHECKPOINT_FREQ} 步保存一次")
    print(f"👁️  可视化: {'启用' if ENABLE_VISUALIZATION else '禁用'}")
    print(f"📈 预计episode数: ~{TOTAL_TIMESTEPS // 50}")
    print("=" * 60)
    print(f"\n💡 说明: 使用{len(DRONE_NAMES)}台无人机协同训练")
    print(f"   - 主训练无人机: {DRONE_NAMES[0]} (用于DQN学习)")
    print(f"   - 协同无人机: {', '.join(DRONE_NAMES[1:])} (提供环境交互)")
    print(f"   - 学到的权重策略将适用于所有无人机")
    print("\n[重要] 请确保Unity AirSim仿真已经运行！")
    
    confirm = input("Unity已运行？(Y/N): ").strip().upper()
    if confirm != 'Y':
        print("请先启动Unity")
        return
    
    try:
        print("\n[1/5] 启动AlgorithmServer...")
        
        # 创建服务器（训练模式不使用学习的权重，禁用AlgorithmServer自带的可视化）
        server = MultiDroneAlgorithmServer(
            drone_names=DRONE_NAMES,
            use_learned_weights=False,
            model_path=None,  # 训练模式不需要加载模型
            enable_visualization=False  # 禁用AlgorithmServer的可视化，使用训练专用可视化
        )
        
        print(f"✅ 服务器创建成功")
        print(f"  无人机配置: {', '.join(DRONE_NAMES)}")
        print(f"  使用训练专用可视化: {'是' if ENABLE_VISUALIZATION else '否'}")
    
        # 启动服务器
        if not server.start():
            print("[错误] AlgorithmServer启动失败")
            return
        
        print("[OK] AlgorithmServer已连接")
        
        # 启动无人机和算法线程
        print("\n[2/5] 启动无人机任务...")
        print("[重要] 训练模式：启动算法线程，训练环境动态改变权重")
        
        # 调用start_mission()启动完整流程
        if not server.start_mission():
            print("[错误] 任务启动失败")
            server.stop()
            return
        
        print("[OK] 无人机已起飞，算法线程运行中")
        
        # 等待系统稳定
        print("\n[3/5] 等待系统稳定...")
        time.sleep(5)
        
        # 创建训练环境
        print("\n[4/5] 创建训练环境...")
        
        env = SimpleWeightEnv(
            server=server,
            drone_name=DRONE_NAMES[0],  # 使用第一台无人机进行DQN训练
            reset_unity=True,          # 标准episode训练
            step_duration=STEP_DURATION  # 使用配置的飞行时长
        )
        print(f"✅ 环境创建成功")
        print(f"  📋 模式: 多无人机协同训练")
        print(f"  🎓 训练无人机: {DRONE_NAMES[0]}")
        print(f"  🤝 协同无人机: {', '.join(DRONE_NAMES[1:]) if len(DRONE_NAMES) > 1 else '无'}")
        print(f"  ⏱️  每步时长: {STEP_DURATION}秒")
        print(f"  🎯 每个episode: {env.reward_config.max_steps}步 = {env.reward_config.max_steps * STEP_DURATION / 60:.1f}分钟")
        print(f"  💡 预计总训练时长: {TOTAL_TIMESTEPS * STEP_DURATION / 60:.1f}分钟")
        
        # 创建并启动训练专用可视化
        if ENABLE_VISUALIZATION:
            print("\n[4.5/5] 启动训练专用可视化...")
            try:
                training_visualizer = TrainingVisualizer(server=server, env=env)
                if training_visualizer.start_visualization():
                    print("✅ 训练可视化已启动")
                    print("💡 可视化窗口应该会弹出，显示训练统计和环境状态")
                    print("💡 按ESC键可关闭可视化窗口（不影响训练）")
                else:
                    print("⚠️  训练可视化启动失败，但训练将继续")
            except Exception as e:
                print(f"⚠️  训练可视化初始化失败: {str(e)}")
                print("💡 训练将继续，但不显示可视化")
                training_visualizer = None
        
        # 创建DDPG模型
        print("\n[5/5] 创建DDPG模型...")
        
        n_actions = env.action_space.shape[0]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.15 * np.ones(n_actions)  # 适度噪声
        )
        
        model = DDPG(
            "MlpPolicy",
            env,
            action_noise=action_noise,
            learning_rate=1e-4,
            buffer_size=5000,        # 小缓冲区（快速训练）
            learning_starts=200,     # 尽早开始学习
            batch_size=64,
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "episode"),
            gradient_steps=-1,
            verbose=0,
            device='cpu'
        )
        
        print("✅ DDPG模型创建成功")
        
        # 开始训练
        print("\n" + "=" * 60)
        print("🎯 开始训练")
        print("=" * 60)
        print(f"📊 训练步数: {TOTAL_TIMESTEPS}")
        print(f"⏸️  按 Ctrl+C 可随时停止")
        print("=" * 60 + "\n")
        
        model_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        training_callback = ImprovedTrainingCallback(
            total_timesteps=TOTAL_TIMESTEPS,
            check_freq=CHECKPOINT_FREQ,
            save_path=model_dir,
            training_visualizer=training_visualizer,  # 传入可视化器
            verbose=1
        )
        
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            log_interval=None,
            callback=training_callback
        )
        
        print("\n" + "=" * 60)
        print("✅ 训练完成！")
        print("=" * 60)
        
        # 保存最终模型
        print("\n💾 保存最终模型...")
        final_model_path = os.path.join(model_dir, 'weight_predictor_airsim')
        model.save(final_model_path)
        print(f"✅ 模型已保存: {final_model_path}.zip")
        
        # 显示训练统计
        print("\n" + "=" * 60)
        print("📊 训练统计")
        print("=" * 60)
        if hasattr(training_callback, 'episode_rewards') and training_callback.episode_rewards:
            print(f"完成episode数: {len(training_callback.episode_rewards)}")
            print(f"总奖励: {sum(training_callback.episode_rewards):.2f}")
            print(f"平均奖励: {np.mean(training_callback.episode_rewards):.2f}")
            print(f"最佳奖励: {max(training_callback.episode_rewards):.2f}")
            print(f"最差奖励: {min(training_callback.episode_rewards):.2f}")
        print("=" * 60)
        
        print("\n📦 生成的模型文件:")
        print(f"  🏆 最佳模型: models/best_model.zip")
        print(f"  📄 最终模型: models/weight_predictor_airsim.zip")
        if CHECKPOINT_FREQ > 0:
            print(f"  💾 检查点: models/checkpoint_*.zip")
        
        print("\n🎯 下一步操作:")
        print("  1️⃣  测试模型: python test_trained_model.py")
        print("  2️⃣  使用模型: python ../AlgorithmServer.py --use-learned-weights")
        print("=" * 60)
        
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
        
        # 停止可视化
        if training_visualizer:
            print("\n停止训练可视化...")
            try:
                training_visualizer.stop_visualization()
                print("[OK] 训练可视化已停止")
            except Exception as e:
                print(f"[警告] 停止可视化时出错: {e}")
        
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

