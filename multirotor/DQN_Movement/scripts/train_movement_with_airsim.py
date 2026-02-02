"""
DQN训练脚本 - 与AirSim集成
使用真实的AirSim环境训练无人机移动策略
"""
import os
import sys
import numpy as np
import json
from datetime import datetime
import threading
import time

# 添加项目路径
# scripts -> DQN_Movement -> multirotor -> 项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

# 添加 multirotor 目录
multirotor_dir = os.path.join(project_root, 'multirotor')
sys.path.insert(0, multirotor_dir)

# 添加 DQN_Movement 目录
dqn_movement_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, dqn_movement_dir)

print("=" * 80)
print("DQN训练 - 无人机移动控制 (AirSim集成)")
print("=" * 80)

# 检查依赖
print("\n[步骤1] 检查依赖...")
try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    import gymnasium
    print(f"  ✓ Stable-Baselines3已安装")
except ImportError:
    print("  ✗ Stable-Baselines3未安装")
    print("    安装命令: pip install stable-baselines3 gymnasium")
    sys.exit(1)

# 导入环境和服务器
from envs.movement_env import MovementEnv, MultiDroneMovementEnv
from AlgorithmServer import MultiDroneAlgorithmServer
from Algorithm.drones_config import DronesConfig

print("\n" + "=" * 80)
print("[步骤2] 加载配置并确定训练无人机")
print("=" * 80)

# 加载无人机配置
drones_config = DronesConfig()
print(f"  ✓ 加载 drones_config.json")
print(f"    - 所有无人机: {drones_config.get_all_drones()}")
print(f"    - 启用的无人机: {drones_config.get_enabled_drones()}")

# 获取 DQN 训练使用的无人机列表
drone_names = drones_config.get_training_drones('dqn')
print(f"  ✓ DQN训练使用的无人机: {drone_names}")

if not drone_names:
    print(f"  ✗ 错误: 没有可用的训练无人机，请检查 drones_config.json")
    sys.exit(1)

# 显示无人机类型（虚拟/实体）
print(f"  \n  训练无人机详情:")
for drone in drone_names:
    drone_type_str = drones_config.get_drone_type(drone)
    is_crazyflie = drones_config.is_crazyflie_mirror(drone)
    type_display = "实体无人机(Crazyflie)" if is_crazyflie else "虚拟无人机(AirSim)"
    print(f"    - {drone}: {type_display}")

# 加载 DQN 训练配置
dqn_config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "movement_dqn_config.json")
with open(dqn_config_path, 'r', encoding='utf-8') as f:
    dqn_config = json.load(f)
print(f"  ✓ 加载 movement_dqn_config.json")

# scanner_config.json 路径（AlgorithmServer需要）
config_file = os.path.join(os.path.dirname(__file__), "..", "scanner_config.json")
if not os.path.exists(config_file):
    print(f"  ✗ scanner_config.json 不存在: {config_file}")
    sys.exit(1)

print("\n" + "=" * 80)
print("[步骤3] 启动AirSim服务器")
print("=" * 80)

# 创建算法服务器（使用DQN控制模式）
print(f"  正在启动服务器 (DQN控制模式)...")
server = MultiDroneAlgorithmServer(
    config_file=config_file,
    drone_names=drone_names,  # 使用从配置文件读取的无人机列表
    use_learned_weights=False,
    control_mode='dqn'  # 关键：使用DQN控制模式
)

# 启动服务器（连接Unity和AirSim）
print(f"  连接Unity和AirSim...")
if not server.start():
    print(f"  ✗ 服务器启动失败")
    sys.exit(1)

print(f"  ✓ 服务器启动成功")

# 关键：启动任务（让无人机起飞并启动算法线程）
print(f"  启动无人机任务...")
if not server.start_mission():
    print(f"  ✗ 无人机任务启动失败")
    sys.exit(1)

print(f"  ✓ 无人机任务启动成功")

# [步骤3.5] 设置实验元数据 (用于跨方案数据对比)
if hasattr(server, 'set_experiment_meta'):
    server.set_experiment_meta(
        algorithm_type='pure_dqn',
        env_type='movement',
        control_mode='dqn'
    )

print("\n" + "=" * 80)
print("[步骤4] 创建训练环境")
print("=" * 80)

# 根据无人机数量选择环境类型
if len(drone_names) == 1:
    # 单机训练
    training_drone = drone_names[0]
    print(f"  模式: 单机训练")
    print(f"  训练无人机: {training_drone}")
    
    env = MovementEnv(server=server, drone_name=training_drone, config_path=dqn_config_path)
    env = Monitor(env)
    
else:
    # 多机训练（参数共享）
    print(f"  模式: 多机训练（参数共享）")
    print(f"  训练无人机: {drone_names}")
    print(f"  无人机数量: {len(drone_names)}")
    print(f"  \n  多机训练详情:")
    for drone in drone_names:
        drone_type = "实体" if drones_config.is_crazyflie_mirror(drone) else "虚拟"
        print(f"    - {drone}: {drone_type}无人机")
    
    env = MultiDroneMovementEnv(server=server, drone_names=drone_names, config_path=dqn_config_path)
    env = Monitor(env)

print(f"  ✓ 环境创建成功")
print(f"    - 观察空间: {env.observation_space.shape}")
print(f"    - 动作空间: {env.action_space.n} (6方向)")
print(f"    - 连接到服务器: {server.running}")

print("\n" + "=" * 80)
print("[步骤5] 创建或加载DQN模型")
print("=" * 80)

# 检查是否有预训练模型
model_dir = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(model_dir, exist_ok=True)

# 日志目录（提前创建）
log_dir = os.path.join(os.path.dirname(__file__), 'logs', 'movement_dqn_airsim')
os.makedirs(log_dir, exist_ok=True)

pretrained_model = os.path.join(model_dir, 'movement_dqn_final.zip')
use_pretrained = os.path.exists(pretrained_model)

if use_pretrained:
    print(f"  ✓ 找到预训练模型: {pretrained_model}")
    print(f"  加载预训练模型继续训练...")
    model = DQN.load(pretrained_model, env=env)
    # 启用 TensorBoard 日志
    model.tensorboard_log = log_dir
    print(f"  ✓ 预训练模型加载成功")
    print(f"  ✓ TensorBoard 日志: {log_dir}")
else:
    print(f"  创建新模型...")
    model = DQN(
        dqn_config['model']['policy'],
        env,
        learning_rate=dqn_config['training']['learning_rate'],
        buffer_size=dqn_config['training']['buffer_size'],
        learning_starts=dqn_config['training']['learning_starts'],
        batch_size=dqn_config['training']['batch_size'],
        tau=dqn_config['training']['tau'],
        gamma=dqn_config['training']['gamma'],
        target_update_interval=dqn_config['training']['target_update_interval'],
        exploration_fraction=dqn_config['training']['exploration_fraction'],
        exploration_initial_eps=dqn_config['training']['exploration_initial_eps'],
        exploration_final_eps=dqn_config['training']['exploration_final_eps'],
        policy_kwargs=dict(net_arch=dqn_config['model']['net_arch']),
        verbose=1,
        tensorboard_log=log_dir  # 启用 TensorBoard 日志
    )
    print(f"  ✓ 新模型创建成功")
    print(f"  ✓ TensorBoard 日志: {log_dir}")

print("\n" + "=" * 80)
print("[步骤6] 设置训练回调")
print("=" * 80)

# 自定义回调
class AirSimProgressCallback(BaseCallback):
    """一步一步加载训练进度回调"""
    
    def __init__(self, total_timesteps, print_freq=500, log_dir=None, verbose=0):
        super(AirSimProgressCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.print_freq = print_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_scanned = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.episode_count = 0
        self.start_time = datetime.now()
        
        # CSV 日志记录
        self.log_dir = log_dir
        if self.log_dir:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.csv_path = os.path.join(self.log_dir, f'dqn_training_{timestamp}.csv')
            # 创建 CSV 文件并写入表头
            with open(self.csv_path, 'w', encoding='utf-8') as f:
                f.write('episode,reward,length,scanned_cells,timestep,elapsed_time,timestamp,collision_count,out_of_range_count,scan_efficiency\n')
            print(f"  ✓ CSV 日志: {self.csv_path}")
        
    def _on_step(self) -> bool:
        # 累计统计
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1
        
        # [DEBUG] 打印 self.locals 的键，帮助诊断
        if self.num_timesteps == 1:
            print(f"\n[DEBUG] 回调函数首次调用，self.locals 的键: {list(self.locals.keys())}")
            print(f"[DEBUG] 检查 episode 结束标志:")
            print(f"  - 'dones' in locals: {'dones' in self.locals}")
            print(f"  - 'terminations' in locals: {'terminations' in self.locals}")
            if 'dones' in self.locals:
                print(f"  - dones 类型: {type(self.locals['dones'])}, 值: {self.locals['dones']}")
            if 'terminations' in self.locals:
                print(f"  - terminations 类型: {type(self.locals['terminations'])}, 值: {self.locals['terminations']}")
            if 'infos' in self.locals:
                print(f"  - infos[0] 的键: {list(self.locals['infos'][0].keys()) if len(self.locals['infos']) > 0 else 'empty'}")
        
        # episode结束检测 - 多种方法
        is_done = False
        done_method = None
        
        # 方法1: 检查 'dones' (旧版 Gym API)
        if 'dones' in self.locals and len(self.locals['dones']) > 0:
            is_done = bool(self.locals['dones'][0])
            if is_done:
                done_method = 'dones'
        
        # 方法2: 检查 'terminations' 和 'truncations' (新版 Gymnasium API)
        if not is_done and 'terminations' in self.locals:
            terminated = bool(self.locals['terminations'][0]) if len(self.locals['terminations']) > 0 else False
            truncated = bool(self.locals.get('truncations', [False])[0]) if 'truncations' in self.locals and len(self.locals.get('truncations', [])) > 0 else False
            is_done = terminated or truncated
            if is_done:
                done_method = f'terminations({terminated})/truncations({truncated})'
        
        # 方法3: 从 infos 中检测 (最可靠的方法)
        if not is_done and 'infos' in self.locals and len(self.locals['infos']) > 0:
            info = self.locals['infos'][0]
            # 检查多种可能的 episode 结束标志
            if 'TimeLimit.truncated' in info or '_final_observation' in info or 'terminal_observation' in info:
                is_done = True
                done_method = 'infos'
        
        if is_done:
            self.episode_count += 1
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            print(f"\n[DEBUG] 检测到 episode 结束 (通过 {done_method})")
            print(f"[DEBUG] Episode {self.episode_count} 完成:")
            print(f"  - Reward: {self.current_episode_reward:.2f}")
            print(f"  - Length: {self.current_episode_length}")
            print(f"  - Timestep: {self.num_timesteps}")
            
            # 获取扫描信息和其他指标
            scanned_cells = 0
            collision_count = 0
            out_of_range_count = 0
            
            if 'infos' in self.locals and len(self.locals['infos']) > 0:
                info = self.locals['infos'][0]
                scanned_cells = info.get('scanned_cells', 0)
                collision_count = info.get('collision_count', 0)
                out_of_range_count = info.get('out_of_range_count', 0)
                
                self.episode_scanned.append(scanned_cells)
                print(f"  - Scanned cells: {scanned_cells}")
                print(f"  - Collisions: {collision_count}")
                print(f"  - Out of range: {out_of_range_count}")
            
            # 计算耗时
            elapsed_time = (datetime.now() - self.start_time).total_seconds()
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            scan_efficiency = scanned_cells / max(elapsed_time, 1.0)
            
            # 写入 CSV 日志
            if self.log_dir:
                try:
                    print(f"[DEBUG] 准备写入 CSV: {self.csv_path}")
                    with open(self.csv_path, 'a', encoding='utf-8') as f:
                        line = f'{self.episode_count},{self.current_episode_reward:.2f},{self.current_episode_length},{scanned_cells},{self.num_timesteps},{elapsed_time:.2f},{timestamp_str},{collision_count},{out_of_range_count},{scan_efficiency:.2f}\n'
                        f.write(line)
                        f.flush()  # 确保立即写入磁盘
                        print(f"  [✅] 成功写入 CSV: {line.strip()}")
                except Exception as e:
                    print(f"  [❌] 写入 CSV 失败: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"[DEBUG] log_dir 为空，跳过 CSV 写入")
            
            self.current_episode_reward = 0
            self.current_episode_length = 0
        
        # 定期打印
        if self.num_timesteps % self.print_freq == 0:
            progress = (self.num_timesteps / self.total_timesteps) * 100
            
            print(f"\n{'=' * 60}")
            print(f"进度: {progress:.1f}% ({self.num_timesteps}/{self.total_timesteps})")
            
            if len(self.episode_rewards) > 0:
                recent_n = min(10, len(self.episode_rewards))
                avg_reward = np.mean(self.episode_rewards[-recent_n:])
                avg_length = np.mean(self.episode_lengths[-recent_n:])
                
                print(f"最近{recent_n}个episodes:")
                print(f"  平均奖励: {avg_reward:.2f}")
                print(f"  平均步数: {avg_length:.1f}")
                
                if len(self.episode_scanned) > 0:
                    avg_scanned = np.mean(self.episode_scanned[-recent_n:])
                    print(f"  平均扫描: {avg_scanned:.1f}个单元格")
            
            print(f"{'=' * 60}")
        
        return True

# 检查点回调
checkpoint_callback = CheckpointCallback(
    save_freq=5000,
    save_path=model_dir,
    name_prefix='movement_dqn_airsim'
)

# 进度回调
progress_callback = AirSimProgressCallback(
    total_timesteps=dqn_config['training']['total_timesteps'],
    print_freq=500,
    log_dir=log_dir  # 传入日志目录，启用 CSV 记录
)

print(f"  ✓ 回调设置完成")

print("\n" + "=" * 80)
print("[步骤7] 开始训练")
print("=" * 80)

total_timesteps = dqn_config['training']['total_timesteps']
print(f"训练步数: {total_timesteps}")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\n⚠ 请确保Unity客户端已连接到服务器")
print(f"按 Ctrl+C 可以随时中断训练并保存模型\n")

try:
    # 开始训练
    print(f"\n[DEBUG] 即将调用 model.learn()...")
    print(f"[DEBUG] total_timesteps = {total_timesteps}")
    print(f"[DEBUG] learning_starts = {dqn_config['training']['learning_starts']}")
    
    # 注：删除了测试性的 env.reset() 调用，因为它会在无人机已移动后触发重置
    # 现在领导者在接收到 start_simulation 指令后才开始移动，无需额外重置
    
    print(f"[DEBUG] \n开始训练循环...\n")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, progress_callback],
        log_interval=10
    )
    
    print("\n" + "=" * 80)
    print("✓ 训练完成！")
    print("=" * 80)
    
except KeyboardInterrupt:
    print("\n\n训练被用户中断")
    print("正在保存当前模型...")
except Exception as e:
    print(f"\n\n✗ 训练出错: {str(e)}")
    import traceback
    traceback.print_exc()
    print("正在保存当前模型...")

print("\n" + "=" * 80)
print("[步骤8] 保存最终模型")
print("=" * 80)

# 保存模型
final_model_path = os.path.join(model_dir, 'movement_dqn_airsim_final')
model.save(final_model_path)
print(f"  ✓ 模型已保存: {final_model_path}.zip")

print("\n" + "=" * 80)
print("[步骤9] 清理")
print("=" * 80)

# 停止服务器
print(f"  正在停止服务器...")
server.stop()
time.sleep(1)
print(f"  ✓ 服务器已停止")

print("\n" + "=" * 80)
print("训练完成总结")
print("=" * 80)
print(f"✓ 最终模型: {final_model_path}.zip")
print(f"✓ 训练日志: {log_dir}")
print(f"✓ 检查点: {model_dir}")
print(f"✓ 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\n下一步:")
print(f"  1. 查看Tensorboard: tensorboard --logdir={log_dir}")
print(f"  2. 测试模型: python test_movement_dqn.py")
print("=" * 80)

