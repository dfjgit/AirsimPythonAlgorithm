"""
虚实融合训练脚本 - 独立模块

功能说明：
    - 在AirSim仿真环境中使用DDPG算法训练APF权重系数
    - 支持虚实融合训练：部分无人机使用实体机数据（isCrazyflieMirror=True）
    - 自动管理配置文件中的isCrazyflieMirror设置，无需手动修改
    - 支持多无人机协同训练模式
    - 集成训练可视化模块，实时显示训练进度和统计信息
    - 支持从已有权重继续训练
    - 自动保存最佳模型和检查点

虚实融合训练原理：
    - 将scanner_config.json中指定无人机的isCrazyflieMirror设置为true
    - 这些无人机将使用实体Crazyflie的状态数据（通过_crazyflie_get_state_for_prediction）
    - 其他无人机仍使用虚拟AirSim环境数据
    - 实现虚拟环境与实体机的融合训练

使用方法：
    python train_with_hybrid.py --config hybrid_train_config.json
    python train_with_hybrid.py --mirror-drones UAV1 UAV2 --total-timesteps 1000

日期：2026-01-23
"""
import os
import sys
import time
import signal
import argparse
import json
import numpy as np
import shutil
import subprocess
from pathlib import Path

# 添加项目根目录到Python路径，以便导入项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ==================== 全局变量 ====================
# 全局标志，用于Ctrl+C中断处理
training_interrupted = False
# ==================================================

def signal_handler(sig, frame):
    """处理Ctrl+C中断信号"""
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
print("虚实融合训练 - DDPG权重APF训练")
print("=" * 60)

# ==================== 依赖检查 ====================
print("\n检查依赖...")
try:
    import torch
    from stable_baselines3 import DDPG
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.callbacks import BaseCallback
    print("[OK] 依赖检查通过")
except ImportError as e:
    print(f"[错误] 缺少依赖: {e}")
    print("请运行: pip install stable-baselines3 torch")
    input("按Enter退出...")
    sys.exit(1)
# ==================================================

# ==================== 导入项目模块 ====================
from envs.simple_weight_env import SimpleWeightEnv

# 独立进程可视化 (pygame 不与训练进程共享)
try:
    from multirotor.Visualization.visualization_ipc import VisualizationIPCServer
    HAS_EXT_VIS = True
except Exception:
    HAS_EXT_VIS = False
    print("警告: 无法导入 VisualizationIPCServer，独立可视化将被禁用")

from envs.crazyflie_data_logger import CrazyflieDataLogger  # 实体无人机数据记录器
from AlgorithmServer import MultiDroneAlgorithmServer
from Algorithm.scanner_config_data import ScannerConfigData
# ==================================================


def _load_train_config(path: str) -> dict:
    """
    加载训练配置文件
    
    功能：
        从 JSON 文件读取训练配置参数
        支持两种格式：
        1. 传统格式：直接返回配置字典
        2. 统一格式：包含 common 和模式专用配置，自动合并
    """
    if not path:
        return {}
    if not os.path.exists(path):
        print(f"⚠️  配置文件不存在: {path}")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        
        # 检查是否为统一配置格式（包含 common 和 hybrid 键）
        if "common" in data and "hybrid" in data:
            # 统一配置格式：合并 common 和 hybrid 配置
            merged_config = {}
            merged_config.update(data.get("common", {}))
            merged_config.update(data.get("hybrid", {}))
            return merged_config
        else:
            # 传统配置格式：直接返回
            return data
    except Exception as exc:
        print(f"⚠️  配置文件读取失败: {exc}")
        return {}


def _get_config_value(cli_value, config: dict, key: str, default):
    """获取配置值（优先级：命令行 > 配置文件 > 默认值）"""
    if cli_value is not None:
        return cli_value
    if key in config:
        return config[key]
    return default


def _setup_hybrid_config(config_file: str, mirror_drones: list) -> str:
    """
    设置虚实融合配置
    
    功能：
        1. 备份原始配置文件
        2. 加载配置文件
        3. 设置指定无人机的isCrazyflieMirror=True
        4. 保存修改后的配置到临时文件
        5. 返回临时配置文件路径
        
    参数：
        config_file: 原始配置文件路径
        mirror_drones: 需要设置为实体镜像的无人机列表（如["UAV1", "UAV2"]）
        
    返回：
        str: 临时配置文件路径
    """
    if not mirror_drones:
        # 如果没有指定镜像无人机，直接返回原配置
        return config_file
    
    # 加载原始配置
    config_data = ScannerConfigData(config_file)
    
    # 设置镜像无人机
    print(f"\n🔧 配置虚实融合训练...")
    print(f"   原始配置文件: {config_file}")
    print(f"   实体镜像无人机: {', '.join(mirror_drones)}")
    
    # 使用DronesConfig加载无人机配置
    from Algorithm.drones_config import DronesConfig
    drones_config = DronesConfig()
    
    # 更新drones_config.json中的镜像设置
    for drone_name in drones_config.get_all_drones():
        is_mirror = drone_name in mirror_drones
        drone_info = drones_config.get_drone_info(drone_name)
        if drone_info:
            drone_info['isCrazyflieMirror'] = is_mirror
            print(f"   ✅ {drone_name}: isCrazyflieMirror = {is_mirror}")
    
    # 保存更新后的配置
    drones_config.save_config()
    print(f"   💾 无人机配置已更新: drones_config.json")
    
    # 创建临时配置文件
    temp_config_dir = os.path.join(os.path.dirname(__file__), "temp_configs")
    os.makedirs(temp_config_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    temp_config_file = os.path.join(temp_config_dir, f"hybrid_config_{timestamp}.json")
    
    # 保存修改后的配置（不再保存droneSettings）
    config_dict = {
        "repulsionCoefficient": config_data.repulsionCoefficient,
        "entropyCoefficient": config_data.entropyCoefficient,
        "distanceCoefficient": config_data.distanceCoefficient,
        "leaderRangeCoefficient": config_data.leaderRangeCoefficient,
        "directionRetentionCoefficient": config_data.directionRetentionCoefficient,
        "groundRepulsionCoefficient": config_data.groundRepulsionCoefficient,
        "updateInterval": config_data.updateInterval,
        "moveSpeed": config_data.moveSpeed,
        "rotationSpeed": config_data.rotationSpeed,
        "scanRadius": config_data.scanRadius,
        "maxRepulsionDistance": config_data.maxRepulsionDistance,
        "minSafeDistance": config_data.minSafeDistance,
        "avoidRevisits": config_data.avoidRevisits,
        "targetSearchRange": config_data.targetSearchRange,
        "revisitCooldown": config_data.revisitCooldown,
        "altitude": config_data.altitude,
        "name": config_data.name,
        "hideFlags": config_data.hideFlags
    }
    
    with open(temp_config_file, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=2)
    
    print(f"   💾 临时配置文件: {temp_config_file}")
    return temp_config_file


def _restore_config(original_config: str, temp_config: str):
    """恢复原始配置文件（训练结束后）"""
    if temp_config != original_config and os.path.exists(temp_config):
        try:
            os.remove(temp_config)
            print(f"✅ 已清理临时配置文件: {temp_config}")
        except Exception as e:
            print(f"⚠️  清理临时配置文件失败: {e}")


def _save_final_weights(server, path: str) -> None:
    """保存各无人机最后的权重系数到JSON文件"""
    if not server or not path:
        return
    weights_by_drone = {}
    for drone_name in server.drone_names:
        algo = server.algorithms.get(drone_name)
        if not algo:
            continue
        weights_by_drone[drone_name] = algo.get_current_coefficients()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(weights_by_drone, f, ensure_ascii=False, indent=2)
        print(f"✅ 初始权重已保存: {path}")
    except Exception as exc:
        print(f"⚠️  保存初始权重失败: {exc}")


def _derive_weights_path(model_path: str) -> str:
    """根据模型路径推导权重文件路径"""
    if not model_path:
        return ""
    if model_path.endswith('.zip'):
        model_path = model_path[:-4]
    return f"{model_path}.json"


def _load_initial_weights(path: str) -> dict:
    """加载初始权重"""
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"⚠️  读取初始权重失败: {exc}")
        return {}

    if not isinstance(data, dict):
        return {}

    keys = [
        "repulsionCoefficient",
        "entropyCoefficient",
        "distanceCoefficient",
        "leaderRangeCoefficient",
        "directionRetentionCoefficient"
    ]
    if all(k in data for k in keys):
        return {"__all__": data}

    return {k: v for k, v in data.items() if isinstance(v, dict)}


def _weights_to_action(weights: dict) -> np.ndarray:
    """将权重字典转换为动作向量"""
    return np.array([
        float(weights.get("repulsionCoefficient", 0.0)),
        float(weights.get("entropyCoefficient", 0.0)),
        float(weights.get("distanceCoefficient", 0.0)),
        float(weights.get("leaderRangeCoefficient", 0.0)),
        float(weights.get("directionRetentionCoefficient", 0.0))
    ], dtype=np.float32)


class ImprovedTrainingCallback(BaseCallback):
    """改进的训练回调类（与train_with_airsim_improved.py相同）"""
    
    def __init__(self, total_timesteps, check_freq=1000, save_path='./models/', 
                 training_visualizer=None, data_logger=None, server=None, mirror_drones=None, verbose=1):
        super(ImprovedTrainingCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.check_freq = check_freq
        self.save_path = save_path
        self.training_visualizer = training_visualizer
        self.data_logger = data_logger  # 数据记录器
        self.server = server  # AlgorithmServer 实例，用于获取实体无人机数据
        self.mirror_drones = mirror_drones or []  # 镜像无人机列表
        self.best_mean_reward = -np.inf
        self.last_print_step = 0
        self.print_interval = max(total_timesteps // 10, 100)
        self.episode_count = 0
        self.episode_rewards = []
        os.makedirs(save_path, exist_ok=True)
        
    def _on_step(self) -> bool:
        global training_interrupted
        if training_interrupted:
            print("\n[中断] 停止训练...")
            return False
        
        if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer) > self.episode_count:
            ep_reward = self.model.ep_info_buffer[-1]['r']
            ep_length = self.model.ep_info_buffer[-1]['l']
            self.episode_rewards.append(ep_reward)
            self.episode_count = len(self.model.ep_info_buffer)
            
            if self.training_visualizer:
                self.training_visualizer.update_training_stats(
                    episode_reward=ep_reward,
                    episode_length=ep_length,
                    is_episode_done=True
                )
            
            # 记录 Episode 统计信息到数据记录器 (仅更新全局统计，Episode 切换由 Env 触发)
            if hasattr(self, 'data_logger') and self.data_logger:
                self.data_logger.record_episode_stats(
                    episode=self.episode_count,
                    reward=ep_reward,
                    length=ep_length
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
                
                if len(self.episode_rewards) >= 3:
                    recent_avg = np.mean(self.episode_rewards[-3:])
                    trend = "📈 上升" if recent_avg > avg_reward else "📉 下降"
                    print(f"║    • 最近趋势: {trend}{' '*35}║")
            
            remaining_steps = self.total_timesteps - self.num_timesteps
            progress = self.num_timesteps / self.total_timesteps * 100
            print(f"║  🎯 训练进度: {self.num_timesteps}/{self.total_timesteps} ({progress:.1f}%){' '*(24-len(str(self.total_timesteps))*2-len(f'{progress:.1f}'))}║")
            print(f"║  ⏳ 剩余步数: {remaining_steps}{' '*(43-len(str(remaining_steps)))}║")
            print(f"{'╚'+'═'*58+'╝'}\n")
        
        if self.num_timesteps - self.last_print_step >= self.print_interval:
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
            else:
                mean_reward = 0
            
            if mean_reward > self.best_mean_reward and mean_reward > 0:
                self.best_mean_reward = mean_reward
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                model_path = os.path.join(self.save_path, f'best_model_{timestamp}')
                self.model.save(model_path)
                print(f"\n🏆 新最佳模型！奖励: {mean_reward:.2f}")
                print(f"💾 已保存: {model_path}.zip\n")
            
            self.last_print_step = self.num_timesteps
        
        # ========== 记录实体无人机飞行数据和权重 ==========
        if self.data_logger and self.server and self.mirror_drones:
            try:
                for drone_name in self.mirror_drones:
                    # 记录飞行数据
                    logging_data = self.server.crazyswarm.get_loggingData_by_droneName(drone_name)
                    if logging_data:
                        self.data_logger.record_flight_data(drone_name, logging_data)
                    
                    # 记录权重变化
                    if drone_name in self.server.algorithms:
                        weights = self.server.algorithms[drone_name].get_current_coefficients()
                        self.data_logger.record_weights(
                            drone_name=drone_name,
                            weights=weights,
                            episode=self.episode_count,
                            step=self.num_timesteps
                        )
            except Exception as e:
                # 静默忽略数据记录错误，避免影响训练
                pass
        # ===========================================
        
        if self.num_timesteps % self.check_freq == 0 and self.num_timesteps > 0:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            checkpoint_path = os.path.join(self.save_path, f'checkpoint_{self.num_timesteps}_{timestamp}')
            self.model.save(checkpoint_path)
            print(f"💾 检查点: checkpoint_{self.num_timesteps}_{timestamp}.zip")
        
        return True


# ==================== 训练参数默认配置 ====================
DEFAULT_DRONE_NAMES = ["UAV1", "UAV2", "UAV3"]
DEFAULT_TOTAL_TIMESTEPS = 100
DEFAULT_STEP_DURATION = 5.0
DEFAULT_CHECKPOINT_FREQ = 1000
DEFAULT_ENABLE_VISUALIZATION = True
DEFAULT_INITIAL_MODEL_PATH = None
DEFAULT_USE_INITIAL_WEIGHTS = True
DEFAULT_MIRROR_DRONES = []  # 默认不设置镜像无人机
# =====================================================

def main():
    """主训练流程函数"""
    # ========== 命令行参数解析 ==========
    parser = argparse.ArgumentParser(description="虚实融合训练 - DDPG权重APF训练")
    parser.add_argument("--config", type=str, default=None, help="训练配置文件路径（JSON）")
    parser.add_argument("--mirror-drones", nargs="+", default=None, help="实体镜像无人机列表（如: UAV1 UAV2）")
    parser.add_argument("--initial-model-path", type=str, default=None, help="初始模型路径")
    parser.add_argument("--use-initial-weights", action="store_true", default=None, help="启用初始权重继承")
    parser.add_argument("--no-initial-weights", action="store_true", default=None, help="禁用初始权重继承")
    parser.add_argument("--overwrite-model", action="store_true", default=None, help="覆盖现有模型（不生成新时间戳）")
    parser.add_argument("--model-name", type=str, default=None, help="指定模型名称（不含.zip）")
    args = parser.parse_args()
    
    # ========== 加载配置并解析参数 ==========
    config = _load_train_config(args.config)
    
    drone_names = _get_config_value(None, config, "drone_names", DEFAULT_DRONE_NAMES)
    total_timesteps = int(_get_config_value(None, config, "total_timesteps", DEFAULT_TOTAL_TIMESTEPS))
    step_duration = float(_get_config_value(None, config, "step_duration", DEFAULT_STEP_DURATION))
    checkpoint_freq = int(_get_config_value(None, config, "checkpoint_freq", DEFAULT_CHECKPOINT_FREQ))
    enable_visualization = bool(_get_config_value(None, config, "enable_visualization", DEFAULT_ENABLE_VISUALIZATION))
    safety_limit = bool(_get_config_value(None, config, "safety_limit", True))
    max_weight_delta = float(_get_config_value(None, config, "max_weight_delta", 0.5))
    
    # 镜像无人机配置
    mirror_drones = args.mirror_drones if args.mirror_drones is not None else _get_config_value(None, config, "mirror_drones", DEFAULT_MIRROR_DRONES)
    if isinstance(mirror_drones, str):
        mirror_drones = [mirror_drones]
    
    # 初始权重使用逻辑
    if args.use_initial_weights is None and args.no_initial_weights is None:
        use_initial_weights = bool(_get_config_value(None, config, "use_initial_weights", DEFAULT_USE_INITIAL_WEIGHTS))
    else:
        use_initial_weights = bool(args.use_initial_weights) and not bool(args.no_initial_weights)
    
    initial_model_path = _get_config_value(
        args.initial_model_path,
        config,
        "initial_model_path",
        DEFAULT_INITIAL_MODEL_PATH
    )
    
    # 模型覆盖逻辑
    overwrite_model = bool(_get_config_value(
        args.overwrite_model if args.overwrite_model is not None else None,
        config,
        "overwrite_model",
        False
    ))
    
    # 模型名称
    model_name = _get_config_value(
        args.model_name,
        config,
        "model_name",
        "weight_predictor_hybrid"
    )
    # ==========================================
    
    # ========== 初始化全局变量 ==========
    server = None
    training_visualizer = None
    data_logger = None  # 实体无人机数据记录器
    temp_config_file = None
    original_config_file = None
    # ====================================================
    
    print("\n" + "=" * 60)
    print("🚀 虚实融合训练 - DDPG权重APF训练")
    print("=" * 60)
    print(f"🚁 无人机数量: {len(drone_names)} 台 ({', '.join(drone_names)})")
    if mirror_drones:
        print(f"🔗 实体镜像无人机: {', '.join(mirror_drones)}")
        print(f"   (这些无人机将使用实体Crazyflie的状态数据)")
    else:
        print(f"🔗 实体镜像无人机: 无 (纯虚拟训练)")
    print(f"📊 训练步数: {total_timesteps} 步")
    print(f"⏱️  每步时长: {step_duration} 秒")
    print(f"💾 检查点: 每 {checkpoint_freq} 步保存一次")
    print(f"👁️  可视化: {'启用' if enable_visualization else '禁用'}")
    print("=" * 60)
    
    print("\n💡 虚实融合训练说明:")
    print(f"   - 虚拟无人机: 使用AirSim仿真环境数据")
    if mirror_drones:
        print(f"   - 实体镜像无人机: 使用实体Crazyflie实时数据")
        print(f"   - 训练将融合虚拟和实体的状态信息")
    else:
        print(f"   - 当前为纯虚拟训练模式")
    print("\n[重要] 请确保Unity AirSim仿真已经运行！")
    if mirror_drones:
        print("[重要] 请确保实体Crazyflie已连接并处于安全可控状态！")
    
    confirm = input("Unity已运行？(Y/N): ").strip().upper()
    if confirm != 'Y':
        print("请先启动Unity")
        return
    
    try:
        # ========== [0/5] 设置虚实融合配置 ==========
        print("\n[0/5] 设置虚实融合配置...")
        # 使用统一的算法配置文件
        original_config_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "apf_algorithm_config.json")
        temp_config_file = _setup_hybrid_config(original_config_file, mirror_drones)
        
        # ========== [1/5] 启动AlgorithmServer ==========
        print("\n[1/5] 启动AlgorithmServer...")
        
        server = MultiDroneAlgorithmServer(
            config_file=temp_config_file,  # 使用临时配置文件
            drone_names=drone_names,
            use_learned_weights=False,
            model_path=None,
            enable_visualization=False
        )
        
        print(f"✅ 服务器创建成功")
        print(f"  无人机配置: {', '.join(drone_names)}")
        print(f"  配置文件: {temp_config_file}")
    
        if not server.start():
            print("[错误] AlgorithmServer启动失败")
            return
        
        print("[OK] AlgorithmServer已连接")
        
        # 启动无人机和算法线程
        print("\n[2/5] 启动无人机任务...")
        if not server.start_mission():
            print("[错误] 任务启动失败")
            server.stop()
            return
        
        print("[OK] 无人机已起飞，算法线程运行中")

        # 启动独立进程可视化（默认启用，可通过环境变量 NO_VIS=1 禁用）
        ipc_server = None
        vis_process = None
        vis_log_f = None
        vis_log_path = None
        _tmp_vis_log_dir = os.path.join(os.path.dirname(__file__), 'logs', 'ddpg_hybrid')
        os.makedirs(_tmp_vis_log_dir, exist_ok=True)

        if HAS_EXT_VIS and os.environ.get('NO_VIS', '0') != '1':
            try:
                ipc_server = VisualizationIPCServer(
                    snapshot_provider=server.get_visualization_snapshot,
                    host='127.0.0.1',
                    port=0,
                    hz=10.0,
                    compress_level=1
                )
                ipc_server.start()
                port = ipc_server.bound_port

                vis_log_path = os.path.join(_tmp_vis_log_dir, 'external_vis.log')
                vis_log_f = open(vis_log_path, 'w', encoding='utf-8')

                python_exe = sys.executable
                vis_entry = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'multirotor', 'Visualization', 'external_visualizer_client.py')
                vis_cmd = [python_exe, vis_entry, '--mode', 'ddpg', '--host', '127.0.0.1', '--port', str(port)]

                vis_env = os.environ.copy()
                vis_env['PYTHONIOENCODING'] = 'utf-8'
                vis_env['PYTHONUTF8'] = '1'

                creationflags = 0
                if os.name == 'nt' and os.environ.get('VIS_NEW_CONSOLE', '0') == '1':
                    creationflags = subprocess.CREATE_NEW_CONSOLE

                vis_process = subprocess.Popen(
                    vis_cmd,
                    stdout=vis_log_f,
                    stderr=vis_log_f,
                    creationflags=creationflags,
                    env=vis_env
                )

                time.sleep(0.5)
                rc = vis_process.poll()
                if rc is not None:
                    print(f"! 独立可视化进程启动后立即退出 (returncode={rc})")
                    print(f"  - 请查看: {vis_log_path}")
                else:
                    print(f"✅ 已启动独立可视化进程 (port={port})")
                    print(f"  - 外部可视化日志: {vis_log_path}")
            except Exception as e:
                print(f"! 启动独立可视化失败: {e}")
                if vis_log_path:
                    print(f"  - 外部可视化日志(若已生成): {vis_log_path}")
                try:
                    if ipc_server:
                        ipc_server.stop()
                except Exception:
                    pass
                ipc_server = None
                vis_process = None

        
        # ========== 创建实体无人机数据记录器 ==========
        # 如果有镜像无人机，则启动数据记录
        if mirror_drones:
            print("\n[2.5/5] 创建实体无人机数据记录器...")
            data_logger = CrazyflieDataLogger(
                drone_names=mirror_drones,
                output_dir=os.path.join(os.path.dirname(__file__), "crazyflie_logs")
            )
            data_logger.start_recording()
            print("✅ 数据记录器已启动")
        # =============================================
        
        # 等待系统稳定
        print("\n[3/5] 等待系统稳定...")
        time.sleep(5)

        # 加载初始权重
        initial_weights = {}
        if use_initial_weights:
            if not initial_model_path:
                print("⚠️  未指定初始模型路径，跳过初始权重加载")
            else:
                initial_weights_path = _derive_weights_path(initial_model_path)
                if os.path.exists(initial_weights_path):
                    print(f"📂 找到权重文件: {initial_weights_path}")
                    initial_weights = _load_initial_weights(initial_weights_path)
                else:
                    print(f"⚠️  权重文件不存在: {initial_weights_path}")
                
            if initial_weights:
                for drone_name in drone_names:
                    weights = initial_weights.get(drone_name) or initial_weights.get("__all__")
                    if weights:
                        server.algorithms[drone_name].set_coefficients(weights)
                print(f"✅ 已加载初始权重")
            else:
                print("⚠️  未找到可用初始权重，使用默认配置权重")
        
        # ========== [4/5] 创建训练环境 ==========
        print("\n[4/5] 创建训练环境...")
        
        env = SimpleWeightEnv(
            server=server,
            drone_name=drone_names[0],
            reset_unity=True,
            step_duration=step_duration,
            safety_limit=safety_limit,
            max_weight_delta=max_weight_delta
        )
        if use_initial_weights and initial_weights:
            training_weights = initial_weights.get(drone_names[0]) or initial_weights.get("__all__")
            if training_weights:
                env.set_initial_action(_weights_to_action(training_weights))
        print(f"✅ 环境创建成功")
        print(f"  📋 模式: 虚实融合训练")
        print(f"  🎓 训练无人机: {drone_names[0]}")
        if mirror_drones:
            print(f"  🔗 实体镜像: {', '.join(mirror_drones)}")
        print(f"  ⏱️  每步时长: {step_duration}秒")
        
        # 训练可视化（已迁移到独立进程 external_visualizer_client.py，避免 pygame 阻塞训练）
        if enable_visualization:
            print("\n[4.5/5] 训练可视化: 已使用独立进程模式启动")
            if vis_log_path:
                print(f"  - 外部可视化日志: {vis_log_path}")
            else:
                print("  - 外部可视化日志: (未生成)")
            # ⚠️ 重要：仍然需要在主进程中创建训练可视化器用于收集统计数据（权重历史等）
            # 外部进程负责显示，主进程中的visualizer负责数据收集
            from multirotor.Visualization.ddpg_training_visualizer import DDPGTrainingVisualizer
            training_visualizer = DDPGTrainingVisualizer(server=server, env=None)
            print("  - 主进程训练数据收集器: 已创建 (用于权重历史统计)")

        # ========== [5/5] 创建DDPG模型 ==========
        print("\n[5/5] 创建DDPG模型...")
        
        n_actions = env.action_space.shape[0]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.15 * np.ones(n_actions)
        )
        
        model = DDPG(
            "MlpPolicy",
            env,
            action_noise=action_noise,
            learning_rate=1e-4,
            buffer_size=5000,
            learning_starts=200,
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
        print(f"📊 训练步数: {total_timesteps}")
        print(f"⏸️  按 Ctrl+C 可随时停止")
        print("=" * 60 + "\n")
        
        model_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        training_callback = ImprovedTrainingCallback(
            total_timesteps=total_timesteps,
            check_freq=checkpoint_freq,
            save_path=model_dir,
            training_visualizer=training_visualizer,
            data_logger=data_logger,
            server=server,
            mirror_drones=mirror_drones,
            verbose=1
        )
        
        model.learn(
            total_timesteps=total_timesteps,
            log_interval=None,
            callback=training_callback
        )
        
        print("\n" + "=" * 60)
        print("✅ 训练完成！")
        print("=" * 60)
        
        # 保存最终模型
        print("\n💾 保存最终模型...")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        final_model_path = os.path.join(model_dir, f"weight_predictor_hybrid_{timestamp}")
        model.save(final_model_path)
        print(f"✅ 模型已保存: {final_model_path}.zip")

        # 保存最后权重系数
        weights_path = _derive_weights_path(final_model_path)
        _save_final_weights(server, weights_path)
        
        # 保存实体无人机数据
        if data_logger:
            print("\n停止并保存实体无人机数据...")
            data_logger.stop_recording()
            data_logger.save_all()
        
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
        
    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("[中断] 正在停止训练...")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n\n[错误] 训练出错: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 保存实体无人机数据（最优先）
        if data_logger:
            try:
                print("\n保存实体无人机训练数据...")
                if data_logger.is_recording:
                    data_logger.stop_recording()
                data_logger.save_all()
            except Exception as e:
                print(f"[警告] 保存数据时出错: {e}")
        
        # 清理资源
        if training_visualizer:
            print("\n停止训练可视化...")
            try:
                training_visualizer.stop_visualization()
            except Exception as e:
                print(f"[警告] 停止可视化时出错: {e}")
        
        if server:
            print("\n停止AlgorithmServer...")
            try:
                for drone_name in drone_names:
                    try:
                        print(f"  降落 {drone_name}...")
                        server.drone_controller.land(drone_name)
                    except:
                        pass
                server.unity_socket.stop()
                print("[OK] AlgorithmServer已停止")
            except Exception as e:
                print(f"[警告] 清理资源时出现错误: {e}")
        
        # 恢复配置文件
        if temp_config_file and original_config_file:
            _restore_config(original_config_file, temp_config_file)
        
        print("\n训练已结束")
        print("按Enter键退出...")
        try:
            input()
        except:
            pass


if __name__ == "__main__":
    main()
