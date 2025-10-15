"""
无人机移动环境 - DQN训练
使用离散动作空间（6方向位移）直接控制无人机移动
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os
import json


class MovementEnv(gym.Env):
    """
    无人机移动学习环境
    
    动作空间: 6个离散动作（上/下/左/右/前/后）
    观察空间: 位置、速度、熵值、leader位置等
    """
    
    def __init__(self, server=None, drone_name="UAV1", config_path=None):
        super(MovementEnv, self).__init__()
        
        self.server = server
        self.drone_name = drone_name
        
        # 加载配置
        self.config = self._load_config(config_path)
        print(f"[OK] 移动DQN环境已加载配置")
        
        # 动作空间: 6个离散动作
        # 0: 向上, 1: 向下, 2: 向左, 3: 向右, 4: 向前, 5: 向后
        self.action_space = spaces.Discrete(6)
        
        # 观察空间维度说明：
        # - 位置(3): x, y, z
        # - 速度(3): vx, vy, vz
        # - 朝向(3): forward_x, forward_y, forward_z
        # - 局部熵值统计(3): 平均熵, 最大熵, 熵标准差
        # - Leader相对位置(3): dx, dy, dz
        # - Leader范围信息(2): 距离, 是否越界
        # - 扫描进度(3): 已扫描比例, 已扫描数量, 未扫描数量
        # - 其他无人机最近距离(1)
        # 总计: 21维
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(21,),
            dtype=np.float32
        )
        
        # 动作到位移的映射（单位：米）
        self.action_step = self.config['movement']['step_size']
        self.action_map = {
            0: np.array([0, 0, self.action_step]),      # 上
            1: np.array([0, 0, -self.action_step]),     # 下
            2: np.array([-self.action_step, 0, 0]),     # 左
            3: np.array([self.action_step, 0, 0]),      # 右
            4: np.array([0, self.action_step, 0]),      # 前
            5: np.array([0, -self.action_step, 0])      # 后
        }
        
        # 状态记录
        self.prev_scanned_cells = 0
        self.prev_position = None
        self.prev_entropy_sum = 0
        self.step_count = 0
        self.episode_reward = 0
        
        # 统计信息
        self.collision_count = 0
        self.out_of_range_count = 0
        
    def _load_config(self, config_path):
        """加载配置文件"""
        if config_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, "movement_dqn_config.json")
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # 返回默认配置
            return self._default_config()
    
    def _default_config(self):
        """默认配置"""
        return {
            "movement": {
                "step_size": 1.0,
                "max_steps": 500
            },
            "rewards": {
                "exploration": 10.0,
                "collision": -50.0,
                "out_of_range": -30.0,
                "smooth_movement": 1.0,
                "entropy_reduction": 5.0,
                "step_penalty": -0.1,
                "success": 100.0
            },
            "thresholds": {
                "collision_distance": 2.0,
                "scanned_entropy": 30.0,
                "nearby_entropy_distance": 10.0,
                "success_scan_ratio": 0.95
            }
        }
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)
        
        # 如果连接了server，发送重置命令到Unity
        if self.server:
            self.server.reset_environment()
            import time
            time.sleep(1.0)  # 等待Unity完成重置
        
        self.prev_scanned_cells = self._count_scanned_cells()
        self.prev_entropy_sum = self._get_total_entropy()
        self.prev_position = None
        self.step_count = 0
        self.episode_reward = 0
        self.collision_count = 0
        self.out_of_range_count = 0
        
        return self._get_state(), {}
    
    def step(self, action):
        """
        执行一步动作
        
        :param action: 0-5的整数，表示6个移动方向
        :return: observation, reward, terminated, truncated, info
        """
        # 确保action是整数（从numpy数组转换）
        if hasattr(action, 'item'):
            action = action.item()
        action = int(action)
        
        # 记录当前位置
        current_state = self._get_state()
        
        # 将动作转换为位移向量
        displacement = self.action_map[action]
        
        # 发送移动指令到server（如果连接）
        if self.server:
            self._apply_movement(displacement)
        
        # 等待一小段时间让环境更新
        if self.server:
            import time
            time.sleep(0.05)  # 50ms
        
        # 获取新状态
        next_state = self._get_state()
        
        # 计算奖励
        reward = self._calculate_reward(action, current_state, next_state)
        self.episode_reward += reward
        
        # 判断是否结束
        self.step_count += 1
        terminated = self._check_done()  # episode自然结束
        truncated = False  # 不使用截断
        
        # 额外信息
        info = {
            'action': action,
            'displacement': displacement.tolist(),
            'scanned_cells': self._count_scanned_cells(),
            'collision_count': self.collision_count,
            'out_of_range_count': self.out_of_range_count,
            'episode_reward': self.episode_reward
        }
        
        return next_state, reward, terminated, truncated, info
    
    def _get_state(self):
        """获取当前观察状态（21维）"""
        if not self.server:
            # 测试模式：返回随机状态
            return np.random.randn(21).astype(np.float32)
        
        try:
            with self.server.data_lock:
                runtime_data = self.server.unity_runtime_data[self.drone_name]
                grid_data = self.server.grid_data
                
                # 1. 位置 (3)
                pos = runtime_data.position
                position = np.array([pos.x, pos.y, pos.z], dtype=np.float32)
                
                # 2. 速度 (3)
                vel = runtime_data.finalMoveDir
                velocity = np.array([
                    vel.x * self.server.config_data.moveSpeed,
                    vel.y * self.server.config_data.moveSpeed,
                    vel.z * self.server.config_data.moveSpeed
                ], dtype=np.float32)
                
                # 3. 朝向 (3)
                fwd = runtime_data.forward
                direction = np.array([fwd.x, fwd.y, fwd.z], dtype=np.float32)
                
                # 4. 局部熵值统计 (3)
                entropy_info = self._get_entropy_info(grid_data, pos)
                
                # 5. Leader相对位置 (3)
                if runtime_data.leader_position:
                    leader_rel = np.array([
                        runtime_data.leader_position.x - pos.x,
                        runtime_data.leader_position.y - pos.y,
                        runtime_data.leader_position.z - pos.z
                    ], dtype=np.float32)
                else:
                    leader_rel = np.zeros(3, dtype=np.float32)
                
                # 6. Leader范围信息 (2)
                if runtime_data.leader_position and runtime_data.leader_scan_radius > 0:
                    dist_to_leader = np.linalg.norm(leader_rel)
                    is_out_of_range = 1.0 if dist_to_leader > runtime_data.leader_scan_radius else 0.0
                    leader_range = np.array([dist_to_leader, is_out_of_range], dtype=np.float32)
                else:
                    leader_range = np.zeros(2, dtype=np.float32)
                
                # 7. 扫描进度 (3)
                scan_info = self._get_scan_info(grid_data)
                
                # 8. 最近无人机距离 (1)
                min_dist = self._get_min_distance_to_others(runtime_data)
                min_dist_array = np.array([min_dist], dtype=np.float32)
                
                # 组合状态向量
                state = np.concatenate([
                    position,           # 3
                    velocity,           # 3
                    direction,          # 3
                    entropy_info,       # 3
                    leader_rel,         # 3
                    leader_range,       # 2
                    scan_info,          # 3
                    min_dist_array      # 1
                ])
                
                return state
                
        except Exception as e:
            print(f"获取状态失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return np.zeros(21, dtype=np.float32)
    
    def _calculate_reward(self, action, prev_state, next_state):
        """计算奖励"""
        if not self.server:
            return 0.0
        
        reward = 0.0
        cfg_reward = self.config['rewards']
        cfg_thresh = self.config['thresholds']
        
        try:
            with self.server.data_lock:
                runtime_data = self.server.unity_runtime_data[self.drone_name]
                
                # 1. 探索奖励：新扫描的单元格
                current_scanned = self._count_scanned_cells()
                new_scanned = current_scanned - self.prev_scanned_cells
                if new_scanned > 0:
                    reward += new_scanned * cfg_reward['exploration']
                self.prev_scanned_cells = current_scanned
                
                # 2. 熵值降低奖励
                current_entropy = self._get_total_entropy()
                entropy_reduced = self.prev_entropy_sum - current_entropy
                if entropy_reduced > 0:
                    reward += entropy_reduced * cfg_reward['entropy_reduction']
                self.prev_entropy_sum = current_entropy
                
                # 3. 碰撞惩罚
                min_dist = self._get_min_distance_to_others(runtime_data)
                if min_dist < cfg_thresh['collision_distance']:
                    reward += cfg_reward['collision']
                    self.collision_count += 1
                
                # 4. 越界惩罚
                if runtime_data.leader_position and runtime_data.leader_scan_radius > 0:
                    pos = runtime_data.position
                    dist_to_leader = np.sqrt(
                        (pos.x - runtime_data.leader_position.x) ** 2 +
                        (pos.y - runtime_data.leader_position.y) ** 2 +
                        (pos.z - runtime_data.leader_position.z) ** 2
                    )
                    if dist_to_leader > runtime_data.leader_scan_radius:
                        reward += cfg_reward['out_of_range']
                        self.out_of_range_count += 1
                
                # 5. 平滑运动奖励
                if self.prev_position:
                    current_pos = runtime_data.position
                    movement = np.sqrt(
                        (current_pos.x - self.prev_position.x) ** 2 +
                        (current_pos.y - self.prev_position.y) ** 2 +
                        (current_pos.z - self.prev_position.z) ** 2
                    )
                    # 鼓励适度移动
                    if 0.5 < movement < 5.0:
                        reward += cfg_reward['smooth_movement']
                
                self.prev_position = runtime_data.position
                
                # 6. 每步小惩罚（鼓励快速完成）
                reward += cfg_reward['step_penalty']
                
                # 7. 成功奖励
                scan_ratio = self._get_scan_ratio()
                if scan_ratio >= cfg_thresh['success_scan_ratio']:
                    reward += cfg_reward['success']
                
        except Exception as e:
            print(f"计算奖励失败: {str(e)}")
        
        return reward
    
    def _check_done(self):
        """判断episode是否结束"""
        # 达到最大步数
        if self.step_count >= self.config['movement']['max_steps']:
            return True
        
        # 扫描完成
        scan_ratio = self._get_scan_ratio()
        if scan_ratio >= self.config['thresholds']['success_scan_ratio']:
            return True
        
        # 碰撞次数过多
        if self.collision_count >= 10:
            return True
        
        return False
    
    def _apply_movement(self, displacement):
        """应用移动到无人机"""
        try:
            with self.server.data_lock:
                runtime_data = self.server.unity_runtime_data[self.drone_name]
                
                # 计算目标位置
                current_pos = runtime_data.position
                target_pos = {
                    'x': current_pos.x + displacement[0],
                    'y': current_pos.y + displacement[1],
                    'z': current_pos.z + displacement[2]
                }
                
                # 通过算法设置目标方向
                # 注意：这里简化处理，实际应该通过Unity客户端直接控制
                direction = displacement / (np.linalg.norm(displacement) + 1e-6)
                
                # 更新runtime_data的移动方向
                from Algorithm.Vector3 import Vector3
                runtime_data.finalMoveDir = Vector3(direction[0], direction[1], direction[2])
                
        except Exception as e:
            print(f"应用移动失败: {str(e)}")
    
    def _get_entropy_info(self, grid_data, position):
        """获取局部熵值统计"""
        if not grid_data or not grid_data.cells:
            return np.array([50.0, 50.0, 0.0], dtype=np.float32)
        
        nearby_distance = self.config['thresholds']['nearby_entropy_distance']
        
        # 找附近单元格
        nearby_cells = [
            cell for cell in grid_data.cells[:100]
            if np.sqrt(
                (cell.center.x - position.x) ** 2 +
                (cell.center.y - position.y) ** 2 +
                (cell.center.z - position.z) ** 2
            ) < nearby_distance
        ]
        
        if not nearby_cells:
            return np.array([50.0, 50.0, 0.0], dtype=np.float32)
        
        entropies = [cell.entropy for cell in nearby_cells]
        return np.array([
            np.mean(entropies),
            np.max(entropies),
            np.std(entropies)
        ], dtype=np.float32)
    
    def _get_scan_info(self, grid_data):
        """获取扫描进度信息"""
        if not grid_data or not grid_data.cells:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        total = len(grid_data.cells)
        scanned = sum(
            1 for cell in grid_data.cells
            if cell.entropy < self.config['thresholds']['scanned_entropy']
        )
        
        return np.array([
            scanned / max(total, 1),
            float(scanned),
            float(total - scanned)
        ], dtype=np.float32)
    
    def _count_scanned_cells(self):
        """统计已扫描单元格数量"""
        if not self.server or not self.server.grid_data:
            return 0
        
        try:
            with self.server.data_lock:
                return sum(
                    1 for cell in self.server.grid_data.cells
                    if cell.entropy < self.config['thresholds']['scanned_entropy']
                )
        except:
            return 0
    
    def _get_total_entropy(self):
        """获取总熵值"""
        if not self.server or not self.server.grid_data:
            return 0.0
        
        try:
            with self.server.data_lock:
                return sum(cell.entropy for cell in self.server.grid_data.cells)
        except:
            return 0.0
    
    def _get_scan_ratio(self):
        """获取扫描完成比例"""
        if not self.server or not self.server.grid_data:
            return 0.0
        
        try:
            with self.server.data_lock:
                total = len(self.server.grid_data.cells)
                if total == 0:
                    return 0.0
                scanned = sum(
                    1 for cell in self.server.grid_data.cells
                    if cell.entropy < self.config['thresholds']['scanned_entropy']
                )
                return scanned / total
        except:
            return 0.0
    
    def _get_min_distance_to_others(self, runtime_data):
        """获取到其他无人机的最小距离"""
        if not runtime_data.otherScannerPositions:
            return 999.0
        
        pos = runtime_data.position
        distances = [
            np.sqrt(
                (pos.x - other_pos.x) ** 2 +
                (pos.y - other_pos.y) ** 2 +
                (pos.z - other_pos.z) ** 2
            )
            for other_pos in runtime_data.otherScannerPositions
        ]
        return min(distances) if distances else 999.0
    
    def render(self, mode='human'):
        """可视化（可选）"""
        pass
    
    def close(self):
        """关闭环境"""
        pass


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("测试 MovementEnv - 无人机移动DQN环境")
    print("=" * 60)
    
    # 创建环境（无server，测试模式）
    env = MovementEnv(server=None, drone_name="UAV1")
    
    print(f"\n观察空间: {env.observation_space}")
    print(f"动作空间: {env.action_space}")
    print(f"动作映射:")
    for action, displacement in env.action_map.items():
        action_name = ['上', '下', '左', '右', '前', '后'][action]
        print(f"  {action}: {action_name} -> {displacement}")
    
    # 重置环境
    state, info = env.reset()
    print(f"\n初始状态shape: {state.shape}")
    print(f"初始状态前5维: {state[:5]}")
    
    # 执行几步测试
    print("\n执行动作测试:")
    for i in range(6):
        action = i  # 测试所有6个动作
        action_name = ['上', '下', '左', '右', '前', '后'][action]
        
        print(f"\n步骤 {i+1}: 动作={action} ({action_name})")
        state, reward, done, info = env.step(action)
        print(f"  奖励: {reward:.2f}")
        print(f"  完成: {done}")
        print(f"  信息: {info}")
        
        if done:
            break
    
    print("\n" + "=" * 60)
    print("[OK] 环境测试通过！")
    print("=" * 60)

