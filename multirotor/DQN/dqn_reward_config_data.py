"""
DQN奖励配置数据类
管理DQN训练过程中的奖励系数和阈值参数
"""
import json
import os


class DQNRewardConfig:
    """DQN奖励配置数据"""
    
    def __init__(self, config_path=None):
        """
        初始化配置
        
        :param config_path: 配置文件路径，如果为None则使用默认路径
        """
        # 奖励系数
        self.exploration_reward = 1.0
        self.collision_penalty = 5.0
        self.out_of_range_penalty = 2.0
        self.smooth_movement_reward = 0.1
        
        # 阈值参数
        self.collision_distance = 2.0
        self.movement_min = 0.5
        self.movement_max = 3.0
        self.scanned_entropy_threshold = 30
        self.nearby_entropy_distance = 10.0
        
        # Episode参数
        self.max_steps = 200
        
        # 动作空间参数
        self.weight_min = 0.5
        self.weight_max = 5.0
        self.std_threshold = 1.5
        self.std_smoothing = 0.7
        self.max_min_ratio = 5
        
        # 如果提供了配置路径，加载配置
        if config_path:
            self.load_from_file(config_path)
    
    def load_from_file(self, config_path):
        """
        从JSON文件加载配置
        
        :param config_path: 配置文件路径
        """
        if not os.path.exists(config_path):
            print(f"警告: 配置文件不存在: {config_path}，使用默认值")
            return
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 加载奖励系数
            if 'reward_coefficients' in config:
                rewards = config['reward_coefficients']
                self.exploration_reward = rewards.get('exploration_reward', self.exploration_reward)
                self.collision_penalty = rewards.get('collision_penalty', self.collision_penalty)
                self.out_of_range_penalty = rewards.get('out_of_range_penalty', self.out_of_range_penalty)
                self.smooth_movement_reward = rewards.get('smooth_movement_reward', self.smooth_movement_reward)
            
            # 加载阈值
            if 'thresholds' in config:
                thresholds = config['thresholds']
                self.collision_distance = thresholds.get('collision_distance', self.collision_distance)
                self.movement_min = thresholds.get('movement_min', self.movement_min)
                self.movement_max = thresholds.get('movement_max', self.movement_max)
                self.scanned_entropy_threshold = thresholds.get('scanned_entropy_threshold', self.scanned_entropy_threshold)
                self.nearby_entropy_distance = thresholds.get('nearby_entropy_distance', self.nearby_entropy_distance)
            
            # 加载Episode参数
            if 'episode' in config:
                episode = config['episode']
                self.max_steps = episode.get('max_steps', self.max_steps)
            
            # 加载动作空间参数
            if 'action_space' in config:
                action_space = config['action_space']
                self.weight_min = action_space.get('weight_min', self.weight_min)
                self.weight_max = action_space.get('weight_max', self.weight_max)
                self.std_threshold = action_space.get('std_threshold', self.std_threshold)
                self.std_smoothing = action_space.get('std_smoothing', self.std_smoothing)
                self.max_min_ratio = action_space.get('max_min_ratio', self.max_min_ratio)
            
            print(f"[OK] 成功加载DQN奖励配置: {config_path}")
            
        except Exception as e:
            print(f"错误: 加载配置文件失败: {str(e)}")
            print("使用默认配置值")
    
    def save_to_file(self, config_path):
        """
        保存配置到JSON文件
        
        :param config_path: 配置文件路径
        """
        config = {
            "reward_coefficients": {
                "exploration_reward": self.exploration_reward,
                "collision_penalty": self.collision_penalty,
                "out_of_range_penalty": self.out_of_range_penalty,
                "smooth_movement_reward": self.smooth_movement_reward
            },
            "thresholds": {
                "collision_distance": self.collision_distance,
                "movement_min": self.movement_min,
                "movement_max": self.movement_max,
                "scanned_entropy_threshold": self.scanned_entropy_threshold,
                "nearby_entropy_distance": self.nearby_entropy_distance
            },
            "episode": {
                "max_steps": self.max_steps
            },
            "action_space": {
                "weight_min": self.weight_min,
                "weight_max": self.weight_max,
                "std_threshold": self.std_threshold,
                "std_smoothing": self.std_smoothing,
                "max_min_ratio": self.max_min_ratio
            },
            "description": {
                "exploration_reward": "每个新扫描的单元格的奖励分数",
                "collision_penalty": "无人机间距离过近时的惩罚分数",
                "out_of_range_penalty": "超出Leader扫描范围时的惩罚分数",
                "smooth_movement_reward": "合理移动距离的奖励分数",
                "collision_distance": "触发碰撞惩罚的最小距离(米)",
                "movement_min": "合理移动的最小距离(米)",
                "movement_max": "合理移动的最大距离(米)",
                "scanned_entropy_threshold": "判断单元格是否已扫描的熵值阈值",
                "nearby_entropy_distance": "查找附近熵值信息的距离范围(米)",
                "max_steps": "每个训练episode的最大步数",
                "weight_min": "权重系数的最小值",
                "weight_max": "权重系数的最大值",
                "std_threshold": "触发权重平滑的标准差阈值",
                "std_smoothing": "权重平滑系数",
                "max_min_ratio": "最大权重与最小权重的最大比例"
            }
        }
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            print(f"[OK] 成功保存DQN奖励配置: {config_path}")
        except Exception as e:
            print(f"错误: 保存配置文件失败: {str(e)}")
    
    def __str__(self):
        """打印配置信息"""
        return f"""
DQN奖励配置:
  奖励系数:
    - 探索奖励: {self.exploration_reward}
    - 碰撞惩罚: {self.collision_penalty}
    - 越界惩罚: {self.out_of_range_penalty}
    - 平滑运动奖励: {self.smooth_movement_reward}
  
  阈值参数:
    - 碰撞距离: {self.collision_distance}m
    - 移动范围: {self.movement_min}m - {self.movement_max}m
    - 扫描熵值阈值: {self.scanned_entropy_threshold}
    - 附近熵值距离: {self.nearby_entropy_distance}m
  
  Episode设置:
    - 最大步数: {self.max_steps}
  
  动作空间:
    - 权重范围: {self.weight_min} - {self.weight_max}
    - 标准差阈值: {self.std_threshold}
    - 平滑系数: {self.std_smoothing}
    - 最大最小比: {self.max_min_ratio}
"""


# 测试代码
if __name__ == "__main__":
    print("测试DQN奖励配置...")
    
    # 1. 创建默认配置
    config = DQNRewardConfig()
    print("默认配置:")
    print(config)
    
    # 2. 保存到文件
    config.save_to_file("dqn_reward_config_test.json")
    
    # 3. 从文件加载
    config2 = DQNRewardConfig("dqn_reward_config_test.json")
    print("\n从文件加载的配置:")
    print(config2)
    
    # 4. 修改参数并保存
    config2.exploration_reward = 2.0
    config2.collision_penalty = 10.0
    config2.save_to_file("dqn_reward_config_modified.json")
    
    print("\n[OK] 配置测试完成！")

