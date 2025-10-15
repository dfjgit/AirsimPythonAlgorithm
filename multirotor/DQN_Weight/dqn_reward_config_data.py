"""
DQN奖励配置数据类
用于加载和管理movement_dqn的配置
"""
import json
import os


class DQNRewardConfig:
    """DQN奖励配置类"""
    
    def __init__(self, config_path=None):
        """
        初始化配置
        :param config_path: 配置文件路径
        """
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), 
                "movement_dqn_config.json"
            )
        
        self.config_path = config_path
        self._load_config()
    
    def _load_config(self):
        """加载配置文件"""
        if not os.path.exists(self.config_path):
            print(f"[警告] 配置文件不存在: {self.config_path}")
            print(f"[警告] 使用默认配置")
            self._set_defaults()
            return
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 移动参数
            self.step_size = config['movement'].get('step_size', 1.0)
            self.max_steps = config['movement'].get('max_steps', 500)
            
            # 奖励参数
            self.exploration_reward = config['rewards'].get('exploration', 10.0)
            self.entropy_reduction_reward = config['rewards'].get('entropy_reduction', 5.0)
            self.collision_penalty = config['rewards'].get('collision', -50.0)
            self.out_of_range_penalty = config['rewards'].get('out_of_range', -30.0)
            self.smooth_movement_reward = config['rewards'].get('smooth_movement', 1.0)
            self.step_penalty = config['rewards'].get('step_penalty', -0.1)
            self.success_reward = config['rewards'].get('success', 100.0)
            
            # 阈值参数
            self.collision_distance = config['thresholds'].get('collision_distance', 2.0)
            self.scanned_entropy_threshold = config['thresholds'].get('scanned_entropy', 30.0)
            self.nearby_entropy_distance = config['thresholds'].get('nearby_entropy_distance', 10.0)
            self.success_scan_ratio = config['thresholds'].get('success_scan_ratio', 0.95)
            
            # 训练参数（如果需要）
            if 'training' in config:
                self.total_timesteps = config['training'].get('total_timesteps', 100000)
                self.learning_rate = config['training'].get('learning_rate', 0.0001)
                self.buffer_size = config['training'].get('buffer_size', 50000)
                self.batch_size = config['training'].get('batch_size', 32)
            
            # 旧配置兼容性（用于SimpleWeightEnv）
            self.weight_min = 0.1
            self.weight_max = 5.0
            self.std_threshold = 2.0
            self.std_smoothing = 0.7
            self.max_min_ratio = 10.0
            self.movement_min = 0.5
            self.movement_max = 5.0
            
            print(f"[OK] DQN配置加载成功: {self.config_path}")
            
        except Exception as e:
            print(f"[错误] 加载配置文件失败: {str(e)}")
            print(f"[警告] 使用默认配置")
            self._set_defaults()
    
    def _set_defaults(self):
        """设置默认配置"""
        # 移动参数
        self.step_size = 1.0
        self.max_steps = 500
        
        # 奖励参数
        self.exploration_reward = 10.0
        self.entropy_reduction_reward = 5.0
        self.collision_penalty = -50.0
        self.out_of_range_penalty = -30.0
        self.smooth_movement_reward = 1.0
        self.step_penalty = -0.1
        self.success_reward = 100.0
        
        # 阈值参数
        self.collision_distance = 2.0
        self.scanned_entropy_threshold = 30.0
        self.nearby_entropy_distance = 10.0
        self.success_scan_ratio = 0.95
        
        # 训练参数
        self.total_timesteps = 100000
        self.learning_rate = 0.0001
        self.buffer_size = 50000
        self.batch_size = 32
        
        # 旧配置兼容性
        self.weight_min = 0.1
        self.weight_max = 5.0
        self.std_threshold = 2.0
        self.std_smoothing = 0.7
        self.max_min_ratio = 10.0
        self.movement_min = 0.5
        self.movement_max = 5.0
    
    def to_dict(self):
        """转换为字典"""
        return {
            'movement': {
                'step_size': self.step_size,
                'max_steps': self.max_steps
            },
            'rewards': {
                'exploration': self.exploration_reward,
                'entropy_reduction': self.entropy_reduction_reward,
                'collision': self.collision_penalty,
                'out_of_range': self.out_of_range_penalty,
                'smooth_movement': self.smooth_movement_reward,
                'step_penalty': self.step_penalty,
                'success': self.success_reward
            },
            'thresholds': {
                'collision_distance': self.collision_distance,
                'scanned_entropy': self.scanned_entropy_threshold,
                'nearby_entropy_distance': self.nearby_entropy_distance,
                'success_scan_ratio': self.success_scan_ratio
            }
        }
    
    def __str__(self):
        """字符串表示"""
        return f"DQNRewardConfig(step_size={self.step_size}, max_steps={self.max_steps})"
    
    def __repr__(self):
        return self.__str__()


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("测试 DQNRewardConfig")
    print("=" * 60)
    
    # 加载配置
    config = DQNRewardConfig()
    
    print(f"\n配置对象: {config}")
    print(f"\n配置详情:")
    print(f"  移动步长: {config.step_size}米")
    print(f"  最大步数: {config.max_steps}")
    print(f"  探索奖励: {config.exploration_reward}")
    print(f"  碰撞惩罚: {config.collision_penalty}")
    print(f"  碰撞距离: {config.collision_distance}米")
    
    print(f"\n配置字典:")
    import pprint
    pprint.pprint(config.to_dict())
    
    print(f"\n[OK] 测试完成")
