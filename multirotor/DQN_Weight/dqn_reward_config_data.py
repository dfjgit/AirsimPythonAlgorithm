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
        """加载配置文件（简化版）"""
        if not os.path.exists(self.config_path):
            print(f"[警告] 配置文件不存在: {self.config_path}")
            self._set_defaults()
            return
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 奖励参数
            rewards = config.get('reward_coefficients', {})
            self.exploration_reward = rewards.get('exploration_reward', 5.0)
            self.out_of_range_penalty = rewards.get('out_of_range_penalty', 10.0)
            
            # 阈值参数
            thresholds = config.get('thresholds', {})
            self.scanned_entropy_threshold = thresholds.get('scanned_entropy_threshold', 30)
            
            # Episode参数
            episode = config.get('episode', {})
            self.max_steps = episode.get('max_steps', 200)
            
            # 动作空间参数
            action_space = config.get('action_space', {})
            self.weight_min = action_space.get('weight_min', 0.5)
            self.weight_max = action_space.get('weight_max', 5.0)
            
            print(f"[OK] 配置加载成功")
            
        except Exception as e:
            print(f"[错误] 加载配置失败: {str(e)}")
            self._set_defaults()
    
    def _set_defaults(self):
        """设置默认配置（简化版）"""
        # 奖励参数
        self.exploration_reward = 5.0
        self.out_of_range_penalty = 10.0
        
        # 阈值参数
        self.scanned_entropy_threshold = 30
        
        # Episode参数
        self.max_steps = 200
        
        # 动作空间参数
        self.weight_min = 0.5
        self.weight_max = 5.0
    
    def to_dict(self):
        """转换为字典（简化版）"""
        return {
            'reward_coefficients': {
                'exploration_reward': self.exploration_reward,
                'out_of_range_penalty': self.out_of_range_penalty
            },
            'thresholds': {
                'scanned_entropy_threshold': self.scanned_entropy_threshold
            },
            'episode': {
                'max_steps': self.max_steps
            },
            'action_space': {
                'weight_min': self.weight_min,
                'weight_max': self.weight_max
            }
        }
    
    def __str__(self):
        """字符串表示"""
        return f"DQNRewardConfig(max_steps={self.max_steps}, exploration={self.exploration_reward})"
    
    def __repr__(self):
        return self.__str__()


# 测试代码
if __name__ == "__main__":
    config = DQNRewardConfig()
    print(f"配置: {config}")
    print(f"探索奖励: {config.exploration_reward}")
    print(f"越界惩罚: {config.out_of_range_penalty}")
    print(f"最大步数: {config.max_steps}")
    print("[OK] 配置加载成功")
