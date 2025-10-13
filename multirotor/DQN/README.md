# DQN学习与人工势场算法集成设计文档

## 1. 项目现状分析

### 1.1 当前系统架构

当前系统通过 `MultiDroneAlgorithmServer` 类作为核心，连接Unity客户端与AirSim模拟器，实现多无人机协同控制：
- **数据流程**：Unity → AlgorithmServer → AirSim → 无人机
- **控制算法**：采用人工势场算法（`ScannerAlgorithm`类）
- **核心控制循环**：通过`_process_drone`方法周期性计算无人机移动方向并发送控制指令
- **DQN集成**：已实现DQN学习功能，可通过`enable_learning=True`参数启用

### 1.2 人工势场算法原理

人工势场算法通过合并以下五个方向向量计算最终移动方向：
1. **熵最优方向**：引导无人机探索未知区域
2. **最短路径方向**：引导无人机向目标移动
3. **排斥力方向**：避免无人机间碰撞
4. **保持Leader范围方向**：保持无人机在Leader范围内
5. **方向保持方向**：保持无人机飞行稳定性

每个方向向量有对应的权重系数，这些系数可以通过DQN学习动态调整。

### 1.3 DQN集成现状

项目已成功集成DQN学习功能：
- **DQNAgent类**：实现完整的DQN算法，包括经验回放、目标网络等
- **DroneLearningEnv类**：提供无人机学习环境接口
- **动态权重调整**：DQN可以实时调整人工势场算法的权重系数
- **训练与部署模式**：支持训练模式和部署模式切换

## 2. DQN集成目标

将DQN学习与人工势场算法结合，实现以下目标：
1. 保留现有人工势场算法的所有功能
2. 使用DQN学习优化人工势场算法的权重系数
3. 支持根据环境反馈自动调整控制策略
4. 提供训练模式和部署模式切换功能

## 3. 集成设计方案

### 3.1 整体架构设计

![集成架构图](架构图应该在此位置)

主要组件包括：
1. **MultiDroneAlgorithmServer**：核心服务类，负责整合DQN和人工势场算法
2. **ScannerAlgorithm**：现有的人工势场算法，保留其核心功能
3. **DQNAgent**：DQN智能体，负责学习优化权重系数
4. **DroneLearningEnv**：无人机学习环境，提供状态观察、奖励计算和动作执行接口

### 3.2 DQN组件设计

#### 3.2.1 状态空间定义

DQN智能体的输入状态空间（18维连续向量）：
- **无人机位置**（3维）：x, y, z坐标
- **无人机速度**（3维）：基于方向向量和移动速度计算
- **无人机方向**（3维）：forward向量的x, y, z分量
- **当前权重系数**（5维）：
  - repulsionCoefficient（排斥力系数）
  - entropyCoefficient（熵系数）
  - distanceCoefficient（距离系数）
  - leaderRangeCoefficient（Leader范围系数）
  - directionRetentionCoefficient（方向保持系数）
- **Leader相对位置**（3维）：相对于Leader的位置偏移
- **扫描效率**（1维）：已扫描区域占总区域的比例

#### 3.2.2 动作空间定义

DQN智能体的输出动作空间（25维离散动作）：
- 每个权重系数有5种调整方式：-2, -1, 0, 1, 2（调整步长）
- 5个权重系数 × 5种调整方式 = 25个离散动作
- 权重调整步长：0.5
- 权重范围限制：0.1 到 10.0

#### 3.2.3 奖励函数设计

奖励函数包含以下几个部分（已实现）：
- **探索奖励**（权重1.0）：基于新增扫描区域面积计算
- **效率奖励**（权重0.5）：基于扫描效率的额外奖励
- **碰撞惩罚**（-5.0）：当无人机间距离小于安全距离时的惩罚
- **越界惩罚**（-2.0）：当无人机超出Leader扫描范围时的惩罚
- **能耗惩罚**（-0.1）：基于方向变化幅度的能耗惩罚
- **完成奖励**（100.0）：当扫描区域达到95%以上时的大额奖励

奖励计算公式：

```
总奖励 = 探索奖励 + 效率奖励 + 碰撞惩罚 + 越界惩罚 + 能耗惩罚 + 完成奖励
```

### 3.3 代码结构改动

#### 3.3.1 DroneLearningEnv类（已实现）

```python
class DroneLearningEnv:
    """无人机强化学习环境，提供OpenAI Gym风格的接口"""
    
    def __init__(self, server, drone_name):
        self.server = server
        self.drone_name = drone_name
        self.state_dim = 18  # 状态空间维度
        self.action_dim = 25  # 动作空间维度（5个权重×5种调整）
        
        # 权重调整配置
        self.coefficient_step = 0.5
        self.coefficient_ranges = {
            'repulsionCoefficient': (0.1, 10.0),
            'entropyCoefficient': (0.1, 10.0),
            'distanceCoefficient': (0.1, 10.0),
            'leaderRangeCoefficient': (0.1, 10.0),
            'directionRetentionCoefficient': (0.1, 10.0)
        }
        
    def reset(self):
        """重置环境并返回初始状态"""
        self.prev_state = None
        self.prev_scanned_area = self._calculate_scanned_area()
        return self.get_state()
        
    def step(self, action):
        """执行动作并返回状态、奖励、完成标志和信息"""
        # 1. 根据动作调整权重系数
        coefficients_adjustment = self._action_to_coefficients(action)
        # 2. 应用权重调整
        # 3. 计算奖励和完成状态
        return next_state, reward, done, info
        
    def get_state(self):
        """获取当前环境状态（18维向量）"""
        # 收集位置、速度、方向、权重系数、Leader位置、扫描效率
        return np.array(state, dtype=np.float32)
        
    def calculate_reward(self, prev_state, current_state):
        """计算奖励值（多组件奖励函数）"""
        # 计算探索、效率、碰撞、越界、能耗、完成奖励
        return total_reward
```

#### 3.3.2 DQNAgent类（已实现）

```python
class DQNAgent:
    """DQN智能体，实现完整的DQN算法"""
    
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, 
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 batch_size=64, target_update=10):
        # 创建策略网络和目标网络
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        
        # 经验回放缓冲区
        self.memory = ReplayBuffer(10000)
        
    def select_action(self, state):
        """ε-贪婪策略选择动作"""
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)  # 探索
        else:
            return self.policy_net(state).max(1)[1].item()  # 利用
            
    def learn(self):
        """从经验回放中学习，更新网络参数"""
        # 采样批次数据
        # 计算Q值和目标Q值
        # 反向传播更新网络
        
    def save_model(self, path):
        """保存训练好的模型"""
        
    def load_model(self, path):
        """加载预训练模型"""
```

#### 3.3.3 ScannerAlgorithm类扩展（需要实现）

```python
class ScannerAlgorithm:
    # 需要添加以下方法以支持DQN动态调整权重
    def set_coefficients(self, coefficients):
        """动态设置权重系数"""
        if 'repulsionCoefficient' in coefficients:
            self.config.repulsionCoefficient = coefficients['repulsionCoefficient']
        if 'entropyCoefficient' in coefficients:
            self.config.entropyCoefficient = coefficients['entropyCoefficient']
        # ... 其他系数设置
        
    def get_current_coefficients(self):
        """获取当前权重系数"""
        return {
            'repulsionCoefficient': self.config.repulsionCoefficient,
            'entropyCoefficient': self.config.entropyCoefficient,
            'distanceCoefficient': self.config.distanceCoefficient,
            'leaderRangeCoefficient': self.config.leaderRangeCoefficient,
            'directionRetentionCoefficient': self.config.directionRetentionCoefficient
        }
```

#### 3.3.4 MultiDroneAlgorithmServer类（已实现）

```python
class MultiDroneAlgorithmServer:
    def __init__(self, config_file=None, drone_names=None, enable_learning=False):
        # 现有初始化代码...
        self.enable_learning = enable_learning
        self.learning_envs = {}
        self.dqn_agents = {}
        
        if self.enable_learning:
            self._init_dqn_learning()
            
    def _init_dqn_learning(self):
        """初始化DQN学习组件"""
        from multirotor.DQN.DqnLearning import DQNAgent
        from multirotor.DQN.DroneLearningEnv import DroneLearningEnv
        
        for drone_name in self.drone_names:
            # 创建学习环境
            env = DroneLearningEnv(self, drone_name)
            self.learning_envs[drone_name] = env
            
            # 创建DQN智能体
            agent = DQNAgent(
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                lr=0.001,
                gamma=0.99,
                epsilon=1.0,
                epsilon_min=0.01,
                epsilon_decay=0.995,
                batch_size=64,
                target_update=10
            )
            self.dqn_agents[drone_name] = agent
            
            # 尝试加载已训练的模型
            try:
                model_path = f"DQN/dqn_{drone_name}_model.pth"
                agent.load_model(model_path)
                logger.info(f"已加载无人机{drone_name}的DQN模型")
            except Exception as e:
                logger.info(f"未找到无人机{drone_name}的DQN模型，将从头开始训练: {str(e)}")
                
    def _process_drone(self, drone_name):
        """无人机算法处理线程（已实现DQN集成）"""
        while self.running:
            if self.enable_learning and drone_name in self.learning_envs:
                # DQN学习模式
                try:
                    # 获取当前状态
                    state = self.learning_envs[drone_name].get_state()
                    
                    # DQN智能体选择动作
                    action = self.dqn_agents[drone_name].select_action(state)
                    
                    # 执行动作并获取反馈
                    next_state, reward, done, info = self.learning_envs[drone_name].step(action)
                    
                    # 存储经验
                    self.dqn_agents[drone_name].memory.push(state, action, reward, next_state, done)
                    
                    # DQN智能体学习
                    self.dqn_agents[drone_name].learn()
                    
                    # 定期保存模型
                    if self.dqn_agents[drone_name].steps_done % 1000 == 0:
                        model_path = f"DQN/dqn_{drone_name}_model_{self.dqn_agents[drone_name].steps_done}.pth"
                        self.dqn_agents[drone_name].save_model(model_path)
                        
                except Exception as e:
                    logger.error(f"无人机{drone_name}DQN学习处理出错: {str(e)}")
                    
            # 执行算法计算最终方向
            final_dir = self.algorithms[drone_name].update_runtime_data(
                self.grid_data, self.unity_runtime_data[drone_name]
            )
            
            # 控制无人机移动
            self._control_drone_movement(drone_name, final_dir.finalMoveDir)
            
            # 发送处理后的数据到Unity
            self._send_processed_data(drone_name, final_dir)
```

## 4. 使用方法

### 4.1 启用DQN学习模式

**方法1：通过配置文件启用**
在`scanner_config.json`中设置`"dqn": {"enabled": true}`：

```python
# 从配置文件读取DQN启用状态
server = MultiDroneAlgorithmServer()

# 启动服务
if server.start():
    server.start_mission()
    
    # 主循环保持运行
    while server.running:
        time.sleep(1)
```

**方法2：通过参数强制启用**
```python
# 强制启用DQN学习，忽略配置文件设置
server = MultiDroneAlgorithmServer(enable_learning=True)
```

### 4.2 传统模式（仅人工势场算法）

**方法1：通过配置文件禁用**
在`scanner_config.json`中设置`"dqn": {"enabled": false}`：

```python
# 从配置文件读取DQN禁用状态
server = MultiDroneAlgorithmServer()
```

**方法2：通过参数强制禁用**
```python
# 强制禁用DQN学习，忽略配置文件设置
server = MultiDroneAlgorithmServer(enable_learning=False)
```

### 4.3 模型保存与加载

DQN模型会自动保存到`DQN/`目录下：
- 训练过程中每1000步自动保存一次
- 模型文件名格式：`dqn_{drone_name}_model_{steps}.pth`
- 启动时会自动尝试加载最新的模型

### 4.4 可视化支持

项目支持实时可视化：
- 运行`python run_visualizer.py`启动独立可视化器
- 或在AlgorithmServer中启用内置可视化功能

## 5. 配置管理

### 5.1 DQN参数配置（已实现配置化）

DQN相关参数已移至配置文件`scanner_config.json`中，主要配置包括：

**DQNAgent参数**：
- 学习率：0.001（可配置）
- 折扣因子：0.99（可配置）
- 探索率：1.0 → 0.01（衰减率0.995，可配置）
- 批次大小：64（可配置）
- 目标网络更新频率：10步（可配置）
- 经验回放缓冲区大小：10000（可配置）
- 模型保存间隔：1000步（可配置）

**DroneLearningEnv参数**：
- 状态空间维度：18（固定）
- 动作空间维度：25（固定）
- 权重调整步长：0.5（可配置）
- 权重范围：0.1 - 10.0（可配置）

**奖励函数权重**：
- 探索奖励权重：1.0（可配置）
- 效率奖励权重：0.5（可配置）
- 碰撞惩罚：-5.0（可配置）
- 越界惩罚：-2.0（可配置）
- 能耗惩罚：-0.1（可配置）
- 完成奖励：100.0（可配置）

### 5.2 配置文件结构

DQN配置已集成到`scanner_config.json`中：

```json
{
  "dqn": {
    "enabled": false,
    "learning_rate": 0.001,
    "gamma": 0.99,
    "epsilon": 1.0,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.995,
    "batch_size": 64,
    "target_update": 10,
    "memory_capacity": 10000,
    "model_save_interval": 1000
  },
  "reward": {
    "exploration_weight": 1.0,
    "efficiency_weight": 0.5,
    "collision_penalty": -5.0,
    "boundary_penalty": -2.0,
    "energy_penalty": -0.1,
    "completion_reward": 100.0
  },
  "learning_env": {
    "coefficient_step": 0.5,
    "coefficient_ranges": {
      "repulsionCoefficient": [0.1, 10.0],
      "entropyCoefficient": [0.1, 10.0],
      "distanceCoefficient": [0.1, 10.0],
      "leaderRangeCoefficient": [0.1, 10.0],
      "directionRetentionCoefficient": [0.1, 10.0]
    }
  }
}
```

## 6. 实现状态

### 6.1 已完成功能

✅ **DQN核心算法**：完整的DQN实现，包括经验回放、目标网络等
✅ **学习环境**：DroneLearningEnv类，提供18维状态空间和25维动作空间
✅ **服务器集成**：MultiDroneAlgorithmServer已集成DQN功能
✅ **模型管理**：自动保存和加载训练好的模型
✅ **可视化支持**：实时可视化无人机状态和学习过程
✅ **多模式支持**：支持DQN学习模式和传统人工势场模式切换

### 6.2 待完善功能

⚠️ **ScannerAlgorithm扩展**：需要添加`set_coefficients`和`get_current_coefficients`方法
✅ **配置管理**：DQN参数已移至配置文件，支持灵活配置
⚠️ **性能优化**：DQN计算可能影响实时性能，需要进一步优化
⚠️ **多智能体协作**：当前为单智能体学习，可扩展为多智能体协作

## 7. 性能与安全性考虑

1. **计算效率**：DQN计算在现有线程中执行，可能影响实时控制性能
2. **内存管理**：经验回放缓冲区大小为10000，可根据系统内存调整
3. **模型安全**：训练过程中每1000步自动保存模型
4. **参数范围限制**：权重系数限制在0.1-10.0范围内，确保系统稳定性
5. **异常处理**：DQN学习出错时会回退到传统人工势场算法

## 8. 未来扩展方向

1. **多智能体协作学习**：实现多无人机协同学习
2. **高级算法集成**：集成DDPG、PPO等更先进的强化学习算法
3. **迁移学习**：将在模拟器中训练的模型应用到真实无人机
4. **自适应参数调优**：实现DQN超参数的自适应调整
5. **分布式训练**：支持多机分布式DQN训练

## 9. 总结

项目已成功实现DQN学习与人工势场算法的集成，主要特点：

- **无缝集成**：DQN学习功能完全集成到现有系统中
- **灵活切换**：支持学习模式和传统模式之间的灵活切换
- **实时学习**：在无人机运行过程中进行在线学习
- **模型持久化**：自动保存和加载训练好的模型
- **可视化支持**：提供实时可视化界面

通过DQN学习，无人机系统能够根据环境反馈不断优化其控制策略，提高任务执行效率和适应性，为无人机自主控制提供了强大的学习能力。