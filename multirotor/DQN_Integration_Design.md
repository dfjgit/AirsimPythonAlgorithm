# DQN学习与人工势场算法集成设计文档

## 1. 项目现状分析

### 1.1 当前系统架构

当前系统通过 `MultiDroneAlgorithmServer` 类作为核心，连接Unity客户端与AirSim模拟器，实现多无人机协同控制：
- **数据流程**：Unity → AlgorithmServer → AirSim → 无人机
- **控制算法**：采用人工势场算法（`ScannerAlgorithm`类）
- **核心控制循环**：通过`_process_drone`方法周期性计算无人机移动方向并发送控制指令

### 1.2 人工势场算法原理

人工势场算法通过合并以下五个方向向量计算最终移动方向：
1. **熵最优方向**：引导无人机探索未知区域
2. **最短路径方向**：引导无人机向目标移动
3. **排斥力方向**：避免无人机间碰撞
4. **保持Leader范围方向**：保持无人机在Leader范围内
5. **方向保持方向**：保持无人机飞行稳定性

每个方向向量有对应的权重系数，这些系数目前是固定的，通过配置文件设置。

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

DQN智能体的输入状态空间包括：
- 当前无人机位置和速度
- 周边障碍物/其他无人机位置
- Leader无人机位置和状态
- 网格熵分布信息
- 当前人工势场算法计算的各方向向量
- 已扫描区域比例和效率指标

状态空间维度示例：20-50维连续向量

#### 3.2.2 动作空间定义

DQN智能体的输出动作空间：
- 调整人工势场算法的五个权重系数：
  - 排斥力系数(repulsionCoefficient)
  - 熵系数(entropyCoefficient)
  - 距离系数(distanceCoefficient)
  - Leader范围系数(leaderRangeCoefficient)
  - 方向保持系数(directionRetentionCoefficient)

可以采用离散化的权重调整步长，如每个系数有5-10个离散取值。

#### 3.2.3 奖励函数设计

奖励函数应包含以下几个部分：
- **探索奖励**：扫描新区域获得正奖励
- **效率奖励**：高效扫描(单位时间扫描面积)获得正奖励
- **碰撞惩罚**：无人机间距离过近或碰撞获得负奖励
- **越界惩罚**：超出Leader范围或地图边界获得负奖励
- **能耗惩罚**：不必要的大幅度转向或加速获得负奖励
- **目标完成奖励**：完成扫描任务获得大额正奖励

### 3.3 代码结构改动

#### 3.3.1 创建DroneLearningEnv类

```python
class DroneLearningEnv:
    """无人机强化学习环境，提供OpenAI Gym风格的接口"""
    def __init__(self, server, drone_name):
        self.server = server  # 引用AlgorithmServer实例
        self.drone_name = drone_name
        self.state_dim = ...  # 状态空间维度
        self.action_dim = ...  # 动作空间维度
        
    def reset(self):
        """重置环境并返回初始状态"""
        # 重置无人机状态和学习环境
        return initial_state
        
    def step(self, action):
        """执行动作并返回状态、奖励、完成标志和信息"""
        # 1. 根据动作调整权重系数
        # 2. 执行一个控制周期
        # 3. 计算状态、奖励和完成标志
        return next_state, reward, done, info
        
    def get_state(self):
        """获取当前环境状态"""
        # 收集并处理环境状态信息
        return state
        
    def calculate_reward(self, prev_state, current_state):
        """计算奖励值"""
        # 根据状态变化计算奖励
        return reward
```

#### 3.3.2 修改ScannerAlgorithm类

```python
class ScannerAlgorithm:
    # 在现有代码基础上添加以下功能
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
            # ... 其他系数
        }
```

#### 3.3.3 修改MultiDroneAlgorithmServer类

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
        from DQN.DqnLearning import DQNAgent
        
        for drone_name in self.drone_names:
            # 创建学习环境
            env = DroneLearningEnv(self, drone_name)
            self.learning_envs[drone_name] = env
            
            # 创建DQN智能体
            agent = DQNAgent(
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                # 其他DQN参数...
            )
            self.dqn_agents[drone_name] = agent
            
            # 尝试加载已训练的模型
            try:
                model_path = f"dqn_{drone_name}_model.pth"
                agent.load_model(model_path)
                print(f"已加载无人机{drone_name}的DQN模型")
            except:
                print(f"未找到无人机{drone_name}的DQN模型，将从头开始训练")
                
    def _process_drone(self, drone_name):
        """无人机算法处理线程"""
        # 现有处理逻辑...
        
        if self.enable_learning:
            # 获取当前状态
            state = self.learning_envs[drone_name].get_state()
            
            # DQN智能体选择动作
            action = self.dqn_agents[drone_name].select_action(state)
            
            # 执行动作并获取反馈
            next_state, reward, done, info = self.learning_envs[drone_name].step(action)
            
            # DQN智能体学习
            self.dqn_agents[drone_name].learn()
            
            # 定期保存模型
            if self.dqn_agents[drone_name].steps_done % 1000 == 0:
                model_path = f"dqn_{drone_name}_model_{self.dqn_agents[drone_name].steps_done}.pth"
                self.dqn_agents[drone_name].save_model(model_path)
        else:
            # 原有人工势场算法逻辑
            final_dir = self.algorithms[drone_name].update_runtime_data(
                self.grid_data, self.unity_runtime_data[drone_name]
            )
            
            # 控制无人机移动
            self._control_drone_movement(drone_name, final_dir.finalMoveDir)
            
        # 发送处理后的数据到Unity
        # ... 现有代码
```

#### 3.3.4 创建DQN训练专用入口

```python
# 创建 train_drone_dqn.py
import time
from AlgorithmServer import MultiDroneAlgorithmServer

if __name__ == "__main__":
    try:
        # 创建启用DQN学习的服务器实例
        server = MultiDroneAlgorithmServer(enable_learning=True)
        
        if server.start():
            # 启动训练任务
            server.start_mission()
            
            # 训练循环
            training_episodes = 1000
            max_steps_per_episode = 1000
            
            for episode in range(training_episodes):
                print(f"开始训练回合 {episode+1}/{training_episodes}")
                
                # 重置环境（如果需要）
                # server.reset_environment()
                
                # 运行一个训练回合
                start_time = time.time()
                steps = 0
                
                while steps < max_steps_per_episode and server.running:
                    time.sleep(0.1)  # 避免占用过多CPU
                    steps += 1
                    
                print(f"回合 {episode+1} 结束，耗时: {time.time() - start_time:.2f}秒")
                
    except KeyboardInterrupt:
        print("用户中断训练")
    finally:
        if 'server' in locals():
            server.stop()
```

## 4. 配置管理

### 4.1 新增DQN配置项

在 `scanner_config.json` 中添加DQN相关配置：

```json
{
  "dqn": {
    "enabled": true,
    "learning_rate": 0.001,
    "gamma": 0.99,
    "epsilon": 1.0,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.995,
    "batch_size": 64,
    "target_update": 10,
    "memory_capacity": 10000,
    "train_episodes": 1000,
    "steps_per_episode": 1000,
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
  // 原有配置项...
}
```

### 4.2 修改ScannerConfigData类

扩展 `ScannerConfigData` 类以支持DQN配置的加载和解析。

## 5. 实现步骤与优先级

1. **第一阶段：基础架构准备**
   - 创建 `DroneLearningEnv` 类
   - 扩展 `ScannerAlgorithm` 类支持动态调整权重
   - 扩展 `ScannerConfigData` 类支持DQN配置

2. **第二阶段：DQN集成**
   - 修改 `MultiDroneAlgorithmServer` 类集成DQN功能
   - 实现状态空间、动作空间和奖励函数
   - 创建DQN训练入口

3. **第三阶段：测试与调优**
   - 单元测试各个组件功能
   - 集成测试DQN与人工势场算法协作
   - 训练并调优DQN模型

## 6. 性能与安全性考虑

1. **计算效率**：DQN计算应在独立线程中执行，避免影响实时控制性能
2. **内存管理**：经验回放缓冲区大小应根据系统内存合理设置
3. **模型安全**：训练过程中定期保存模型，防止意外情况导致训练数据丢失
4. **参数范围限制**：DQN调整的权重系数应设置合理的上下限，确保系统稳定性

## 7. 未来扩展方向

1. 支持多智能体协作学习
2. 集成更先进的强化学习算法（如DDPG、PPO等）
3. 实现迁移学习，将在模拟器中训练的模型应用到真实无人机
4. 开发可视化工具，展示DQN学习过程和效果

---

通过上述设计，我们可以在保留现有人工势场算法的基础上，成功集成DQN学习功能，使无人机系统能够通过环境反馈不断优化其控制策略，提高任务执行效率和适应性。