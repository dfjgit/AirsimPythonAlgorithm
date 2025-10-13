# DQN V2 设计文档

**版本**: V2.0  
**状态**: 📝 设计中  
**创建日期**: 2025-10-13

基于 [dqn_training_workflow.png](../../docs/images/dqn_training_workflow.png) 的新架构设计

---

## 🎯 核心设计理念

### V2与V1的关键区别

| 维度 | V1设计 | V2设计 |
|------|--------|--------|
| **学习目标** | 直接学习动作 | 学习APF权重系数 |
| **动作空间** | 25维（权重调整） | 连续空间（α1, α2, α3） |
| **执行方式** | DQN选动作→APF执行 | DQN预测权重→APF执行 |
| **部署模式** | 实时学习 | 离线训练+在线推理 |
| **性能开销** | 高（每步学习） | 低（只做推理） |

---

## 🏗️ 三种工作模式

根据架构图，系统支持三种工作模式：

### 模式1️⃣: DQN训练流程

**目标**: 学习最优的APF权重系数（α1, α2, α3, α4, α5）

```
┌─────────────────────────────────────────┐
│           DQN训练流程                    │
├─────────────────────────────────────────┤
│                                         │
│  动作空间                               │
│  (由APF计算得出)                        │
│         ↓                               │
│  状态空间                               │
│  - 空间区域信息（熵值分布）             │
│  - 无人机信息（位置、速度、方向）       │
│         ↓                               │
│  Agent根据状态空间调用APF算法            │
│  获得下一步动作                          │
│  (固定 α1, α2, α3, α4, α5)             │
│         ↓                               │
│  奖励函数                               │
│  (对当前动作进行评分)                   │
│         ↓                               │
│  循环训练                               │
└─────────────────────────────────────────┘
```

**关键点**:
- DQN不直接输出动作，而是输出**权重系数**
- APF使用这些权重计算最终动作
- 奖励函数评估动作的质量

### 模式2️⃣: 训练后调用流程（推理模式）

**目标**: 使用训练好的模型进行快速推理

```
┌─────────────────────────────────────────┐
│         训练后调用流程                   │
├─────────────────────────────────────────┤
│                                         │
│  状态空间                               │
│         ↓                               │
│  pth/ONNX                               │
│  (量化子模型)                           │
│         ↓                               │
│  α1, α2, α3, α4, α5                    │
│         ↓                               │
│  APF算法                                │
│         ↓                               │
│  下一步动作                             │
└─────────────────────────────────────────┘
```

**关键点**:
- 轻量级推理（ONNX/量化模型）
- 快速输出权重系数
- 低CPU开销

### 模式3️⃣: APF执行流程（传统模式）

**目标**: 使用固定权重直接执行

```
┌─────────────────────────────────────────┐
│          APF执行流程                     │
├─────────────────────────────────────────┤
│                                         │
│  状态空间                               │
│         ↓                               │
│  APF                                    │
│  (α1, α2, α3, α4, α5)                  │
│  [配置文件中的固定值]                   │
│         ↓                               │
│  下一步动作                             │
└─────────────────────────────────────────┘
```

**关键点**:
- 使用配置文件中的固定权重
- 无机器学习开销
- 当前系统使用的模式

---

## 📐 技术架构设计

### 状态空间设计

**维度**: 约15-20维

```python
state = [
    # 空间区域信息 (6维)
    average_entropy_nearby,      # 附近区域平均熵值
    max_entropy_nearby,          # 附近区域最大熵值
    scanned_ratio,               # 已扫描区域比例
    unexplored_direction_x,      # 未探索方向X
    unexplored_direction_y,      # 未探索方向Y
    unexplored_direction_z,      # 未探索方向Z
    
    # 无人机信息 (9维)
    position_x, position_y, position_z,           # 位置
    velocity_x, velocity_y, velocity_z,           # 速度
    direction_x, direction_y, direction_z,        # 朝向
    
    # Leader相关 (3维)
    distance_to_leader,          # 到Leader距离
    relative_position_x,         # 相对Leader位置X
    relative_position_z,         # 相对Leader位置Z
]
```

### 动作空间设计

**类型**: 连续动作空间

```python
action = [
    α1,  # repulsionCoefficient      (排斥力权重)
    α2,  # entropyCoefficient        (熵权重)
    α3,  # distanceCoefficient       (距离权重)
    α4,  # leaderRangeCoefficient    (Leader范围权重)
    α5,  # directionRetentionCoefficient (方向保持权重)
]

# 取值范围: [0.1, 10.0]
# 输出层激活函数: Sigmoid缩放到有效范围
```

### 奖励函数设计

```python
reward = (
    # 探索奖励（主要目标）
    + w1 * new_scanned_area                    # 新扫描区域面积
    + w2 * entropy_reduction                   # 熵值降低量
    
    # 效率奖励
    + w3 * scan_efficiency                     # 单位时间扫描效率
    - w4 * path_length                         # 路径长度惩罚
    
    # 安全惩罚
    - w5 * collision_risk                      # 碰撞风险
    - w6 * boundary_violation                  # 超出Leader范围
    
    # 稳定性奖励
    + w7 * smooth_movement                     # 运动平滑度
    - w8 * direction_change                    # 方向变化惩罚
)
```

---

## 🔧 技术实现方案

### 方案选择：离线训练 + 在线推理

#### 阶段1: 离线训练（在GPU机器上）

**训练环境**：
- 使用模拟数据或AirSim录制的轨迹
- GPU加速训练
- 大批量经验回放

**网络架构**：
```python
class WeightPredictionNetwork(nn.Module):
    """预测APF权重系数的神经网络"""
    
    def __init__(self, state_dim=18, weight_dim=5):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, weight_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # 输出权重范围 [0.1, 10.0]
        weights = torch.sigmoid(self.fc4(x)) * 9.9 + 0.1
        return weights
```

**训练算法**: DDPG（Deep Deterministic Policy Gradient）
- 适合连续动作空间
- 比DQN更适合权重预测
- Actor-Critic架构

#### 阶段2: 模型转换（优化部署）

```
PyTorch模型 (.pth)
    ↓
ONNX格式 (.onnx)
    ↓
量化优化（INT8）
    ↓
轻量级模型 (< 5MB)
```

**工具**：
- ONNX Runtime: 高性能推理
- 量化: 减少模型大小和计算量
- 优化: 针对CPU优化

#### 阶段3: 在线推理（在目标机器上）

**推理引擎**: ONNX Runtime（纯CPU）

```python
class WeightPredictor:
    """轻量级权重预测器"""
    
    def __init__(self, model_path):
        import onnxruntime as ort
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']  # 强制CPU
        )
        
    def predict_weights(self, state):
        """预测APF权重系数"""
        input_name = self.session.get_inputs()[0].name
        output = self.session.run(None, {input_name: state})
        weights = output[0]  # [α1, α2, α3, α4, α5]
        return weights
```

---

## 📊 数据流设计

### 训练阶段数据流

```
环境状态（Unity/AirSim）
    ↓
状态提取器
    ↓
[position, velocity, entropy_map, leader_info]
    ↓
DDPG Actor网络
    ↓
[α1, α2, α3, α4, α5]
    ↓
APF算法
    ↓
最终移动方向
    ↓
执行并观察反馈
    ↓
计算奖励
    ↓
存储经验 (state, weights, reward, next_state)
    ↓
DDPG Critic网络评估
    ↓
更新Actor和Critic
```

### 推理阶段数据流

```
环境状态
    ↓
状态归一化
    ↓
ONNX模型推理
    ↓
[α1, α2, α3, α4, α5]
    ↓
直接传给APF算法
    ↓
最终移动方向
    ↓
执行
```

---

## 🎯 实现计划

### Phase 1: 数据收集（1-2周）

**目标**: 收集训练数据

```python
# 数据收集器
class TrajectoryCollector:
    """收集无人机飞行轨迹数据"""
    
    def collect_episode(self):
        """收集一个完整任务的数据"""
        episode_data = []
        
        while not done:
            # 收集状态
            state = extract_state()
            
            # 使用当前权重
            current_weights = config.get_weights()
            
            # 执行APF
            action = apf.compute(state, current_weights)
            
            # 执行并获取反馈
            next_state, reward = execute_and_observe(action)
            
            # 记录
            episode_data.append({
                'state': state,
                'weights': current_weights,
                'action': action,
                'reward': reward,
                'next_state': next_state
            })
        
        return episode_data
```

**数据集要求**:
- 至少100个完整任务
- 包含不同场景（开阔、障碍物、多无人机）
- 包含不同权重配置的结果

### Phase 2: 训练环境搭建（1周）

**环境**: Google Colab（免费GPU）或本地GPU机器

```python
# 训练环境
class WeightLearningEnv:
    """学习APF权重的环境"""
    
    def __init__(self):
        self.state_dim = 18
        self.action_dim = 5  # 5个权重系数
        
    def reset(self):
        """重置环境"""
        # 从数据集中加载初始状态
        return initial_state
    
    def step(self, weights):
        """
        执行一步
        :param weights: [α1, α2, α3, α4, α5]
        :return: next_state, reward, done, info
        """
        # 使用APF计算动作
        action = self.apf.compute(self.state, weights)
        
        # 模拟执行（使用数据集或仿真）
        next_state = self.simulate(action)
        
        # 计算奖励
        reward = self.compute_reward(weights, action, next_state)
        
        return next_state, reward, done, info
```

### Phase 3: DDPG训练（2-3周）

**算法选择**: DDPG（Deep Deterministic Policy Gradient）

**原因**:
- ✅ 适合连续动作空间（权重是连续值）
- ✅ Actor-Critic架构稳定
- ✅ 样本效率高
- ✅ 成熟的实现（Stable-Baselines3）

**训练代码**:
```python
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise

# 创建环境
env = WeightLearningEnv()

# 添加探索噪声
n_actions = env.action_dim
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions),
    sigma=0.1 * np.ones(n_actions)
)

# 创建DDPG模型
model = DDPG(
    "MlpPolicy",
    env,
    action_noise=action_noise,
    learning_rate=1e-3,
    buffer_size=100000,
    learning_starts=1000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    verbose=1
)

# 训练
model.learn(total_timesteps=100000)

# 保存模型
model.save("apf_weight_predictor")
```

### Phase 4: 模型转换（1周）

**转换流程**:

```python
import torch
import onnx
from onnxruntime.quantization import quantize_dynamic

# 1. 导出PyTorch模型到ONNX
dummy_input = torch.randn(1, state_dim)
torch.onnx.export(
    model,
    dummy_input,
    "weight_predictor.onnx",
    export_params=True,
    opset_version=11,
    input_names=['state'],
    output_names=['weights']
)

# 2. 量化模型（减小尺寸，提高速度）
quantize_dynamic(
    "weight_predictor.onnx",
    "weight_predictor_quantized.onnx",
    weight_type=QuantType.QUInt8
)

# 3. 验证模型
import onnxruntime as ort
session = ort.InferenceSession("weight_predictor_quantized.onnx")
test_output = session.run(None, {'state': test_state})
print(f"预测权重: {test_output[0]}")
```

### Phase 5: 集成到AlgorithmServer（1周）

**实现方式**: 轻量级推理模式

```python
class MultiDroneAlgorithmServer:
    def __init__(self, config_file=None, use_learned_weights=False):
        # ... 现有初始化 ...
        
        self.use_learned_weights = use_learned_weights
        self.weight_predictor = None
        
        if self.use_learned_weights:
            self._init_weight_predictor()
    
    def _init_weight_predictor(self):
        """初始化轻量级权重预测器"""
        try:
            import onnxruntime as ort
            model_path = os.path.join(
                os.path.dirname(__file__),
                'DQN', 'models', 'weight_predictor_quantized.onnx'
            )
            
            if os.path.exists(model_path):
                self.weight_predictor = ort.InferenceSession(
                    model_path,
                    providers=['CPUExecutionProvider']
                )
                logger.info(f"权重预测模型加载成功: {model_path}")
            else:
                logger.warning("未找到权重预测模型，使用配置文件权重")
                self.use_learned_weights = False
                
        except Exception as e:
            logger.error(f"权重预测器初始化失败: {str(e)}")
            self.use_learned_weights = False
    
    def _get_apf_weights(self, drone_name):
        """获取APF权重系数"""
        if self.use_learned_weights and self.weight_predictor:
            # 使用模型预测
            state = self._extract_state(drone_name)
            weights = self._predict_weights(state)
            return weights
        else:
            # 使用配置文件固定值
            return {
                'repulsionCoefficient': self.config_data.repulsionCoefficient,
                'entropyCoefficient': self.config_data.entropyCoefficient,
                'distanceCoefficient': self.config_data.distanceCoefficient,
                'leaderRangeCoefficient': self.config_data.leaderRangeCoefficient,
                'directionRetentionCoefficient': self.config_data.directionRetentionCoefficient
            }
    
    def _predict_weights(self, state):
        """使用ONNX模型预测权重"""
        input_name = self.weight_predictor.get_inputs()[0].name
        output = self.weight_predictor.run(None, {input_name: state.reshape(1, -1)})
        weights_array = output[0][0]  # [α1, α2, α3, α4, α5]
        
        return {
            'repulsionCoefficient': float(weights_array[0]),
            'entropyCoefficient': float(weights_array[1]),
            'distanceCoefficient': float(weights_array[2]),
            'leaderRangeCoefficient': float(weights_array[3]),
            'directionRetentionCoefficient': float(weights_array[4])
        }
    
    def _process_drone(self, drone_name):
        """无人机处理线程（支持学习权重）"""
        while self.running:
            # 获取当前最优权重（可能来自模型预测）
            weights = self._get_apf_weights(drone_name)
            
            # 临时设置权重
            self.algorithms[drone_name].set_coefficients(weights)
            
            # 执行APF算法
            final_dir = self.algorithms[drone_name].update_runtime_data(
                self.grid_data, self.unity_runtime_data[drone_name]
            )
            
            # 控制无人机移动
            self._control_drone_movement(drone_name, final_dir.finalMoveDir)
            
            time.sleep(self.config_data.updateInterval)
```

---

## 📊 性能对比

| 模式 | CPU占用 | 内存占用 | 响应延迟 | 效果 |
|------|---------|----------|----------|------|
| **APF固定权重** | 5-10% | ~100MB | <50ms | 基线 |
| **ONNX推理** | 8-12% | ~150MB | <100ms | 自适应 |
| **V1实时训练** | 40-60% | ~500MB | 200-500ms | ❌卡顿 |

---

## 🎓 关键改进

### 相比V1的优势

1. **性能优化**：
   - ✅ 只做推理，不做训练
   - ✅ ONNX Runtime高效
   - ✅ INT8量化加速
   - ✅ CPU占用低

2. **架构简化**：
   - ✅ 学习目标明确（只学权重）
   - ✅ 动作空间小（5维连续）
   - ✅ 与APF解耦
   - ✅ 易于调试

3. **部署友好**：
   - ✅ 无PyTorch依赖
   - ✅ 模型文件小
   - ✅ 可选功能（不影响基础功能）
   - ✅ 平滑降级

4. **训练效率**：
   - ✅ 离线训练无时间压力
   - ✅ 可使用GPU加速
   - ✅ 样本效率高
   - ✅ 训练质量好

---

## 🗂️ 文件组织

### 训练阶段（GPU环境）

```
training/                           # 训练专用目录
├── train_ddpg.py                  # DDPG训练脚本
├── weight_learning_env.py         # 训练环境
├── data_collector.py              # 数据收集器
├── dataset/                       # 训练数据集
│   ├── trajectory_001.json
│   ├── trajectory_002.json
│   └── ...
├── models/                        # 训练中的模型
│   ├── checkpoints/
│   └── logs/
└── requirements_training.txt      # 训练依赖
```

### 部署阶段（目标环境）

```
multirotor/DQN/
├── models/                        # 部署模型
│   └── weight_predictor_quantized.onnx
├── weight_predictor.py            # 推理封装
└── requirements_inference.txt     # 只需onnxruntime
```

---

## 📝 配置文件扩展

### scanner_config.json 新增配置

```json
{
    // ... 现有APF配置 ...
    
    "weight_prediction": {
        "enabled": false,                    // 是否启用权重预测
        "model_path": "DQN/models/weight_predictor_quantized.onnx",
        "update_frequency": 10,              // 权重更新频率（每N步）
        "fallback_to_config": true          // 模型失败时回退到配置权重
    }
}
```

---

## 🔬 实验设计

### 实验1: 基线性能测试
- 使用固定权重配置
- 记录扫描效率、覆盖率、时间
- 建立性能基线

### 实验2: 不同场景测试
- 开阔区域
- 密集障碍物
- 多无人机（2-5架）
- 找出每种场景的最优权重

### 实验3: 权重预测训练
- 使用收集的数据训练DDPG
- 验证预测权重的效果
- 对比固定权重和学习权重

### 实验4: 性能评估
- CPU占用率
- 内存占用
- 推理延迟
- 扫描效率提升

---

## ✅ 验收标准

### 功能验收
- [ ] ONNX模型能正确预测5个权重
- [ ] 预测权重范围在 [0.1, 10.0]
- [ ] 集成到AlgorithmServer无报错
- [ ] 可以在配置文件中启用/禁用

### 性能验收
- [ ] CPU占用 < 15%
- [ ] 推理延迟 < 50ms
- [ ] 内存增加 < 100MB
- [ ] 不影响主循环实时性

### 效果验收
- [ ] 学习权重优于固定权重（至少10%）
- [ ] 在不同场景下都能自适应
- [ ] 系统稳定运行无崩溃

---

## 🚀 开发路线图

### 里程碑1: 数据准备（Week 1-2）
- [ ] 实现轨迹数据收集器
- [ ] 收集至少100个任务的数据
- [ ] 数据预处理和标注

### 里程碑2: 训练环境（Week 3）
- [ ] 实现WeightLearningEnv
- [ ] 定义状态空间和奖励函数
- [ ] 单元测试

### 里程碑3: 模型训练（Week 4-5）
- [ ] 实现DDPG训练脚本
- [ ] 训练模型（GPU环境）
- [ ] 超参数调优
- [ ] 模型评估

### 里程碑4: 模型部署（Week 6）
- [ ] 转换为ONNX格式
- [ ] 模型量化优化
- [ ] 集成到AlgorithmServer
- [ ] 性能测试

### 里程碑5: 测试与优化（Week 7-8）
- [ ] 完整功能测试
- [ ] 性能优化
- [ ] 文档完善
- [ ] 发布V2.0

---

## 💡 关键技术点

### 1. 为什么学习权重而不是动作？

**优势**:
- APF算法已经很好，只需要找最优权重
- 动作空间小（5维 vs 25维）
- 学习更快，更稳定
- 可解释性强（知道每个权重的含义）

### 2. 为什么使用DDPG而不是DQN？

**原因**:
- DQN适合离散动作
- DDPG适合连续动作（权重是连续值）
- DDPG收敛更快
- 更适合控制问题

### 3. 为什么离线训练？

**原因**:
- 避免实时训练的性能开销
- 可以使用GPU加速
- 训练质量更高
- 部署更简单

### 4. 为什么使用ONNX？

**原因**:
- 跨平台
- 推理速度快
- 无需PyTorch依赖
- 模型文件小

---

## 🔄 与V1的对比

| 维度 | V1 | V2 |
|------|----|----|
| **学习方式** | 在线实时学习 | 离线预训练 |
| **算法** | DQN | DDPG |
| **动作空间** | 25维离散 | 5维连续 |
| **学习目标** | 权重调整幅度 | 权重绝对值 |
| **部署** | PyTorch | ONNX Runtime |
| **性能** | 40-60% CPU | <15% CPU |
| **状态** | ❌ 已废弃 | ✅ 设计中 |

---

## 📚 参考资料

### 算法论文
- DDPG: Continuous control with deep reinforcement learning (2015)
- APF: Real-time obstacle avoidance using harmonic potential functions (1996)

### 工具文档
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [AirSim](https://microsoft.github.io/AirSim/)

### 项目文档
- [V1归档](./README_V1_ARCHIVED.md) - V1实现和教训
- [CPU优化](./CPU_OPTIMIZATION.md) - 性能优化指南
- [需求模板](./V2_REQUIREMENTS.md) - 详细需求规划

---

## 🎯 下一步行动

1. **评审设计方案** - 确认技术路线
2. **搭建训练环境** - 准备GPU环境
3. **收集训练数据** - 运行数据收集器
4. **开始训练** - 实现DDPG训练
5. **测试部署** - 验证ONNX推理

---

**设计人**: AI Assistant  
**日期**: 2025-10-13  
**版本**: V2.0 Design Draft

