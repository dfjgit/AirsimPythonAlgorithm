# DQN V2 实现指南

基于权重预测的轻量级实现方案

---

## 🎯 核心思想

**不学习动作，而是学习权重！**

```
传统DQN:  状态 → DQN → 动作
V2设计:   状态 → DDPG → 权重 → APF → 动作
```

**关键优势**:
- 动作空间小（5维连续 vs 25维离散）
- 充分利用现有APF算法
- 可解释性强（知道为什么这样飞）

---

## 📝 快速实现步骤

### 步骤1: 数据收集器（最简版）

创建文件: `multirotor/DQN/data_collector.py`

```python
"""
轨迹数据收集器
在正常运行时记录状态、权重、奖励
"""
import json
import time
import numpy as np
from datetime import datetime


class TrajectoryCollector:
    """收集无人机飞行轨迹数据供DQN训练使用"""
    
    def __init__(self, output_dir='training/dataset'):
        self.output_dir = output_dir
        self.current_episode = []
        self.episode_count = 0
        
    def start_episode(self):
        """开始新的数据收集episode"""
        self.current_episode = []
        
    def record_step(self, drone_name, state_dict, weights_dict, reward):
        """
        记录一步数据
        
        :param state_dict: {
            'position': [x, y, z],
            'velocity': [vx, vy, vz],
            'entropy_nearby': float,
            'distance_to_leader': float,
            ...
        }
        :param weights_dict: {
            'repulsionCoefficient': float,
            'entropyCoefficient': float,
            ...
        }
        :param reward: float - 这一步的奖励值
        """
        step_data = {
            'timestamp': time.time(),
            'drone_name': drone_name,
            'state': state_dict,
            'weights': weights_dict,
            'reward': reward
        }
        self.current_episode.append(step_data)
    
    def end_episode(self, success=True):
        """结束当前episode并保存"""
        if len(self.current_episode) == 0:
            return
        
        # 计算episode统计
        total_reward = sum(step['reward'] for step in self.current_episode)
        
        episode_data = {
            'episode_id': self.episode_count,
            'timestamp': datetime.now().isoformat(),
            'steps': len(self.current_episode),
            'total_reward': total_reward,
            'success': success,
            'trajectory': self.current_episode
        }
        
        # 保存到文件
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        filename = f"{self.output_dir}/episode_{self.episode_count:04d}.json"
        
        with open(filename, 'w') as f:
            json.dump(episode_data, f, indent=2)
        
        print(f"保存episode {self.episode_count}: {len(self.current_episode)}步, 总奖励: {total_reward:.2f}")
        
        self.episode_count += 1
        self.current_episode = []
```

**使用方法**（在AlgorithmServer中）:
```python
# 在__init__中
self.data_collector = TrajectoryCollector() if enable_data_collection else None

# 在_process_drone中
if self.data_collector:
    state = self._extract_state(drone_name)
    weights = self._get_current_weights(drone_name)
    reward = self._calculate_reward(drone_name)
    self.data_collector.record_step(drone_name, state, weights, reward)
```

---

### 步骤2: 状态提取器

创建文件: `multirotor/DQN/state_extractor.py`

```python
"""
从运行时数据中提取DQN状态向量
"""
import numpy as np


class StateExtractor:
    """提取DQN状态向量"""
    
    @staticmethod
    def extract(server, drone_name):
        """
        提取状态向量
        
        :return: numpy array, shape=(18,)
        """
        with server.data_lock:
            runtime_data = server.unity_runtime_data[drone_name]
            grid_data = server.grid_data
            
            # 1. 无人机位置 (3维)
            pos = runtime_data.position
            position = [pos.x, pos.y, pos.z]
            
            # 2. 无人机速度 (3维)
            vel = runtime_data.finalMoveDir * server.config_data.moveSpeed
            velocity = [vel.x, vel.y, vel.z]
            
            # 3. 无人机朝向 (3维)
            fwd = runtime_data.forward
            direction = [fwd.x, fwd.y, fwd.z]
            
            # 4. 附近熵值信息 (3维)
            entropy_info = StateExtractor._get_entropy_info(
                grid_data, pos, server.config_data.scanRadius
            )
            
            # 5. Leader相对信息 (3维)
            leader_info = [0.0, 0.0, 0.0]
            if runtime_data.leader_position:
                leader_pos = runtime_data.leader_position
                leader_info = [
                    leader_pos.x - pos.x,
                    leader_pos.y - pos.y,
                    leader_pos.z - pos.z
                ]
            
            # 6. 扫描进度 (3维)
            scan_info = StateExtractor._get_scan_info(grid_data)
        
        # 组合状态向量 (18维)
        state = np.array(
            position + velocity + direction + 
            entropy_info + leader_info + scan_info,
            dtype=np.float32
        )
        
        return state
    
    @staticmethod
    def _get_entropy_info(grid_data, position, radius):
        """获取附近区域的熵值信息"""
        if not grid_data or not grid_data.cells:
            return [0.0, 0.0, 0.0]
        
        # 找到附近的单元格
        nearby_cells = [
            cell for cell in grid_data.cells
            if (cell.center - position).magnitude() < radius * 2
        ]
        
        if not nearby_cells:
            return [50.0, 50.0, 0.0]  # 默认中等熵值
        
        entropies = [cell.entropy for cell in nearby_cells]
        
        return [
            np.mean(entropies),      # 平均熵值
            np.max(entropies),       # 最大熵值
            np.std(entropies)        # 熵值标准差
        ]
    
    @staticmethod
    def _get_scan_info(grid_data):
        """获取扫描进度信息"""
        if not grid_data or not grid_data.cells:
            return [0.0, 0.0, 0.0]
        
        total_cells = len(grid_data.cells)
        scanned_cells = sum(1 for cell in grid_data.cells if cell.entropy < 30)
        
        return [
            scanned_cells / max(total_cells, 1),     # 扫描比例
            scanned_cells,                           # 已扫描数量
            total_cells - scanned_cells              # 未扫描数量
        ]
```

---

### 步骤3: 轻量级推理器

创建文件: `multirotor/DQN/weight_predictor.py`

```python
"""
轻量级权重预测器（ONNX推理）
"""
import os
import numpy as np


class WeightPredictor:
    """使用ONNX模型预测APF权重系数"""
    
    def __init__(self, model_path=None):
        """
        初始化预测器
        :param model_path: ONNX模型路径
        """
        self.session = None
        self.model_loaded = False
        
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
    
    def _load_model(self, model_path):
        """加载ONNX模型"""
        try:
            import onnxruntime as ort
            
            # 创建推理会话（强制CPU）
            self.session = ort.InferenceSession(
                model_path,
                providers=['CPUExecutionProvider']
            )
            
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            self.model_loaded = True
            print(f"✓ 权重预测模型加载成功: {model_path}")
            
        except ImportError:
            print("⚠ onnxruntime未安装，权重预测功能不可用")
            print("  安装方法: pip install onnxruntime")
        except Exception as e:
            print(f"✗ 模型加载失败: {str(e)}")
    
    def predict(self, state):
        """
        预测APF权重系数
        
        :param state: numpy array, shape=(18,)
        :return: dict with weight coefficients
        """
        if not self.model_loaded or self.session is None:
            return None
        
        try:
            # 确保输入shape正确
            if len(state.shape) == 1:
                state = state.reshape(1, -1)
            
            # ONNX推理
            output = self.session.run(
                [self.output_name],
                {self.input_name: state.astype(np.float32)}
            )
            
            # 输出: [α1, α2, α3, α4, α5]
            weights = output[0][0]
            
            return {
                'repulsionCoefficient': float(weights[0]),
                'entropyCoefficient': float(weights[1]),
                'distanceCoefficient': float(weights[2]),
                'leaderRangeCoefficient': float(weights[3]),
                'directionRetentionCoefficient': float(weights[4])
            }
            
        except Exception as e:
            print(f"权重预测失败: {str(e)}")
            return None
    
    def is_available(self):
        """检查预测器是否可用"""
        return self.model_loaded


# 使用示例
if __name__ == "__main__":
    # 创建预测器
    predictor = WeightPredictor("models/weight_predictor_quantized.onnx")
    
    if predictor.is_available():
        # 创建测试状态
        test_state = np.random.randn(18).astype(np.float32)
        
        # 预测权重
        weights = predictor.predict(test_state)
        
        print("预测的APF权重:")
        for key, value in weights.items():
            print(f"  {key}: {value:.2f}")
    else:
        print("预测器不可用")
```

---

### 步骤4: 奖励计算器

创建文件: `multirotor/DQN/reward_calculator.py`

```python
"""
奖励函数计算器
根据无人机行为计算奖励值
"""
import numpy as np


class RewardCalculator:
    """计算强化学习奖励"""
    
    def __init__(self, config):
        self.config = config
        # 奖励权重
        self.w_exploration = 1.0      # 探索奖励
        self.w_efficiency = 0.5       # 效率奖励
        self.w_collision = -5.0       # 碰撞惩罚
        self.w_boundary = -2.0        # 越界惩罚
        self.w_smooth = 0.3           # 平滑运动奖励
        
        # 记录上一步状态
        self.prev_scanned_area = {}
        self.prev_position = {}
    
    def calculate(self, drone_name, server):
        """
        计算当前步的奖励
        
        :return: float - 奖励值
        """
        reward = 0.0
        
        with server.data_lock:
            runtime_data = server.unity_runtime_data[drone_name]
            grid_data = server.grid_data
            
            # 1. 探索奖励（新扫描区域）
            current_scanned = self._count_scanned_cells(grid_data)
            if drone_name in self.prev_scanned_area:
                new_scanned = current_scanned - self.prev_scanned_area[drone_name]
                reward += self.w_exploration * new_scanned
            self.prev_scanned_area[drone_name] = current_scanned
            
            # 2. 碰撞惩罚
            min_distance = self._get_min_distance_to_others(runtime_data)
            if min_distance < self.config.minSafeDistance:
                reward += self.w_collision * (self.config.minSafeDistance - min_distance)
            
            # 3. 越界惩罚
            if runtime_data.leader_position:
                distance_to_leader = (runtime_data.position - runtime_data.leader_position).magnitude()
                if distance_to_leader > runtime_data.leader_scan_radius:
                    reward += self.w_boundary * (distance_to_leader - runtime_data.leader_scan_radius)
            
            # 4. 平滑运动奖励
            if drone_name in self.prev_position:
                movement = (runtime_data.position - self.prev_position[drone_name]).magnitude()
                # 奖励稳定的移动速度
                ideal_movement = self.config.moveSpeed * self.config.updateInterval
                smoothness = 1.0 - abs(movement - ideal_movement) / ideal_movement
                reward += self.w_smooth * max(0, smoothness)
            self.prev_position[drone_name] = runtime_data.position
        
        return reward
    
    def _count_scanned_cells(self, grid_data):
        """统计已扫描单元格数量"""
        if not grid_data or not grid_data.cells:
            return 0
        return sum(1 for cell in grid_data.cells if cell.entropy < 30)
    
    def _get_min_distance_to_others(self, runtime_data):
        """获取到其他无人机的最小距离"""
        if not runtime_data.otherScannerPositions:
            return float('inf')
        
        distances = [
            (runtime_data.position - other_pos).magnitude()
            for other_pos in runtime_data.otherScannerPositions
        ]
        return min(distances) if distances else float('inf')
```

---

### 步骤5: 集成到AlgorithmServer

修改 `multirotor/AlgorithmServer.py`:

```python
class MultiDroneAlgorithmServer:
    def __init__(self, config_file=None, drone_names=None, 
                 use_weight_prediction=False, 
                 collect_training_data=False):
        """
        :param use_weight_prediction: 是否使用模型预测权重
        :param collect_training_data: 是否收集训练数据
        """
        # ... 现有初始化 ...
        
        # DQN V2相关
        self.use_weight_prediction = use_weight_prediction
        self.weight_predictor = None
        
        if use_weight_prediction:
            self._init_weight_predictor()
        
        # 数据收集
        self.collect_training_data = collect_training_data
        self.data_collector = None
        self.reward_calculator = None
        
        if collect_training_data:
            from multirotor.DQN.data_collector import TrajectoryCollector
            from multirotor.DQN.reward_calculator import RewardCalculator
            self.data_collector = TrajectoryCollector()
            self.reward_calculator = RewardCalculator(self.config_data)
            self.data_collector.start_episode()
    
    def _init_weight_predictor(self):
        """初始化权重预测器（ONNX）"""
        try:
            from multirotor.DQN.weight_predictor import WeightPredictor
            
            model_path = os.path.join(
                os.path.dirname(__file__),
                'DQN', 'models', 'weight_predictor.onnx'
            )
            
            self.weight_predictor = WeightPredictor(model_path)
            
            if self.weight_predictor.is_available():
                logger.info("权重预测模式已启用")
            else:
                logger.warning("权重预测模型不可用，使用配置文件权重")
                self.use_weight_prediction = False
                
        except Exception as e:
            logger.error(f"权重预测器初始化失败: {str(e)}")
            self.use_weight_prediction = False
    
    def _get_apf_weights(self, drone_name):
        """获取APF权重（可能来自模型预测）"""
        if self.use_weight_prediction and self.weight_predictor:
            # 提取状态
            from multirotor.DQN.state_extractor import StateExtractor
            state = StateExtractor.extract(self, drone_name)
            
            # 预测权重
            weights = self.weight_predictor.predict(state)
            
            if weights:
                return weights
        
        # 回退到配置文件权重
        return {
            'repulsionCoefficient': self.config_data.repulsionCoefficient,
            'entropyCoefficient': self.config_data.entropyCoefficient,
            'distanceCoefficient': self.config_data.distanceCoefficient,
            'leaderRangeCoefficient': self.config_data.leaderRangeCoefficient,
            'directionRetentionCoefficient': self.config_data.directionRetentionCoefficient
        }
    
    def _process_drone(self, drone_name):
        """无人机处理线程（支持权重预测和数据收集）"""
        while self.running:
            try:
                # ... 现有的数据检查 ...
                
                # 获取当前权重（可能来自模型预测）
                weights = self._get_apf_weights(drone_name)
                
                # 设置到APF算法
                self.algorithms[drone_name].set_coefficients(weights)
                
                # 执行APF算法
                final_dir = self.algorithms[drone_name].update_runtime_data(
                    self.grid_data, self.unity_runtime_data[drone_name]
                )
                
                # 数据收集（如果启用）
                if self.data_collector and self.reward_calculator:
                    from multirotor.DQN.state_extractor import StateExtractor
                    state_dict = StateExtractor.extract(self, drone_name)
                    reward = self.reward_calculator.calculate(drone_name, self)
                    self.data_collector.record_step(drone_name, state_dict, weights, reward)
                
                # 控制无人机移动
                self._control_drone_movement(drone_name, final_dir.finalMoveDir)
                
                # 发送数据到Unity
                self._send_processed_data(drone_name, final_dir)
                
                time.sleep(self.config_data.updateInterval)
                
            except Exception as e:
                logger.error(f"无人机{drone_name}处理出错: {str(e)}")
                time.sleep(self.config_data.updateInterval)
```

---

## 🚀 使用流程

### 阶段A: 数据收集（1-2周）

```bash
# 启用数据收集模式运行
python multirotor/AlgorithmServer.py --collect-data

# 运行多次，收集不同场景的数据
# - 开阔区域
# - 密集障碍物
# - 多无人机
# 等

# 数据会保存到 training/dataset/ 目录
```

### 阶段B: 训练（在GPU环境）

```bash
# 在有GPU的机器上
cd training/
python train_ddpg.py --data-dir dataset/ --epochs 1000

# 训练完成后会生成
# - weight_predictor.pth (PyTorch模型)
# - weight_predictor.onnx (ONNX模型)
# - weight_predictor_quantized.onnx (量化模型)
```

### 阶段C: 部署（目标机器）

```bash
# 1. 复制模型文件
cp weight_predictor_quantized.onnx multirotor/DQN/models/

# 2. 安装ONNX Runtime（轻量级）
pip install onnxruntime

# 3. 启用权重预测模式
python multirotor/AlgorithmServer.py --use-weight-prediction

# 4. 观察效果
# 系统会自动使用模型预测权重，无需手动调参
```

---

## 📊 预期性能

### 推理性能

| 指标 | 目标值 | 说明 |
|------|--------|------|
| **推理延迟** | < 10ms | ONNX INT8量化模型 |
| **CPU占用** | +3-5% | 相比纯APF的增加 |
| **内存占用** | +50MB | ONNX Runtime + 模型 |
| **模型大小** | < 2MB | 量化后 |

### 效果提升

| 指标 | 固定权重 | 学习权重 | 提升 |
|------|----------|----------|------|
| **扫描覆盖率** | 85% | 93% | +8% |
| **扫描时间** | 300s | 270s | -10% |
| **路径效率** | 基线 | +15% | +15% |
| **自适应性** | 低 | 高 | 🌟 |

---

## 🎓 技术要点

### 1. 为什么是5个权重？

对应APF算法的5个权重系数：
- **α1** = repulsionCoefficient (排斥力)
- **α2** = entropyCoefficient (熵)
- **α3** = distanceCoefficient (距离)
- **α4** = leaderRangeCoefficient (Leader范围)
- **α5** = directionRetentionCoefficient (方向保持)

### 2. DDPG网络输出范围

```python
# Actor网络输出
raw_output = self.actor(state)  # 范围: (-∞, +∞)

# 使用Sigmoid缩放到 [0.1, 10.0]
weights = torch.sigmoid(raw_output) * 9.9 + 0.1
```

### 3. 数据收集策略

**多样化采集**:
- 使用不同的固定权重配置运行
- 记录好的和坏的行为
- 包含边界情况

**数据标注**:
- 每步自动计算奖励
- 记录最终任务成功/失败
- 保存完整轨迹

---

## 🛠️ 开发工具

### 训练脚本模板

创建文件: `training/train_ddpg.py`

```python
"""
DDPG训练脚本
"""
import argparse
from stable_baselines3 import DDPG
from weight_learning_env import WeightLearningEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='dataset/', help='训练数据目录')
    parser.add_argument('--epochs', type=int, default=1000, help='训练轮数')
    parser.add_argument('--save-path', default='weight_predictor', help='模型保存路径')
    args = parser.parse_args()
    
    print("=" * 60)
    print("DDPG训练 - APF权重预测")
    print("=" * 60)
    
    # 创建环境
    env = WeightLearningEnv(data_dir=args.data_dir)
    print(f"环境创建成功: state_dim={env.state_dim}, action_dim={env.action_dim}")
    
    # 创建DDPG模型
    model = DDPG(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=100000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        verbose=1,
        tensorboard_log="./logs/"
    )
    
    # 训练
    print(f"\n开始训练 {args.epochs} 轮...")
    model.learn(total_timesteps=args.epochs * 1000)
    
    # 保存PyTorch模型
    model.save(args.save_path)
    print(f"\n模型已保存: {args.save_path}.zip")
    
    # 转换为ONNX
    print("\n转换为ONNX格式...")
    export_to_onnx(model, args.save_path + ".onnx")
    
    print("\n✓ 训练完成！")


if __name__ == "__main__":
    main()
```

---

## 📦 依赖管理

### 训练环境依赖

创建文件: `training/requirements.txt`

```
# 深度学习框架
torch>=1.9.0
stable-baselines3>=1.0
tensorboard>=2.0

# 数据处理
numpy>=1.19.0
pandas>=1.2.0

# 模型转换
onnx>=1.10.0
onnx-simplifier>=0.3.0

# 可视化
matplotlib>=3.3.0
seaborn>=0.11.0
```

### 推理环境依赖

更新 `requirements.txt`:

```
# ... 现有依赖 ...

# DQN权重预测（可选）
onnxruntime>=1.10.0  # 轻量级推理引擎
```

---

## ✅ 检查清单

### 开发前检查
- [ ] 理解V2设计理念
- [ ] 准备GPU训练环境（或使用Colab）
- [ ] 了解DDPG算法原理
- [ ] 熟悉ONNX模型部署

### 数据收集检查
- [ ] 数据收集器实现正确
- [ ] 状态提取完整
- [ ] 奖励计算合理
- [ ] 至少收集100个episode

### 训练检查
- [ ] 训练环境搭建完成
- [ ] DDPG收敛正常
- [ ] 权重输出在有效范围内
- [ ] 训练曲线合理

### 部署检查
- [ ] ONNX模型转换成功
- [ ] 量化模型大小合理
- [ ] CPU推理速度满足要求
- [ ] 集成无报错

---

## 🎯 成功标准

### 最低标准（MVP）
- ✅ 能收集训练数据
- ✅ 能训练出收敛的模型
- ✅ 能转换为ONNX并推理
- ✅ 推理延迟< 50ms

### 理想标准
- ✅ 扫描效率提升> 10%
- ✅ CPU占用增加< 5%
- ✅ 适应不同场景
- ✅ 稳定运行无崩溃

---

**设计完成日期**: 2025-10-13  
**下一步**: 实现数据收集器并开始收集数据

