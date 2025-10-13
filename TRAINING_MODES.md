# DQN训练模式说明

本文档说明两种DQN训练模式的区别和使用场景。

---

## 📋 训练模式对比

| 特性 | 模拟数据训练 | 真实AirSim训练 |
|------|-------------|----------------|
| **数据来源** | 随机生成 | Unity AirSim仿真 |
| **训练速度** | ⚡ 非常快 | 🐌 较慢 |
| **训练质量** | 📊 基础 | 🎯 高质量 |
| **Unity依赖** | ❌ 不需要 | ✅ 必需 |
| **适用阶段** | 快速原型 | 实际部署前 |
| **训练时长** | 3-5分钟 | 30-60分钟 |

---

## 1️⃣ 模拟数据训练

### 📝 描述
使用随机生成的模拟数据进行训练，不需要连接到Unity AirSim。

### 🎯 使用场景
- ✅ 快速测试DQN算法
- ✅ 验证训练流程
- ✅ 调试代码
- ✅ 快速迭代实验

### 📂 相关文件
- 脚本: `multirotor/DQN/train_simple.py`
- 批处理: `train_dqn.bat`
- 环境: `SimpleWeightEnv(server=None)` - 使用模拟数据

### 🚀 使用方法

**方法1: 使用主菜单**
```
start.bat → [3] 训练DQN模型 (模拟数据)
```

**方法2: 直接运行**
```bash
train_dqn.bat
```

**方法3: Python脚本**
```bash
cd multirotor/DQN
python train_simple.py
```

### ⚙️ 训练参数
```python
total_timesteps = 200000  # 20万步
训练时间 ≈ 3-5分钟
```

### ✅ 优点
- 🚀 训练速度极快
- 💻 无需Unity
- 🔧 便于调试
- 📊 适合快速实验

### ⚠️ 缺点
- 🎲 数据不真实
- 📉 效果可能不理想
- ❌ 无法验证实际表现

### 📊 适用情况
```
开发阶段 → 快速原型 → 算法调试
```

---

## 2️⃣ 真实AirSim训练 ⭐

### 📝 描述
连接到Unity AirSim仿真环境，使用真实的传感器数据、无人机状态和环境信息进行训练。

### 🎯 使用场景
- ✅ 实际部署前的训练
- ✅ 获得最佳模型效果
- ✅ 验证算法在真实环境中的表现
- ✅ 最终模型训练

### 📂 相关文件
- 脚本: `multirotor/DQN/train_with_airsim.py`
- 批处理: `train_with_airsim.bat`
- 环境: `SimpleWeightEnv(server=server)` - 连接真实server

### 🚀 使用方法

**前提条件**:
1. ✅ Unity AirSim场景已运行
2. ✅ 场景中有无人机和环境
3. ✅ 网络连接正常

**方法1: 使用主菜单**
```
start.bat → [4] 训练DQN模型 (真实AirSim环境)
```

**方法2: 直接运行**
```bash
# 1. 先启动Unity AirSim
# 2. 然后运行
train_with_airsim.bat
```

**方法3: Python脚本**
```bash
# 1. 先启动Unity
# 2. 运行
cd multirotor/DQN
python train_with_airsim.py
```

### ⚙️ 训练参数
```python
total_timesteps = 100000  # 10万步（可调整）
训练时间 ≈ 30-60分钟（取决于Unity性能）
无人机数量 = 1  # 训练时使用单机
可视化 = True  # 可实时查看训练过程
```

### ✅ 优点
- 🎯 使用真实环境数据
- 📈 模型质量更高
- 🔍 可实时观察训练效果
- ✅ 更接近实际应用

### ⚠️ 缺点
- 🐌 训练速度较慢
- 💻 需要Unity运行
- 🔋 资源消耗较大
- ⏰ 耗时较长

### 📊 适用情况
```
最终训练 → 实际部署 → 性能验证
```

---

## 🔄 推荐工作流程

### 阶段1: 开发调试阶段
```
1. 使用模拟数据训练 (train_dqn.bat)
   ↓
2. 快速验证算法和代码
   ↓
3. 调整奖励配置 (dqn_reward_config.json)
   ↓
4. 重复迭代，直到算法稳定
```

### 阶段2: 测试验证阶段
```
1. 使用AirSim环境训练 (train_with_airsim.bat)
   ↓
2. 使用真实环境数据
   ↓
3. 观察训练过程和可视化
   ↓
4. 测试模型效果 (test_dqn_model.bat)
```

### 阶段3: 最终部署阶段
```
1. 使用最佳模型
   ↓
2. 在真实环境中运行 (run_with_dqn.bat)
   ↓
3. 对比固定权重和DQN权重效果
   ↓
4. 根据需要重新训练优化
```

---

## 📊 训练数据对比

### 模拟数据训练
```python
# 环境数据：随机生成
position = np.random.randn(3)
velocity = np.random.randn(3)
entropy = np.random.uniform(0, 100)
# ... 其他随机数据

# 奖励：基于简单规则
reward = 0  # 固定或简单计算
```

### 真实AirSim训练
```python
# 环境数据：来自Unity仿真
position = runtime_data.position  # 真实位置
velocity = runtime_data.velocity  # 真实速度
entropy = grid_data.cells[i].entropy  # 真实熵值

# 奖励：基于实际扫描效果
reward = new_scanned_cells * exploration_reward
reward -= collision_penalty  # 真实碰撞检测
reward -= out_of_range_penalty  # 真实越界检测
```

---

## 🎯 如何选择训练模式

### 使用模拟数据训练，如果：
- ⏰ 时间紧迫，需要快速结果
- 🔧 正在调试算法逻辑
- 💻 Unity未安装或不可用
- 📊 只是测试训练流程

### 使用真实AirSim训练，如果：
- 🎯 需要最佳模型效果
- ✅ Unity环境已准备好
- ⏰ 有充足的训练时间
- 🚀 准备实际部署使用

---

## 🔧 训练参数调整

### 模拟数据训练参数
文件: `multirotor/DQN/train_simple.py`

```python
# 快速训练
total_timesteps = 50000

# 标准训练
total_timesteps = 200000

# 深度训练
total_timesteps = 500000
```

### 真实AirSim训练参数
文件: `multirotor/DQN/train_with_airsim.py`

```python
# 快速测试
TOTAL_TIMESTEPS = 10000

# 标准训练（推荐）
TOTAL_TIMESTEPS = 100000

# 深度训练
TOTAL_TIMESTEPS = 200000

# 无人机数量
DRONE_NAMES = ["UAV1"]  # 训练时建议用1个

# 可视化
USE_VISUALIZATION = True  # 可观察训练过程
```

---

## 💡 训练技巧

### 1. 分阶段训练
```
阶段1: 模拟数据 (50K步) → 验证算法
阶段2: 真实环境 (100K步) → 精细调优
阶段3: 合并模型 → 最佳效果
```

### 2. 检查点管理
```
真实AirSim训练会自动保存检查点：
- checkpoint_5000.zip
- checkpoint_10000.zip
- best_model.zip (最佳模型)
- weight_predictor_airsim.zip (最终模型)
```

### 3. 训练监控
```python
# 观察这些指标
ep_rew_mean  # 平均奖励（应逐渐增加）
ep_len_mean  # episode长度
critic_loss  # 评论家损失（应逐渐减小）
```

### 4. 奖励调整
修改 `dqn_reward_config.json`:
```json
{
    "reward_coefficients": {
        "exploration_reward": 3.0,  // ↑ 更激进
        "collision_penalty": 5.0,
        "out_of_range_penalty": 7.0
    }
}
```

---

## 🐛 常见问题

### Q1: 训练时奖励一直是0？
**原因**: 奖励配置不合理或模型还在探索

**解决**:
1. 增加 `exploration_reward`
2. 降低惩罚系数
3. 继续训练，观察是否上升

---

### Q2: 真实训练时Unity崩溃？
**原因**: 资源不足或Unity bug

**解决**:
1. 关闭其他程序
2. 降低Unity画质设置
3. 减少训练步数
4. 使用检查点继续训练

---

### Q3: 模型效果不好？
**原因**: 训练不充分或参数不当

**解决**:
1. 增加训练步数
2. 调整奖励配置
3. 使用真实环境重新训练
4. 对比不同模型效果

---

### Q4: 训练速度太慢？
**真实AirSim训练**:
- 关闭可视化: `USE_VISUALIZATION = False`
- 降低Unity帧率
- 减少无人机数量

**模拟数据训练**:
- 已经是最快的了

---

## 📈 性能对比

| 指标 | 模拟数据 | 真实AirSim |
|------|---------|-----------|
| 训练速度 (步/秒) | ~1000 | ~30-50 |
| 200K步耗时 | 3-5分钟 | 60-120分钟 |
| 模型质量 | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| 资源消耗 | 低 | 高 |
| 实际效果 | 一般 | 优秀 |

---

## 🎓 建议

**初学者**:
```
1. 从模拟数据训练开始
2. 理解训练流程
3. 再尝试真实环境训练
```

**进阶用户**:
```
1. 直接使用真实环境训练
2. 调整高级参数
3. 优化奖励函数
```

**生产环境**:
```
1. 必须使用真实环境训练
2. 充分的训练步数
3. 多次训练取最佳模型
```

---

**最后更新**: 2025-10-13

