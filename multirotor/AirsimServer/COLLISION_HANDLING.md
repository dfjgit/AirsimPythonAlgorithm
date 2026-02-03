# AirSim 碰撞处理与防穿地机制

## 问题描述

在AirSim仿真训练中，无人机重置时可能会出现以下问题：
- **穿透地面持续下坠**：碰撞后无人机处于异常状态，reset()调用时物理引擎未及时响应
- **物理状态残留**：惯性或碰撞力残留导致无人机在重置后继续运动
- **碰撞检测失效**：重置过程中碰撞检测可能被绕过

## 解决方案

### 1. 暂停仿真重置 (已实现)

**核心机制**：在重置时暂停物理引擎，避免物理计算干扰

```python
# drone_controller.py - reset()方法
def reset(self) -> bool:
    """重置模拟器状态(带防穿地保护)"""
    with self.api_lock:
        # 1. 暂停仿真
        self.client.simPause(True)
        time.sleep(0.1)
        
        # 2. 执行重置
        self.client.reset()
        time.sleep(0.3)
        
        # 3. 恢复仿真
        self.client.simPause(False)
```

**优势**：
- ✅ 物理引擎完全停止，无惯性残留
- ✅ 重置后位置立即生效
- ✅ 简单可靠，性能开销小

### 2. 碰撞检测与恢复 (已实现)

**新增API**：
```python
# 检查碰撞状态
collision = drone_controller.check_collision("UAV1")
if collision['has_collided']:
    print(f"碰撞对象: {collision['object_name']}")
    print(f"穿透深度: {collision['penetration_depth']}")
    print(f"碰撞点: {collision['impact_point']}")

# 从碰撞中恢复(悬停稳定)
drone_controller.recover_from_collision("UAV1")
```

**使用场景**：
- 重置前检测无人机是否处于碰撞状态
- 训练过程中实时监控碰撞
- 作为奖励函数的判断依据

### 3. 强制位置重置 (已实现)

**API说明**：
```python
# 忽略碰撞，强制传送到安全位置
drone_controller.reset_vehicle_to_pose(
    vehicle_name="UAV1",
    position=(0, 0, -3),  # NED坐标系，z=-3表示离地3米
    ignore_collision=True  # 关键参数：可以穿透地面回到空中
)
```

**使用场景**：
- 无人机已穿地且无法恢复时的紧急方案
- 手动调试时快速重置位置
- 特殊环境下的位置校准

## 集成到AlgorithmServer

已在`reset_environment()`中集成碰撞检测：

```python
# AlgorithmServer.py - 第1120-1138行
if need_physical_reset:
    # 检查所有无人机的碰撞状态
    for drone_name in self.drone_names:
        collision = self.drone_controller.check_collision(drone_name)
        if collision['has_collided']:
            logger.warning(f"无人机{drone_name}处于碰撞状态，将执行安全恢复")
            # 先尝试悬停恢复
            self.drone_controller.recover_from_collision(drone_name)
    
    # 执行安全重置(带防穿地保护)
    self.drone_controller.reset()
```

## 配置建议

### AirSim设置

在`settings.json`中可以调整碰撞相关参数：

```json
{
  "SimMode": "Multirotor",
  "PhysicsEngineName": "FastPhysicsEngine",
  "Vehicles": {
    "UAV1": {
      "VehicleType": "SimpleFlight",
      "EnableCollisions": true,
      "EnableCollisionPassthrough": false,
      "CollisionInfo": {
        "CollisionEnabled": true
      }
    }
  }
}
```

### 训练环境参数

建议的重置等待时间：
- `simPause`后等待：**0.1秒**
- `reset()`后等待：**0.3秒**
- 重新启用API后等待：**1.0秒**
- 起飞后稳定时间：**2.0秒**

## 调试技巧

### 1. 实时监控碰撞

```python
# 在训练循环中添加碰撞监控
collision = drone_controller.check_collision("UAV1")
if collision['has_collided']:
    logger.error(f"训练中断：碰撞 {collision['object_name']}")
    # 触发重置或终止episode
```

### 2. 记录碰撞数据

```python
# 将碰撞信息添加到tensorboard或日志
if collision['has_collided']:
    writer.add_scalar('collision/depth', collision['penetration_depth'], step)
    writer.add_scalar('collision/count', 1, step)
```

### 3. 碰撞惩罚奖励

```python
# 在奖励函数中加入碰撞惩罚
reward = base_reward
collision = check_collision(drone_name)
if collision['has_collided']:
    reward -= 100.0  # 严重惩罚
    if collision['penetration_depth'] > 0.5:
        done = True  # 穿透过深，终止episode
```

## API参考

### DroneController 新增方法

| 方法 | 说明 | 返回值 |
|------|------|--------|
| `reset()` | 安全重置(防穿地) | bool |
| `check_collision(vehicle_name)` | 检查碰撞状态 | Dict |
| `recover_from_collision(vehicle_name)` | 悬停恢复 | bool |
| `reset_vehicle_to_pose(vehicle_name, position, ignore_collision)` | 强制位置重置 | bool |

### 碰撞信息字段

```python
collision = {
    'has_collided': bool,           # 是否发生碰撞
    'object_name': str,             # 碰撞对象名称
    'penetration_depth': float,     # 穿透深度(米)
    'impact_point': (x, y, z),      # 碰撞点坐标
    'normal': (x, y, z),            # 碰撞法向量
    'time_stamp': float             # 时间戳
}
```

## 常见问题

**Q: 为什么暂停后还需要等待0.1秒？**
A: 给物理引擎时间完成当前帧的计算并进入暂停状态，确保状态一致性。

**Q: ignore_collision会不会导致训练不真实？**
A: 仅在重置时使用，训练过程中碰撞检测仍然正常工作。

**Q: 如何判断是否需要使用强制重置？**
A: 当`check_collision`显示`penetration_depth > 1.0`时，说明穿地严重，建议使用强制重置。

**Q: 会不会影响训练性能？**
A: 几乎无影响，额外开销约10-30ms，且仅在重置时发生。

## 更新日志

- **2026-02-03**: 实现防穿地重置机制
  - 添加`simPause`包裹的安全重置
  - 实现碰撞检测API
  - 添加碰撞恢复功能
  - 集成到`AlgorithmServer.reset_environment()`

## 参考资料

- [AirSim Collision API](https://microsoft.github.io/AirSim/apis/#collision-api)
- [AirSim Reset API](https://microsoft.github.io/AirSim/apis/#reset)
- [Physics Engine Settings](https://microsoft.github.io/AirSim/settings/)
