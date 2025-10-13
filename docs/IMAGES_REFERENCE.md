# 项目图片资源说明

本文档说明项目中各个图片的内容和用途。

---

## 📊 架构设计图

### 1. system_architecture_with_dqn.png
**原文件名**: img.png  
**类型**: 系统整体架构图  
**内容**:
- AirSim仿真环境
- Unity操作界面
- Python Server
- CrazySwarm Server
- ROS空间
- MotionCapture
- Crazyflie机器固件
- DQN训练模块（环境数据域划分等）

**用途**:
- 展示整个系统的完整架构
- 说明各组件之间的数据流
- 展示DQN如何集成到系统中

**相关文档**: 
- `multirotor/DQN/README.md` - DQN集成设计
- `multirotor/DQN/README_V1_ARCHIVED.md` - V1架构记录

---

## 🧠 DQN相关设计图

### 2. dqn_architecture_detailed.png
**原文件名**: img_2.png  
**类型**: DQN详细架构图  
**内容**:
- DQN训练模块（预测模型）
  - 动作空间
  - 环境信息（空间区域、熵值、无人机状态）
  - DQN Agent（调用APF算法、调用AirSim API执行行为、奖励函数）
- 训练后调用流程
  - 状态空间
  - pth/ONNX（量化子模型）
  - 下一步动作
- APF执行流程
  - 状态空间
  - APF (α1, α2, α3)
  - 下一步动作

**用途**:
- 详细说明DQN训练和推理架构
- 展示从训练到部署的流程
- 对比DQN模式和纯APF模式

**相关文档**:
- `multirotor/DQN/README_V1_ARCHIVED.md` - V1实现详情
- `multirotor/DQN/V2_REQUIREMENTS.md` - V2设计参考

---

### 3. dqn_workflow_simple.png
**原文件名**: img_1.png  
**类型**: DQN简化工作流程图  
**内容**:
- 强化学习环境准备
- 智能体与模型配置阶段
- 训练循环阶段
- 环境交互与决策执行流程
  - 状态观测获取
  - 动作选择与执行
  - 奖励计算

**用途**:
- 展示DQN工作流程的核心步骤
- 说明训练循环的基本流程
- 适合快速理解DQN工作原理

**相关文档**:
- `multirotor/DQN/README.md` - DQN基础概念
- `multirotor/DQN/CPU_OPTIMIZATION.md` - 性能优化

---

### 4. dqn_training_workflow.png
**原文件名**: img_3.png  
**类型**: DQN训练工作流程图（三模式对比）  
**内容**:
- DQN训练流程
  - 动作空间（由APF计算得出）
  - 状态空间（空间区域信息、无人机信息）
  - Agent根据状态空间、动作空间调用APF算法、获得下一步动作（固定α1, α2, α3）
  - 奖励函数（对当前动作进行评分）
- 训练后调用流程
  - 状态空间
  - pth/ONNX（量化子模型）
  - 下一步动作
  - 循环训练
- APF执行流程
  - 状态空间
  - APF (α1, α2, α3)
  - 下一步动作

**用途**:
- 对比三种工作模式
- 说明DQN如何与APF结合
- 展示训练到部署的完整流程

**相关文档**:
- `multirotor/DQN/V2_REQUIREMENTS.md` - V2方案选择参考

---

### 5. airsim_dqn_workflow.png
**原文件名**: img_4.png  
**类型**: AirSIM-DQN工作流程图  
**内容**:
- 强化学习环境准备
- 智能体与模型配置阶段
- 训练循环阶段
- 环境交互与决策执行流程
  - 状态观测获取
  - 动作选择与执行
  - 奖励计算
- 标题: AirSIM-DQN工作流

**用途**:
- 特定于AirSim环境的DQN工作流
- 展示与仿真器的交互流程
- 说明环境反馈机制

**相关文档**:
- `multirotor/DQN/README_V1_ARCHIVED.md` - AirSim集成经验

---

## 📝 图片使用建议

### 文档引用

在编写文档时，可以使用以下方式引用图片：

**从项目根目录引用**：
```markdown
![系统架构](docs/images/system_architecture_with_dqn.png)
*图1: 完整系统架构，展示DQN如何集成到无人机控制系统*

![DQN架构](docs/images/dqn_architecture_detailed.png)
*图2: DQN详细架构，包含训练和推理流程*
```

**从DQN目录引用**：
```markdown
![系统架构](../../docs/images/system_architecture_with_dqn.png)
```

### 适用场景

| 图片 | 适用于 |
|------|--------|
| system_architecture_with_dqn.png | 项目总览、系统设计文档 |
| dqn_architecture_detailed.png | DQN技术文档、设计说明 |
| dqn_workflow_simple.png | 快速入门、概念介绍 |
| dqn_training_workflow.png | 训练流程说明、模式对比 |
| airsim_dqn_workflow.png | AirSim集成文档 |

---

## 🔄 版本说明

这些图片记录了DQN V1版本的设计思路。虽然V1因性能问题已归档，但这些架构图对于理解系统设计和规划V2版本仍有重要参考价值。

**V1状态**: 🔴 已归档  
**图片用途**: 📚 设计参考、经验总结

---

## 📂 文件组织

所有图片已组织在专门的文档目录中：

```
docs/
├── images/
│   ├── system_architecture_with_dqn.png
│   ├── dqn_architecture_detailed.png
│   ├── dqn_workflow_simple.png
│   ├── dqn_training_workflow.png
│   └── airsim_dqn_workflow.png
└── IMAGES_REFERENCE.md (本文档)
```

---

**创建日期**: 2025-10-13  
**更新日期**: 2025-10-13

