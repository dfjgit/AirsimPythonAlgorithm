# 项目文档目录

本目录包含项目的所有文档和图片资源。

---

## 📚 文档结构

```
docs/
├── images/                          # 图片资源目录
│   ├── system_architecture_with_dqn.png      # 系统整体架构图
│   ├── dqn_architecture_detailed.png         # DQN详细架构图
│   ├── dqn_workflow_simple.png               # DQN简化工作流程
│   ├── dqn_training_workflow.png             # DQN训练工作流程
│   └── airsim_dqn_workflow.png               # AirSim-DQN工作流程
├── IMAGES_REFERENCE.md              # 图片说明文档
└── README.md                        # 本文档
```

---

## 📖 核心文档索引

### 配置文档
- [Configuration_Guide.md](../multirotor/Configuration_Guide.md) - 配置文件参数说明

### DQN相关文档
- [DQN主文档](./DQN/README.md) - DQN集成设计文档
- [快速开始](./DQN/QUICK_START.md) - 快速上手指南
- [V2设计](./DQN/V2_DESIGN.md) - V2详细设计
- [实现指南](./DQN/IMPLEMENTATION_GUIDE.md) - 实现指南
- [V2需求](./DQN/V2_REQUIREMENTS.md) - V2版本需求规划
- [V1归档](./DQN/README_V1_ARCHIVED.md) - V1版本完整记录
- [CPU优化](./DQN/CPU_OPTIMIZATION.md) - DQN性能优化

### 图片资源
- [IMAGES_REFERENCE.md](./IMAGES_REFERENCE.md) - 所有图片的详细说明

---

## 🎯 快速导航

### 新用户
1. 阅读项目README: [../README.MD](../README.MD)
2. 查看配置指南: [Configuration_Guide.md](../multirotor/Configuration_Guide.md)
3. 了解系统架构: 查看 [system_architecture_with_dqn.png](./images/system_architecture_with_dqn.png)

### 开发者
1. 查看代码文档: `multirotor/` 目录
2. 理解APF算法: `multirotor/Algorithm/` 目录
3. DQN开发参考: `multirotor/DQN/` 目录

### V2开发
1. 阅读V1归档: [README_V1_ARCHIVED.md](../multirotor/DQN/README_V1_ARCHIVED.md)
2. 填写V2需求: [V2_REQUIREMENTS.md](../multirotor/DQN/V2_REQUIREMENTS.md)
3. 参考架构图: [dqn_architecture_detailed.png](./images/dqn_architecture_detailed.png)

---

## 📊 图片资源快览

### 系统架构
![系统架构](./images/system_architecture_with_dqn.png)
*完整系统架构，包含AirSim、Unity、DQN等组件*

### DQN架构
![DQN架构](./images/dqn_architecture_detailed.png)
*DQN详细架构，展示训练、推理、APF流程*

### DQN工作流
![DQN简化流程](./images/dqn_workflow_simple.png)
*DQN核心工作流程*

---

## 🔄 文档维护

### 添加新文档
1. 将Markdown文档放在 `docs/` 根目录
2. 将图片放在 `docs/images/` 目录
3. 更新本README的索引

### 图片命名规范
格式: `{主题}_{类型}_{描述}.png`

示例:
- `system_architecture_overview.png` - 系统架构总览
- `algorithm_flowchart_apf.png` - APF算法流程图
- `ui_screenshot_main.png` - 主界面截图

---

## 📞 相关链接

- **项目主页**: [../README.MD](../README.MD)
- **源代码**: `../multirotor/`
- **配置文件**: `../multirotor/scanner_config.json`
- **问题追踪**: *[添加Issue链接]*

---

**创建日期**: 2025-10-13  
**维护者**: 项目团队

