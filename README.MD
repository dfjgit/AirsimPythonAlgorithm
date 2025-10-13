# AirSim Algorithm Python

基于AirSim的多无人机协同扫描算法实现（Python版本）

---

## 📋 项目简介

本项目实现了一个多无人机协同扫描系统，使用**人工势场（APF）算法**进行路径规划和区域探索。系统集成了AirSim仿真环境和Unity可视化界面，支持实时数据交互和算法可视化。

### 核心功能
- ✅ 多无人机协同控制
- ✅ 人工势场算法（APF）
- ✅ 基于熵值的探索策略
- ✅ Leader-Follower模式
- ✅ 实时可视化
- ✅ Unity-AirSim双向通信

---

## 🚀 快速开始

### 环境要求
- Python 3.7+
- AirSim 仿真器
- Unity (可选，用于3D可视化)

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行程序
```bash
python multirotor/AlgorithmServer.py
```

---

## 📊 系统架构

![系统架构](docs/images/system_architecture_with_dqn.png)

### 主要组件

#### 1. AlgorithmServer
- 核心算法服务器
- 负责无人机控制和算法执行
- 处理AirSim和Unity的数据交互

#### 2. APF算法
- 人工势场算法实现
- 多因素权重计算：
  - 排斥力（避障）
  - 熵值（探索）
  - 距离（导航）
  - Leader范围（保持编队）
  - 方向保持（稳定飞行）

#### 3. 可视化组件
- 实时2D可视化
- 熵值颜色渐变显示（绿→黄→红）
- 无人机位置和运动方向
- Leader位置和扫描范围

---

## 🎨 可视化说明

### 熵值颜色系统
- 🟢 **绿色 (0-30)**: 区域已充分扫描
- 🟡 **黄色 (30-70)**: 区域部分扫描
- 🔴 **红色 (70-100)**: 区域未扫描

### 显示元素
- **绿色圆圈**: 无人机
- **淡蓝色圆圈**: Leader
- **白色箭头**: 移动方向
- **绿色圆环**: 扫描范围
- **彩色点**: 网格单元（按熵值着色）

---

## ⚙️ 配置说明

### 配置文件
主配置文件: `multirotor/scanner_config.json`

### 核心参数
```json
{
    "repulsionCoefficient": 2.0,      // 排斥力权重
    "entropyCoefficient": 2.0,        // 熵权重
    "distanceCoefficient": 2.0,       // 距离权重
    "leaderRangeCoefficient": 2.0,    // Leader范围权重
    "directionRetentionCoefficient": 2.0,  // 方向保持权重
    "updateInterval": 1,               // 更新间隔（秒）
    "moveSpeed": 2.0,                  // 移动速度（米/秒）
    "scanRadius": 3.0,                 // 扫描半径（米）
    "altitude": 2.0                    // 飞行高度（米）
}
```

详细配置说明: [Configuration_Guide.md](multirotor/Configuration_Guide.md)

---

## 📁 项目结构

```
AirsimAlgorithmPython/
├── multirotor/                      # 核心代码目录
│   ├── Algorithm/                   # 算法实现
│   │   ├── scanner_algorithm.py    # APF算法核心
│   │   ├── simple_visualizer.py    # 可视化组件
│   │   ├── scanner_config_data.py  # 配置数据类
│   │   └── ...
│   ├── DQN/                         # DQN相关（已归档）
│   │   ├── README.md               # DQN设计文档
│   │   ├── README_V1_ARCHIVED.md   # V1归档
│   │   ├── V2_REQUIREMENTS.md      # V2需求模板
│   │   └── CPU_OPTIMIZATION.md     # 性能优化指南
│   ├── AlgorithmServer.py          # 主服务器
│   ├── scanner_config.json         # 配置文件
│   └── Configuration_Guide.md      # 配置指南
├── docs/                            # 文档目录
│   ├── images/                     # 图片资源
│   ├── IMAGES_REFERENCE.md         # 图片说明
│   └── README.md                   # 文档索引
├── requirements.txt                 # Python依赖
└── README.MD                        # 本文档
```

---

## 🎓 算法说明

### 人工势场算法（APF）

APF算法通过组合多个"势场力"来计算无人机的最终移动方向：

1. **排斥力**: 避免与其他无人机碰撞
2. **熵引力**: 吸引无人机探索高熵值（未知）区域
3. **距离引力**: 引导无人机向目标区域移动
4. **Leader范围力**: 保持无人机在Leader扫描范围内
5. **方向保持力**: 维持飞行方向的稳定性

最终方向 = 各力的加权和

### 熵值计算

熵值表示区域的不确定性：
- 高熵值：区域未被扫描，信息不确定
- 低熵值：区域已被扫描，信息确定

---

## 🔧 开发指南

### 添加新的无人机
修改 `AlgorithmServer.py` 中的无人机列表：
```python
drone_names = ["UAV1", "UAV2", "UAV3"]  # 添加更多无人机
```

### 调整算法参数
修改 `scanner_config.json` 中的权重系数

### 自定义可视化
修改 `Algorithm/simple_visualizer.py`

---

## 🐛 故障排查

### 程序启动卡住
- 检查AirSim是否正在运行
- 确认Unity客户端连接状态
- 查看日志输出定位问题

### 可视化不显示
- 确认pygame已正确安装
- 检查是否有图形界面环境
- 查看控制台错误信息

### 性能问题
- 降低 `updateInterval`
- 减少同时运行的无人机数量
- 关闭不必要的可视化

---

## 📚 文档资源

- [配置指南](multirotor/Configuration_Guide.md) - 详细的参数配置说明
- [图片说明](docs/IMAGES_REFERENCE.md) - 架构图和流程图说明
- [DQN文档](multirotor/DQN/README.md) - DQN相关设计（已归档）

---

## ⚠️ DQN状态说明

**DQN V1版本已归档**：由于在无独显环境下存在性能问题，DQN功能已暂时移除。

- 📋 V1详细记录: [docs/DQN/README_V1_ARCHIVED.md](docs/DQN/README_V1_ARCHIVED.md)
- 🔄 V2需求文档: [docs/DQN/V2_REQUIREMENTS.md](docs/DQN/V2_REQUIREMENTS.md)
- 🎯 V2设计文档: [docs/DQN/V2_DESIGN.md](docs/DQN/V2_DESIGN.md)
- 🚀 快速开始: [docs/DQN/QUICK_START.md](docs/DQN/QUICK_START.md)

当前系统使用**纯APF算法**，性能稳定流畅。

---

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

### 代码规范
- 遵循PEP 8编码规范
- 添加适当的注释和文档
- 提交前进行测试

---

## 📄 许可证

*[添加许可证信息]*

---

## 📞 联系方式

*[添加联系方式]*

---

**最后更新**: 2025-10-13

