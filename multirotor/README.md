# AirSim Python Algorithm Server

这是一个基于AirSim的Python算法服务器，用于控制无人机进行自动扫描任务。

## 项目结构

- **AlgorithmServer.py** - 主服务器文件，融合了算法和AirSim控制功能
- **AirsimServer/** - 包含与AirSim交互的组件
  - **drone_controller.py** - 无人机控制器，提供连接、起飞、降落等功能
- **Algorithm/** - 包含算法相关的代码和配置
  - **HexGridDataModel.py** - 蜂窝网格数据模型
  - **Vector3.py** - 三维向量类
  - **scanner_algorithm.py** - 扫描算法实现
  - **scanner_config.json** - 算法配置文件
  - **scanner_config_data.py** - 配置数据类
  - **scanner_runtime_data.py** - 运行时数据类
- **setup_path.py** - 设置Python路径的工具文件

## 功能概述

AlgorithmServer是一个融合了算法和AirSim控制的统一服务，它直接调用AirSim API进行通信，不再使用Socket通信。主要功能包括：

- 连接到AirSim模拟器
- 控制无人机起飞、降落、移动等
- 更新网格数据
- 运行扫描算法
- 获取和保存扫描结果
- 加载和重置配置

## 使用方法

1. 确保已安装AirSim和必要的Python依赖
2. 启动AirSim模拟器
3. 运行AlgorithmServer：
   ```
   python AlgorithmServer.py
   ```

## 配置

可以在**Algorithm/scanner_config.json**文件中修改算法配置参数。

## 注意事项

- AlgorithmServer需要在AirSim模拟器运行的情况下才能正常工作
- 如果连接失败，请检查AirSim是否正在运行，以及配置是否正确