# AirSim DroneServer

这是一个基于AirSim的无人机控制服务器实现，提供Socket接口用于远程控制AirSim模拟器中的无人机。

## 项目结构

```
├── airsim/             # AirSim Python客户端库
├── multirotor/         # 无人机相关代码
│   ├── DroneServer.py  # 主服务器文件
│   ├── components/     # 拆分的组件模块
│   ├── server_api.md   # API文档
│   └── setup_path.py   # 路径设置
├── requirements.txt    # 项目依赖项
├── setup.py            # 项目安装配置
└── install_offline.bat # 离线安装脚本
```

## 离线安装说明

### 前提条件
- 已安装Python 3.x
- 如果需要全新环境，先下载所有依赖包到dependencies目录

### 使用方法

#### 步骤1: 准备依赖包（可选）
如果要在完全离线的环境中使用，首先需要在有网络的环境中下载所有依赖包：

```bash
# 在有网络的环境中运行
pip download -r requirements.txt -d dependencies
```

#### 步骤2: 执行离线安装
在目标机器上（可以是离线环境），运行以下命令或双击install_offline.bat文件：

```bash
install_offline.bat
```

这将安装所有依赖项并配置项目。

## 运行DroneServer

安装完成后，可以通过以下命令启动DroneServer：

```bash
cd multirotor
python DroneServer.py
```

## API文档
请参考multirotor目录下的server_api.md文件了解详细的API使用方法。

## 依赖项
项目依赖以下Python包（版本号详见requirements.txt）：
- backports.ssl_match_hostname
- msgpack-python
- msgpack-rpc-python
- numpy
- opencv-python
- tornado

## 注意事项
- 确保已正确安装AirSim模拟器
- 在运行DroneServer之前，建议先启动AirSim模拟器
- 如需修改服务器配置，请编辑DroneServer.py文件中的相关参数