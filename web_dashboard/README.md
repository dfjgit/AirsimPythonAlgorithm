# Web 控制台使用指南

## 简介

这是无人机数据采集系统的 Web 控制台，基于 Node.js + Socket.io 实现，提供实时数据监控、图表可视化和远程控制功能。

## 功能特性

- ✅ **实时数据监控**：覆盖率、未侦察数量、已侦察数量等
- ✅ **实时图表**：使用 Chart.js 绘制覆盖率趋势曲线
- ✅ **远程控制**：通过浏览器控制数据采集的开始/停止
- ✅ **文件管理**：查看和下载历史数据文件
- ✅ **跨平台访问**：支持在局域网内的任何设备（手机、平板、电脑）访问

## 安装步骤

### 1. 安装 Node.js 依赖

```bash
cd web_dashboard
npm install
```

### 2. 安装 Python 依赖

在项目根目录下：

```bash
pip install python-socketio[client]
```

或者安装完整的 requirements.txt：

```bash
pip install -r requirements.txt
```

## 使用方法

### 第一步：启动 Web 服务器

打开一个终端，运行：

```bash
cd web_dashboard
npm start
```

或者：

```bash
node server.js
```

你会看到：

```
Web控制台运行在: http://localhost:3000
数据目录: D:\Project\AirsimProject\AirsimAlgorithmPython\multirotor\data_logs
```

### 第二步：打开浏览器

访问 `http://localhost:3000`

此时你会看到一个现代化的仪表盘，右上角显示灰色的"Python离线"。

### 第三步：启动 Python 算法服务

在另一个终端运行你的 Python 算法服务（例如 `python multirotor/AlgorithmServer.py`）。

- Python 启动后会自动连接到 Node.js 服务器
- 网页右上角会瞬间变成绿色的 **"Python在线"**

### 第四步：开始使用

1. 点击 **"开始采集"** 按钮
2. 观察实时数据更新和覆盖率趋势图
3. 点击 **"停止并保存"** 完成数据采集
4. 在文件列表中点击文件名即可下载 CSV 文件

## 架构说明

```
┌─────────────┐      Socket.io      ┌──────────────┐      Socket.io      ┌─────────────┐
│   Python    │ ◄─────────────────► │   Node.js    │ ◄─────────────────► │   浏览器     │
│ 算法服务器  │                     │  Web 服务器  │                     │  控制台      │
└─────────────┘                     └──────────────┘                     └─────────────┘
     │                                    │
     │                                    │
     └──────────► data_logs/ ◄───────────┘
              (CSV 文件存储)
```

### 数据流向

1. **Python → Node.js → 浏览器**：
   - Python 算法服务器通过 `WebBridge` 发送实时遥测数据
   - Node.js 服务器接收并广播给所有连接的浏览器
   - 浏览器实时更新界面和图表

2. **浏览器 → Node.js → Python**：
   - 用户在浏览器点击"开始采集"或"停止采集"
   - Node.js 服务器接收命令并转发给 Python
   - Python 执行相应的操作

## 故障排除

### 问题1：无法连接到 Web 服务器

**症状**：Python 启动后显示 "无法连接到 Web 控制台"

**解决方案**：
1. 确保 Node.js 服务器正在运行（`npm start`）
2. 检查端口 3000 是否被占用
3. 确认防火墙没有阻止连接

### 问题2：网页显示 "Python离线"

**症状**：浏览器右上角一直显示灰色状态

**解决方案**：
1. 确认 Python 算法服务器已启动
2. 检查 Python 控制台是否有连接错误
3. 确认 `python-socketio` 已正确安装

### 问题3：图表不更新

**症状**：数据数值在更新，但图表不动

**解决方案**：
1. 确保已点击"开始采集"按钮
2. 检查浏览器控制台是否有 JavaScript 错误
3. 刷新页面重试

## 高级配置

### 修改端口

编辑 `server.js`：

```javascript
const PORT = 3000;  // 改为你想要的端口
```

### 修改数据目录

编辑 `server.js`：

```javascript
const DATA_LOGS_DIR = path.join(__dirname, '../multirotor/data_logs');
// 改为你的数据目录路径
```

### 局域网访问

默认情况下，Web 服务器只监听 localhost。要允许局域网访问：

1. 找到你的本机 IP 地址（例如：192.168.1.100）
2. 在局域网内的其他设备上访问：`http://192.168.1.100:3000`

## 技术栈

- **后端**：Node.js + Express + Socket.io
- **前端**：HTML5 + Bootstrap 5 + Chart.js
- **Python 通信**：python-socketio

## 许可证

MIT License

