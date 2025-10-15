# 🛠️ DQN无人机移动控制 - 安装指南

## 📋 系统要求

- **Python版本**: 3.7 - 3.10 (推荐 3.8 或 3.9)
- **操作系统**: Windows, Linux, macOS
- **内存**: 至少4GB RAM
- **存储**: 至少2GB可用空间
- **GPU**: 可选，有GPU会快很多

## 🚀 快速安装

### 步骤1: 检查Python版本

```bash
python --version
```

应该显示 Python 3.7 或更高版本。

### 步骤2: 安装依赖

#### 方法A: 使用requirements文件（推荐）

```bash
# 进入DQN目录
cd AirsimAlgorithmPython/multirotor/DQN

# 安装依赖
pip install -r requirements_movement.txt
```

#### 方法B: 手动安装

```bash
pip install torch stable-baselines3 numpy gym tensorboard
```

#### 方法C: 使用国内镜像（中国用户推荐）

```bash
pip install -r requirements_movement.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 步骤3: 验证安装

运行测试脚本验证环境：

```bash
python movement_env.py
```

如果看到类似下面的输出，说明安装成功：

```
============================================================
测试 MovementEnv - 无人机移动DQN环境
============================================================

观察空间: Box(21,)
动作空间: Discrete(6)
动作映射:
  0: 上 -> [ 0.  0.  1.]
  1: 下 -> [ 0.  0. -1.]
  ...

[OK] 环境测试通过！
```

## 📦 依赖说明

### 核心依赖

| 包名 | 版本 | 用途 |
|-----|------|------|
| torch | >=1.9.0 | 深度学习框架 |
| stable-baselines3 | >=1.5.0 | 强化学习算法库 |
| numpy | >=1.19.0 | 数值计算 |
| gym | >=0.21.0 | 强化学习环境接口 |

### 可选依赖

| 包名 | 版本 | 用途 |
|-----|------|------|
| tensorboard | >=2.8.0 | 训练可视化 |

## 🔧 常见安装问题

### Q1: torch安装失败

**原因**: PyTorch需要根据系统选择合适版本

**解决方法**:

访问 [PyTorch官网](https://pytorch.org/get-started/locally/) 选择合适版本

```bash
# CPU版本（Windows）
pip install torch torchvision torchaudio

# GPU版本（需要CUDA）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Q2: stable-baselines3安装失败

**解决方法**:

```bash
# 更新pip
python -m pip install --upgrade pip

# 再次尝试
pip install stable-baselines3
```

### Q3: 提示"ModuleNotFoundError"

**原因**: 某个依赖包未安装

**解决方法**:

```bash
# 查看错误信息中缺少的包名，手动安装
pip install <包名>
```

### Q4: 安装速度很慢（中国大陆用户）

**解决方法**: 使用国内镜像源

```bash
# 清华镜像（推荐）
pip install -r requirements_movement.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 或永久配置
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q5: 提示Python版本不兼容

**解决方法**:

```bash
# 安装指定版本的Python（推荐3.9）
# 或使用conda创建虚拟环境
conda create -n dqn_env python=3.9
conda activate dqn_env
pip install -r requirements_movement.txt
```

## 🎯 GPU支持（可选）

如果你有NVIDIA GPU，可以安装GPU版本的PyTorch来加速训练：

### 步骤1: 检查CUDA版本

```bash
nvidia-smi
```

查看CUDA Version（如：CUDA 11.8）

### 步骤2: 安装对应版本的PyTorch

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 步骤3: 验证GPU可用

```python
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
```

应该显示 `CUDA可用: True`

## 🐍 虚拟环境（推荐）

使用虚拟环境可以避免依赖冲突：

### 使用venv

```bash
# 创建虚拟环境
python -m venv dqn_venv

# 激活虚拟环境
# Windows
dqn_venv\Scripts\activate
# Linux/Mac
source dqn_venv/bin/activate

# 安装依赖
pip install -r requirements_movement.txt
```

### 使用conda

```bash
# 创建环境
conda create -n dqn_env python=3.9

# 激活环境
conda activate dqn_env

# 安装依赖
pip install -r requirements_movement.txt
```

## ✅ 完整安装验证清单

运行以下命令，确保所有组件都正常：

```bash
# 1. 检查Python版本
python --version  # 应该是 3.7+

# 2. 检查torch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# 3. 检查stable-baselines3
python -c "import stable_baselines3; print('SB3: OK')"

# 4. 检查numpy
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"

# 5. 检查gym
python -c "import gym; print(f'Gym: {gym.__version__}')"

# 6. 测试环境
python multirotor/DQN/movement_env.py
```

如果所有命令都正常执行，说明安装成功！

## 🎓 下一步

安装完成后，您可以：

1. **快速开始**: 阅读 `README_MOVEMENT.md`
2. **详细文档**: 阅读 `MOVEMENT_DQN.md`
3. **开始训练**: 运行 `train_movement_dqn.bat`

## 📞 获取帮助

如果遇到问题：

1. 查看错误信息
2. 检查Python版本和依赖版本
3. 查看本文档的常见问题部分
4. 搜索相关错误信息

## 🎉 祝您使用愉快！

安装成功后就可以开始训练您的第一个DQN无人机模型了！

---

**最后更新**: 2024-10-14  
**文档版本**: v1.0.0

