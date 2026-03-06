# 熵值累积修复说明

## 📋 问题描述

### 原始问题
训练过程中发现"信息熵值越来越少"的异常现象：
- 每个Episode的扫描格子数（scanned_cells）都是0或接近0
- 扫描进度无法累积
- 训练效率极低

### 根本原因
在 `AlgorithmServer.reset_environment()` 中，每次Episode结束都会调用 `grid_data.reset_entropy()`，将所有格子的熵值重置为100（未扫描状态）。这导致：
- ✅ Episode 0扫描了18个格子
- ❌ Episode 1-164都显示0个扫描格子（因为每次都重置了）

---

## ✅ 修复方案

### 1. 修改 `AlgorithmServer.py`
添加 `reset_grid` 参数到 `reset_environment()` 方法：

```python
def reset_environment(self, reason: str = "Unknown", reset_grid: bool = False) -> None:
    """重置运行环境

    Args:
        reason: 重置原因
        reset_grid: 是否重置网格熵值（默认False，保持扫描进度累积）
    """
    # ... 原有代码 ...

    # 根据参数决定是否重置网格熵值
    if reset_grid:
        self.grid_data.reset_entropy()
        logger.info("[重置] 网格熵值已重置为100（完全重新扫描）")
    else:
        logger.info("[重置] 保持网格熵值（扫描进度累积）")
```

### 2. 修改 `SimpleWeightEnv`
添加 `reset_grid_entropy` 初始化参数：

```python
def __init__(
    self,
    server=None,
    drone_name="UAV1",
    reward_config_path=None,
    reset_unity=True,
    reset_grid_entropy=False,  # 新增参数
    step_duration=5.0,
    safety_limit=True,
    max_weight_delta=0.5,
):
```

在调用 `reset_environment()` 时传递参数：
```python
self.server.reset_environment(reason=reason, reset_grid=self.reset_grid_entropy)
```

### 3. 更新训练脚本
在 `train_with_airsim_improved.py` 中：
- 添加配置文件支持
- 默认使用 `reset_grid_entropy=False`（累积模式）
- 添加日志输出显示当前模式

---

## 🚀 使用方法

### 默认模式（推荐）
```json
{
  "reset_grid_entropy": false
}
```
- ✅ 扫描进度跨Episode累积
- ✅ 适合长期训练
- ✅ 学习完整扫描策略

### 重置模式
```json
{
  "reset_grid_entropy": true
}
```
- 每个Episode重新扫描
- 适合独立任务训练

### 命令行使用
```bash
# 使用默认模式（累积扫描进度）
python train_with_airsim_improved.py

# 使用配置文件
python train_with_airsim_improved.py --config configs/training_config_with_entropy_fix.json
```

---

## 📊 验证方法

运行验证脚本：
```bash
python verify_entropy_fix.py
```

预期输出：
```
✅ AlgorithmServer: 通过
✅ SimpleWeightEnv: 通过
✅ 训练脚本: 通过
🎉 所有测试通过！
```

---

## 🔍 训练效果对比

### 修复前
| Episode | 扫描格子数 | 扫描比例 | 说明 |
|---------|----------|---------|------|
| 0 | 18 | 2.25% | 首次扫描 |
| 1 | 0 | 0.00% | ❌ 重置后丢失 |
| 2-164 | ~0 | ~0.00% | ❌ 每次都重置 |

### 修复后（预期）
| Episode | 扫描格子数 | 扫描比例 | 说明 |
|---------|----------|---------|------|
| 0 | 18 | 2.25% | 首次扫描 |
| 1 | 35 | 4.31% | ✅ 累积增长 |
| 2 | 52 | 6.41% | ✅ 持续增长 |
| ... | ... | ... | ✅ 持续累积 |

---

## 📝 修改文件清单

1. ✅ `AlgorithmServer.py`
   - 修改 `reset_environment()` 方法签名
   - 添加 `reset_grid` 参数（默认False）
   - 根据参数决定是否重置熵值

2. ✅ `envs/simple_weight_env.py`
   - 添加 `reset_grid_entropy` 初始化参数
   - 传递参数到 `reset_environment()`

3. ✅ `train_with_airsim_improved.py`
   - 添加配置文件支持
   - 默认使用累积模式
   - 添加日志输出

4. ✅ `verify_entropy_fix.py` (新增)
   - 自动化验证脚本

5. ✅ `configs/training_config_with_entropy_fix.json` (新增)
   - 配置文件示例

---

## ⚠️ 注意事项

1. **向后兼容**
   - 默认值设为 `False`，保持扫描进度累积
   - 如果需要旧的行为，设置 `reset_grid_entropy=True`

2. **重启Unity**
   - 如果Unity已经运行，建议重启以清除缓存状态

3. **训练数据**
   - 修复前的训练数据无法重新利用（熵值已重置）
   - 建议重新开始训练

---

## 🎯 预期改进

修复后，训练应该能看到：
1. ✅ 扫描格子数逐Episode增加
2. ✅ 扫描比例持续上升
3. ✅ 更高的训练效率
4. ✅ 更好的长期策略学习

---

## 📞 问题反馈

如果遇到问题，请检查：
1. 运行 `verify_entropy_fix.py` 确认修复已正确应用
2. 检查配置文件中的 `reset_grid_entropy` 设置
3. 查看训练日志中的模式提示
4. 确认Unity已重启

---

**修复日期**: 2026-03-05
**修复人**: Claude Code
**版本**: v1.0
