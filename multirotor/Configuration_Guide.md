# 配置文件参数说明

本文档描述了 `scanner_config.json` 配置文件中各个参数的作用。

## 人工势场算法参数

### 权重系数
- **repulsionCoefficient** (2.0): 排斥力权重，控制无人机间避障强度
- **entropyCoefficient** (2.0): 熵权重，控制探索未知区域的倾向
- **distanceCoefficient** (2.0): 距离权重，控制向目标移动的倾向
- **leaderRangeCoefficient** (2.0): Leader范围权重，控制保持在Leader范围内的倾向
- **directionRetentionCoefficient** (2.0): 方向保持权重，控制飞行稳定性

### 基础参数
- **updateInterval** (1): 算法更新间隔，单位：秒
- **moveSpeed** (2.0): 无人机移动速度，单位：米/秒
- **rotationSpeed** (120.0): 无人机旋转速度，单位：度/秒
- **scanRadius** (1.0): 扫描半径，单位：米
- **altitude** (2.0): 飞行高度，单位：米

### 安全参数
- **maxRepulsionDistance** (5.0): 最大排斥距离，单位：米
- **minSafeDistance** (1.0): 最小安全距离，单位：米

### 目标选择策略
- **avoidRevisits** (true): 是否避免重复访问已扫描区域
- **targetSearchRange** (20.0): 目标搜索范围，单位：米
- **revisitCooldown** (10.0): 重复访问冷却时间，单位：秒

## 系统参数

- **name** ("ScannerConfigData"): 配置名称标识
- **hideFlags** (0): 隐藏标志（系统内部使用）

## 使用建议

1. **首次使用**: 建议使用默认参数值
2. **性能调优**: 根据实际环境调整权重系数
3. **安全设置**: 根据无人机尺寸调整安全距离参数
4. **环境适应**: 根据扫描区域大小调整搜索范围参数
5. **速度调整**: 根据场景复杂度调整移动速度和更新间隔

## 参数调优建议

### 场景类型

#### 开阔区域
- 提高 `moveSpeed` (3.0-5.0)
- 增大 `scanRadius` (5.0-8.0)
- 提高 `updateInterval` (0.5-1.0)

#### 密集障碍物
- 降低 `moveSpeed` (1.0-2.0)
- 减小 `scanRadius` (2.0-4.0)
- 降低 `updateInterval` (0.2-0.5)
- 提高 `repulsionCoefficient` (3.0-5.0)

#### 多无人机协同
- 提高 `minSafeDistance` (2.0-3.0)
- 提高 `repulsionCoefficient` (3.0-4.0)
- 平衡各权重系数
