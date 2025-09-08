# Unity与Python算法集成指南

本文档介绍如何将AutoScanner的算法逻辑从Unity转移到外部Python脚本中执行。

## 实现原理

整个集成方案基于文件交换机制：
1. Unity定期将AutoScanner的参数序列化并保存到JSON文件
2. Python脚本读取JSON文件，处理数据，计算新的移动方向
3. Python将处理结果保存到另一个JSON文件
4. Unity读取结果文件并应用新的参数

## 文件结构

### Unity端文件
- **ScannerDataSerializer.cs** - 负责序列化和反序列化AutoScanner的参数
- **AutoScanner.cs** - 已修改，添加了与Python通信的功能

### Python端文件
- **StreamingAssets/scanner_algorithm.py** - Python算法实现示例

## 使用方法

### 1. 在Unity中配置

1. 确保场景中的AutoScanner组件已添加必要的引用（hexGrid和leader）

2. 在Inspector面板中找到"外部Python通信配置"部分：
   - 勾选 **useExternalPythonProcessing** 启用外部处理
   - 设置 **pythonInputFilePath** 和 **pythonOutputFilePath**（默认路径已设置好）
   - 调整 **pythonProcessInterval** 控制数据交换频率

### 2. 运行Python算法

1. 确保已安装Python环境

2. 运行算法脚本：
   ```bash
   cd d:\Project\UnityProject\Airsim2022\Assets\StreamingAssets
   python scanner_algorithm.py
   ```

   注意：您可能需要根据实际情况调整路径。

### 3. 运行Unity场景

启动Unity场景，AutoScanner会自动开始与Python脚本通信。

## Python算法实现指南

### 数据结构

Python脚本中的 `ScannerData` 类对应Unity中的AutoScanner参数。您可以访问以下关键属性：

- **系数设置**：`repulsionCoefficient`, `entropyCoefficient`, `distanceCoefficient`, `leaderRangeCoefficient`, `directionRetentionCoefficient`
- **基础参数**：`moveSpeed`, `rotationSpeed`, `scanRadius`
- **位置和方向**：`position`, `forward`, `finalMoveDir`等
- **Leader信息**：`leaderPosition`, `leaderScanRadius`
- **已访问蜂窝**：`visitedCells`列表

### 实现自定义算法

要实现自定义算法，请修改`ScannerAlgorithm`类的`process`方法：

1. 根据输入的`scanner_data`计算新的方向向量
2. 更新`scanner_data`对象的属性
3. 返回处理后的`scanner_data`对象

示例算法框架：
```python
def process(self, data):
    # 计算权重
    repulsion_weight, entropy_weight, distance_weight, leader_range_weight, direction_retention_weight = self.calculate_weights(data)
    
    # TODO: 实现您的自定义算法逻辑
    # 计算新的各方向向量
    new_score_dir = self.calculate_score_direction(data)
    new_path_dir = self.calculate_path_direction(data)
    new_collide_dir = self.calculate_repulsion_direction(data)
    new_leader_range_dir = self.calculate_leader_range_direction(data)
    new_direction_retention_dir = self.calculate_direction_retention_direction(data)
    
    # 合并所有向量
    new_final_move_dir = (
        new_score_dir * entropy_weight + 
        new_path_dir * distance_weight + 
        new_collide_dir * repulsion_weight +
        new_leader_range_dir * leader_range_weight +
        new_direction_retention_dir * direction_retention_weight
    )
    
    # 归一化最终方向
    if new_final_move_dir.magnitude() > 0.1:
        new_final_move_dir = new_final_move_dir.normalized()
    else:
        new_final_move_dir = data.forward
    
    # 更新数据对象
    data.finalMoveDir = new_final_move_dir
    
    return data
```

## 调试提示

1. **检查文件路径**：确保Unity和Python使用相同的文件路径

2. **日志输出**：Unity会在Console窗口输出调试信息，Python脚本也有打印输出

3. **检查中间文件**：您可以直接打开JSON文件查看序列化的数据内容

4. **逐步测试**：先从简单的算法开始，验证通信正常后再实现复杂逻辑

## 注意事项

1. 文件交换机制可能会有一定的延迟，根据需要调整`pythonProcessInterval`

2. 确保Python脚本有足够的权限读写文件

3. 当多个扫描器同时运行时，您可能需要修改方案以支持多实例

4. 对于更复杂的应用场景，考虑使用套接字(Socket)或其他更高效的通信方式替代文件交换