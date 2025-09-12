from enum import Enum
from typing import List, Any

# 数据包类型枚举
class PackType(Enum):
    grid_data = "grid_data"    # HexGridDataModel，地图数据只解析第一个
    config_data = "config_data"  # ScannerConfigData
    runtime_data = "runtime_data" # ScannerRuntimeData

# 数据包数据结构（通信发送的数据包）
class DataPacks:
    def __init__(self):
        self.type: PackType = None  # 数据包类型
        self.time_span: str = ""    # 时间跨度字符串
        self.pack_data_list: List[Any] = []  # 数据包列表，对应C#中的List<IDataPack>
        
    def __repr__(self) -> str:
        return (f"DataPacks(type={self.type}, time_span={self.time_span}, "
                f"pack_data_list_count={len(self.pack_data_list)})")
