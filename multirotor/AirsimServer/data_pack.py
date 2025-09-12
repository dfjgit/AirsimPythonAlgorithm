from enum import Enum
from typing import List, Any, Union


# 数据包类型枚举
class PackType(Enum):
    grid_data = "grid_data"  # HexGridDataModel，地图数据
    config_data = "config_data"  # ScannerConfigData
    runtime_data = "runtime_data"  # ScannerRuntimeData


# 数据包数据结构（通信发送的数据包）
class DataPacks:
    def __init__(self):
        self.type: PackType = None  # 数据包类型
        self.time_span: str = ""  # 时间戳字符串
        # 支持列表（runtime_data）或字典（grid_data/config_data）
        self.pack_data_list: Union[List[Any], dict] = []

    def __repr__(self) -> str:
        return (f"DataPacks(type={self.type}, time_span={self.time_span}, "
                f"pack_data_type={type(self.pack_data_list).__name__})")