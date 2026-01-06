from enum import Enum, IntEnum
from dataclasses import dataclass, asdict
from typing import Any, Dict

class EnumCrazyflieOperate(IntEnum):
    """无人机操作枚举（与C#的EnumDroneOperate完全对应）"""
    TakeOff = 1          # 起飞
    Land = 2             # 降落
    GoTo = 3             # 前往目标点
    UploadTra = 4        # 上传轨迹
    StartTra = 5         # 启动轨迹
    StartLogging = 6     # 开始日志记录
    GetLogging = 7       # 获取日志
    StopLogging = 8      # 停止日志记录
    Stop = 9             # 停止操作
    VelWorld = 10        # 世界坐标系速度控制


@dataclass
class CrazyflieOperate:
    """无人机操作数据类（对应C#的CrazyflieOperate）"""
    id: int                  # 无人机ID
    operate: EnumCrazyflieOperate # 操作类型（枚举）
    operateData: Any = None  # 操作附加数据（对应C#的object）

    # ------------------- 对应C#的Create静态方法 -------------------
    @classmethod
    def create(cls, id: int, enum_operate: EnumCrazyflieOperate, operate_data: Any = None) -> "CrazyflieOperate":
        """创建CrazyflieOperate实例（与C#的Create方法功能一致）"""
        return cls(
            id=id,
            operate=enum_operate,
            operateData=operate_data
        )

    # ------------------- 序列化：对象 → 字典（适配JSON） -------------------
    def to_dict(self) -> Dict:
        """
        转换为JSON兼容的字典：
        - 枚举`operate`转成对应的int值
        - `operateData`保留原始类型（需保证其本身可JSON序列化）
        """
        raw_dict = asdict(self)
        # 枚举转int值
        raw_dict["operate"] = self.operate.value
        return raw_dict

    # ------------------- 反序列化：字典 → 对象 -------------------
    @classmethod
    def from_dict(cls, data_dict: Dict) -> "CrazyflieOperate":
        """
        从字典反序列化为对象：
        - 从int值恢复`operate`枚举
        - `operateData`直接解析为对应类型
        """
        # int值转枚举
        operate_enum = EnumCrazyflieOperate(data_dict["operate"])
        return cls(
            id=data_dict["id"],
            operate=operate_enum,
            operateData=data_dict.get("operateData")  # 兼容无operateData的情况
        )

    # ------------------- 快捷方法：直接处理JSON字符串 -------------------
    def to_json(self) -> str:
        """对象 → JSON字符串"""
        import json
        return json.dumps(self.to_dict())
    

    @classmethod
    def from_json(cls, json_str: str) -> "CrazyflieOperate":
        """JSON字符串 → 对象"""
        import json
        data_dict = json.loads(json_str)
        return cls.from_dict(data_dict)