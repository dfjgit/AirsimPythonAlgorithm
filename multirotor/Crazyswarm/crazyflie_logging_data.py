import json
from typing import Union, List, Dict
from dataclasses import dataclass, asdict


@dataclass
class CrazyflieLoggingData:
    """无人机状态数据类（对应C#的CrazyflieLoggingData）"""
    # 字段顺序与C#保持一致，类型标注匹配（C# double → Python float）
    Id: int  # 无人机ID
    X: float  # X轴位置坐标
    Y: float  # Y轴位置坐标
    Z: float  # Z轴位置坐标
    Time: float  # 时间戳（秒）
    Qx: float  # 四元数x分量
    Qy: float  # 四元数y分量
    Qz: float  # 四元数z分量
    Qw: float  # 四元数w分量（实部）
    Speed: float  # 移动速度
    XSpeed: float  # X轴方向速度
    YSpeed: float  # Y轴方向速度
    ZSpeed: float  # Z轴方向速度
    AcceleratedSpeed: float  # 合加速度
    XAcceleratedSpeed: float  # X轴加速度
    YAcceleratedSpeed: float  # Y轴加速度
    ZAcceleratedSpeed: float  # Z轴加速度
    XEulerAngle: float  # X轴欧拉角（度）
    YEulerAngle: float  # Y轴欧拉角（度）
    ZEulerAngle: float  # Z轴欧拉角（度）
    XPalstance: float  # X轴角速度
    YPalstance: float  # Y轴角速度
    ZPalstance: float  # Z轴角速度
    XAccfPalstance: float  # X轴角加速度
    YAccfPalstance: float  # Y轴角加速度
    ZAccfPalstance: float  # Z轴角加速度
    Battery: float  # 电池电压（V）

    # ------------------- 序列化：对象 → 字典（适配JSON键名） -------------------
    def to_dict(self) -> dict:
        """
        转换为JSON格式的字典（键名与C#的JsonProperty一致，全小写）
        例如：Id → "id"，X → "x"
        """
        # 先通过asdict获取数据类的原始字典（键名是大驼峰，如Id）
        raw_dict = asdict(self)
        # 转换键名为小写（匹配C#的JsonProperty）
        json_dict = {k.lower(): v for k, v in raw_dict.items()}
        return json_dict
    
    @classmethod
    def from_json_to_dicts(cls, json_str: str) -> Union[Dict, List[Dict]]:
        """
        接收JSON字符串（单个对象/数组），转换为数据类的字段字典（大驼峰键名）
        - 若JSON是单个对象 → 返回单个字典
        - 若JSON是数组 → 返回字典列表
        """
        # 1. 解析JSON字符串，得到原始Python对象（可能是dict或list）
        raw_json_obj = json.loads(json_str)

        # 2. 定义“单个字典转大驼峰”的内部函数
        def _convert_to_camel_case(raw_dict: Dict) -> Dict:
            return {
                k[:1].upper() + k[1:]: v  # 小写键转大驼峰：id→Id, x→X
                for k, v in raw_dict.items()
            }

        # 3. 分情况处理：单个对象/数组
        if isinstance(raw_json_obj, list):
            # 情况1：JSON是数组 → 遍历每个元素，转成大驼峰字典的列表
            return [_convert_to_camel_case(item) for item in raw_json_obj]
        elif isinstance(raw_json_obj, dict):
            # 情况2：JSON是单个对象 → 转成大驼峰字典
            return _convert_to_camel_case(raw_json_obj)
        else:
            raise ValueError(f"不支持的JSON类型：{type(raw_json_obj)}，仅支持对象/数组")

    # ------------------- 反序列化：字典 → 对象 -------------------
    @classmethod
    def from_dict(cls, data_dict: dict) -> "CrazyflieLoggingData":
        """
        从JSON字典反序列化为对象（自动适配小写键名→大驼峰字段名）
        """
        # 转换键名为大驼峰（匹配数据类字段名）
        class_dict = {}
        for k, v in data_dict.items():
            # 小写键转大驼峰：如 "id" → "Id"，"xspeed" → "XSpeed"
            camel_case_key = k[:1].upper() + k[1:]
            class_dict[camel_case_key] = v
        # 创建并返回数据类实例
        return cls(**class_dict)

    # ------------------- 快捷方法：直接处理JSON字符串 -------------------
    def to_json(self, indent: int = 4) -> str:
        """对象 → JSON字符串（带格式化）"""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "CrazyflieLoggingData":
        """JSON字符串 → 对象"""
        data_dict = json.loads(json_str)
        return cls.from_dict(data_dict)
    
    @classmethod
    def from_dict_list(cls, dict_list: List[dict]) -> List["CrazyflieLoggingData"]:
        """将大驼峰字典列表转换为CrazyflieLoggingData实例列表"""
        return [cls(**log_dict) for log_dict in dict_list]