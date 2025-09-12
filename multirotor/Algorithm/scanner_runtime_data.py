from typing import List, Dict, Any
from .Vector3 import Vector3


class ScannerRuntimeData:
    """扫描器实时数据类，对应C#中的ScannerRuntimeData"""
    uavname: str
    # 方向向量（Python提供,Unity绘制）
    scoreDir: Vector3
    collideDir: Vector3
    pathDir: Vector3
    leaderRangeDir: Vector3
    directionRetentionDir: Vector3
    finalMoveDir: Vector3

    # 当前位置和方向信息（Unity提供）
    position: Vector3
    forward: Vector3

    # Leader信息（Unity提供）
    leader_position: Vector3
    leader_scan_radius: float
    leader_velocity: Vector3  # 添加leader_velocity属性

    # 已访问蜂窝记录和其它扫描者坐标（Unity提供）
    visited_cells: List[Vector3]
    otherScannerPositions: List[Vector3]

    def __init__(self, direction=None, position=None, velocity=None, 
                 leader_position=None, leader_velocity=None, visited_cells=None):
        self.uavname = ""
        # 初始化向量默认值
        self.position = position if position is not None else Vector3()
        self.forward = Vector3(0, 0, 1)
        self.scoreDir = Vector3()
        self.collideDir = Vector3()
        self.pathDir = Vector3()
        self.leaderRangeDir = Vector3()
        self.directionRetentionDir = Vector3()
        self.finalMoveDir = Vector3()
        self.direction = direction if direction is not None else Vector3(0, 0, 1)  # 初始化direction
        self.velocity = velocity if velocity is not None else Vector3()  # 初始化velocity

        # 初始化领导者信息
        self.leader_position = leader_position if leader_position is not None else Vector3()
        self.leader_scan_radius = 0.0
        self.leader_velocity = leader_velocity if leader_velocity is not None else Vector3()  # 初始化leader_velocity

        # 初始化列表
        self.visited_cells = visited_cells.copy() if visited_cells is not None else []
        self.otherScannerPositions = []

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典以便序列化"""
        return {
            'uavname': self.uavname,
            # 方向向量
            'scoreDir': self.scoreDir.to_dict(),
            'collideDir': self.collideDir.to_dict(),
            'pathDir': self.pathDir.to_dict(),
            'leaderRangeDir': self.leaderRangeDir.to_dict(),
            'directionRetentionDir': self.directionRetentionDir.to_dict(),
            'finalMoveDir': self.finalMoveDir.to_dict(),

            # 当前位置和方向
            'position': self.position.to_dict(),
            'forward': self.forward.to_dict(),

            # Leader信息
            'leaderPosition': self.leader_position.to_dict(),
            'leaderScanRadius': self.leader_scan_radius,

            # 已访问记录和其它扫描者坐标
            'visitedCells': [cell.to_dict() for cell in self.visited_cells],
            'otherScannerPositions': [pos.to_dict() for pos in self.otherScannerPositions]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """从字典创建ScannerRuntimeData实例"""
        instance = cls()
        
        # 解析向量数据
        vector_keys = [
            ('uavname', 'uavname'),
            ('position', 'position'),
            ('forward', 'forward'),
            ('scoreDir', 'scoreDir'),
            ('collideDir', 'collideDir'),
            ('pathDir', 'pathDir'),
            ('leaderRangeDir', 'leaderRangeDir'),
            ('directionRetentionDir', 'directionRetentionDir'),
            ('finalMoveDir', 'finalMoveDir'),
            ('leaderPosition', 'leaderPosition')
        ]
        for attr_name, json_key in vector_keys:
            data_dict = data.get(json_key, {})
            if isinstance(data_dict, dict):
                setattr(instance, attr_name, Vector3.from_dict(data_dict))

        # 解析领导者扫描半径
        if 'leaderScanRadius' in data:
            instance.leader_scan_radius = data['leaderScanRadius']

        # 解析已访问蜂窝
        if 'visitedCells' in data and isinstance(data['visitedCells'], list):
            instance.visited_cells = [
                Vector3.from_dict(cell_data)
                for cell_data in data['visitedCells']
                if isinstance(cell_data, dict)
            ]

        # 解析其它扫描者坐标
        if 'otherScannerPositions' in data and isinstance(data['otherScannerPositions'], list):
            instance.otherScannerPositions = [
                Vector3.from_dict(pos_data) 
                for pos_data in data['otherScannerPositions']
                if isinstance(pos_data, dict)
            ]

        return instance

    def add_visited_cell(self, cell_position: Vector3) -> None:
        """添加已访问的蜂窝位置（去重）"""
        if not any(
            cell.x == cell_position.x and
            cell.y == cell_position.y and
            cell.z == cell_position.z
            for cell in self.visited_cells
        ):
            self.visited_cells.append(cell_position)

    def clear_visited_cells(self) -> None:
        """清空已访问的蜂窝记录"""
        self.visited_cells.clear()

    def get_visited_cell_count(self) -> int:
        """获取已访问蜂窝单元数量"""
        return len(self.visited_cells)

    def get_visited_cells(self) -> List[Vector3]:
        """获取已访问的蜂窝单元列表"""
        return self.visited_cells.copy()

    def update_move_direction(self, new_direction: Vector3) -> None:
        """更新移动方向（自动归一化）"""
        if new_direction.magnitude() > 0.001:
            self.finalMoveDir = new_direction.normalized()

    @property
    def leader_distance(self) -> float:
        """计算与领导者的距离"""
        return (self.position - self.leader_position).magnitude()

    def is_leader_within_range(self, scanRadius: float) -> bool:
        """检查领导者是否在扫描范围内"""
        return self.leader_distance <= scanRadius

    def copy(self):
        """创建对象的深拷贝"""
        new_data = ScannerRuntimeData()
        # 深拷贝向量对象
        vector_attrs = [attr for attr in dir(self) if isinstance(getattr(self, attr), Vector3)]
        for attr in vector_attrs:
            original_vec = getattr(self, attr)
            setattr(new_data, attr, Vector3(original_vec.x, original_vec.y, original_vec.z))
        # 拷贝其他属性
        new_data.leader_scan_radius = self.leader_scan_radius
        # 深拷贝列表
        new_data.visited_cells = [Vector3(cell.x, cell.y, cell.z) for cell in self.visited_cells]
        new_data.otherScannerPositions = [Vector3(pos.x, pos.y, pos.z) for pos in self.otherScannerPositions]
        return new_data
    
    def __repr__(self) -> str:
        return f"RUNTIME_DATA Start:{self.position}To:{self.finalMoveDir}"