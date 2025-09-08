import Vector3

from Vector3 import Vector3
import json
from typing import Optional, List

class ScannerData:
    def __init__(self, json_data=None):
        # 初始化默认值（系数）
        self.repulsionCoefficient = 2.0
        self.entropyCoefficient = 3.0
        self.distanceCoefficient = 2.0
        self.leaderRangeCoefficient = 3.0
        self.directionRetentionCoefficient = 2.0
        self.updateInterval = 0.2
        # 初始化默认值（参数）
        self.moveSpeed = 2.0
        self.rotationSpeed = 120.0
        self.scanRadius = 5.0
# 初始化默认值（参数）
        self.maxRepulsionDistance = 5.0
        self.minSafeDistance = 2.0
# 初始化默认值（参数）
        self.avoidRevisits = True
        self.targetSearchRange = 20.0
        self.revisitCooldown = 60.0

        # 初始化Vector3类型的数据
        self.position = Vector3()
        self.forward = Vector3(0, 0, 1)
        self.scoreDir = Vector3()
        self.collideDir = Vector3()
        self.pathDir = Vector3()
        self.leaderRangeDir = Vector3()
        self.directionRetentionDir = Vector3()
        self.finalMoveDir = Vector3()

        # 初始化Leader信息
        self.leaderPosition = Vector3()
        self.leaderScanRadius = 0.0

        # 初始化已访问蜂窝记录
        self.visitedCells = []

        # 如果提供了json_data，则解析数据
        if json_data is not None:
            self.parse_json_data(json_data)

    def parse_json_data(self, json_data):
        """解析JSON数据到对象属性"""
        # 验证输入数据类型
        if not isinstance(json_data, dict):
            try:
                json_data = json.loads(json_data)
            except (json.JSONDecodeError, TypeError):
                raise ValueError("Invalid JSON data format")

        # 解析系数和配置参数
        self.repulsionCoefficient = self._get_float(json_data, 'repulsionCoefficient', 2.0)
        self.entropyCoefficient = self._get_float(json_data, 'entropyCoefficient', 3.0)
        self.distanceCoefficient = self._get_float(json_data, 'distanceCoefficient', 2.0)
        self.leaderRangeCoefficient = self._get_float(json_data, 'leaderRangeCoefficient', 3.0)
        self.directionRetentionCoefficient = self._get_float(json_data, 'directionRetentionCoefficient', 2.0)
        self.updateInterval = self._get_float(json_data, 'updateInterval', 0.2)

        self.moveSpeed = self._get_float(json_data, 'moveSpeed', 2.0)
        self.rotationSpeed = self._get_float(json_data, 'rotationSpeed', 120.0)
        self.scanRadius = self._get_float(json_data, 'scanRadius', 5.0)

        self.maxRepulsionDistance = self._get_float(json_data, 'maxRepulsionDistance', 5.0)
        self.minSafeDistance = self._get_float(json_data, 'minSafeDistance', 2.0)

        self.avoidRevisits = json_data.get('avoidRevisits', True)
        self.targetSearchRange = self._get_float(json_data, 'targetSearchRange', 20.0)
        self.revisitCooldown = self._get_float(json_data, 'revisitCooldown', 60.0)

        # 解析Vector3类型的数据
        pos_data = json_data.get('position', {})
        self.position = Vector3(pos_data.get('x', 0), pos_data.get('y', 0), pos_data.get('z', 0))

        forward_data = json_data.get('forward', {})
        self.forward = Vector3(forward_data.get('x', 0), forward_data.get('y', 0), forward_data.get('z', 1))

        score_dir_data = json_data.get('scoreDir', {})
        self.scoreDir = Vector3(score_dir_data.get('x', 0), score_dir_data.get('y', 0), score_dir_data.get('z', 0))

        collide_dir_data = json_data.get('collideDir', {})
        self.collideDir = Vector3(collide_dir_data.get('x', 0), collide_dir_data.get('y', 0),
                                  collide_dir_data.get('z', 0))

        path_dir_data = json_data.get('pathDir', {})
        self.pathDir = Vector3(path_dir_data.get('x', 0), path_dir_data.get('y', 0), path_dir_data.get('z', 0))

        leader_range_dir_data = json_data.get('leaderRangeDir', {})
        self.leaderRangeDir = Vector3(leader_range_dir_data.get('x', 0), leader_range_dir_data.get('y', 0),
                                      leader_range_dir_data.get('z', 0))

        direction_retention_dir_data = json_data.get('directionRetentionDir', {})
        self.directionRetentionDir = Vector3(direction_retention_dir_data.get('x', 0),
                                             direction_retention_dir_data.get('y', 0),
                                             direction_retention_dir_data.get('z', 0))

        final_move_dir_data = json_data.get('finalMoveDir', {})
        self.finalMoveDir = Vector3(final_move_dir_data.get('x', 0), final_move_dir_data.get('y', 0),
                                    final_move_dir_data.get('z', 0))

        # 解析Leader信息
        leader_pos_data = json_data.get('leaderPosition', {})
        self.leaderPosition = Vector3(leader_pos_data.get('x', 0), leader_pos_data.get('y', 0),
                                      leader_pos_data.get('z', 0))
        self.leaderScanRadius = self._get_float(json_data, 'leaderScanRadius', 0.0)

        # 解析已访问蜂窝记录
        self.visitedCells = []
        for cell_data in json_data.get('visitedCells', []):
            self.visitedCells.append(Vector3(cell_data.get('x', 0), cell_data.get('y', 0), cell_data.get('z', 0)))

    def _get_float(self, data_dict: dict, key: str, default: float) -> float:
        """安全地从字典中获取浮点数"""
        value = data_dict.get(key, default)
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def to_dict(self) -> dict:
        """将对象转换回字典格式以便序列化"""
        result = {
            'repulsionCoefficient': self.repulsionCoefficient,
            'entropyCoefficient': self.entropyCoefficient,
            'distanceCoefficient': self.distanceCoefficient,
            'leaderRangeCoefficient': self.leaderRangeCoefficient,
            'directionRetentionCoefficient': self.directionRetentionCoefficient,
            'updateInterval': self.updateInterval,

            'moveSpeed': self.moveSpeed,
            'rotationSpeed': self.rotationSpeed,
            'scanRadius': self.scanRadius,

            'maxRepulsionDistance': self.maxRepulsionDistance,
            'minSafeDistance': self.minSafeDistance,

            'avoidRevisits': self.avoidRevisits,
            'targetSearchRange': self.targetSearchRange,
            'revisitCooldown': self.revisitCooldown,

            'position': {'x': self.position.x, 'y': self.position.y, 'z': self.position.z},
            'forward': {'x': self.forward.x, 'y': self.forward.y, 'z': self.forward.z},
            'scoreDir': {'x': self.scoreDir.x, 'y': self.scoreDir.y, 'z': self.scoreDir.z},
            'collideDir': {'x': self.collideDir.x, 'y': self.collideDir.y, 'z': self.collideDir.z},
            'pathDir': {'x': self.pathDir.x, 'y': self.pathDir.y, 'z': self.pathDir.z},
            'leaderRangeDir': {'x': self.leaderRangeDir.x, 'y': self.leaderRangeDir.y, 'z': self.leaderRangeDir.z},
            'directionRetentionDir': {'x': self.directionRetentionDir.x, 'y': self.directionRetentionDir.y,
                                      'z': self.directionRetentionDir.z},
            'finalMoveDir': {'x': self.finalMoveDir.x, 'y': self.finalMoveDir.y, 'z': self.finalMoveDir.z},

            'leaderPosition': {'x': self.leaderPosition.x, 'y': self.leaderPosition.y, 'z': self.leaderPosition.z},
            'leaderScanRadius': self.leaderScanRadius,

            'visitedCells': [{'x': cell.x, 'y': cell.y, 'z': cell.z} for cell in self.visitedCells]
        }
        return result

    def to_json(self) -> str:
        """将对象转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def is_within_safe_distance(self, other_position: Vector3) -> bool:
        """检查与另一个位置的距离是否在安全距离内"""
        distance = (self.position - other_position).magnitude()
        return distance >= self.minSafeDistance

    def add_visited_cell(self, cell_position: Vector3) -> None:
        """添加已访问的蜂窝位置"""
        # 检查是否已经存在相同位置的蜂窝
        for cell in self.visitedCells:
            if (cell.x == cell_position.x and 
                cell.y == cell_position.y and 
                cell.z == cell_position.z):
                return
        self.visitedCells.append(cell_position)

    def clear_visited_cells(self) -> None:
        """清空已访问的蜂窝记录"""
        self.visitedCells.clear()

    def get_visited_cell_count(self) -> int:
        """获取已访问的蜂窝数量"""
        return len(self.visitedCells)

    def update_move_direction(self, new_direction: Vector3) -> None:
        """更新移动方向"""
        if new_direction.magnitude() > 0.001:
            self.finalMoveDir = new_direction.normalized()

    def calculate_leader_distance(self) -> float:
        """计算与领导者的距离"""
        return (self.position - self.leaderPosition).magnitude()

    def is_leader_within_range(self) -> bool:
        """检查领导者是否在扫描范围内"""
        return self.calculate_leader_distance() <= self.leaderScanRadius

    def copy(self) -> 'ScannerData':
        """创建当前对象的深拷贝"""
        new_data = ScannerData()
        new_data.__dict__.update(self.__dict__)
        # 深拷贝列表和Vector3对象
        new_data.visitedCells = [Vector3(cell.x, cell.y, cell.z) for cell in self.visitedCells]
        new_data.position = Vector3(self.position.x, self.position.y, self.position.z)
        new_data.forward = Vector3(self.forward.x, self.forward.y, self.forward.z)
        new_data.scoreDir = Vector3(self.scoreDir.x, self.scoreDir.y, self.scoreDir.z)
        new_data.collideDir = Vector3(self.collideDir.x, self.collideDir.y, self.collideDir.z)
        new_data.pathDir = Vector3(self.pathDir.x, self.pathDir.y, self.pathDir.z)
        new_data.leaderRangeDir = Vector3(self.leaderRangeDir.x, self.leaderRangeDir.y, self.leaderRangeDir.z)
        new_data.directionRetentionDir = Vector3(self.directionRetentionDir.x, self.directionRetentionDir.y, self.directionRetentionDir.z)
        new_data.finalMoveDir = Vector3(self.finalMoveDir.x, self.finalMoveDir.y, self.finalMoveDir.z)
        new_data.leaderPosition = Vector3(self.leaderPosition.x, self.leaderPosition.y, self.leaderPosition.z)
        return new_data

    def __str__(self) -> str:
        """返回对象的字符串表示"""
        return f"ScannerData(position={self.position}, move_speed={self.moveSpeed}, visited_cells={len(self.visitedCells)})"

    def validate(self) -> bool:
        """验证数据的有效性"""
        # 验证关键参数是否为有效值
        if self.moveSpeed <= 0:
            return False
        if self.scanRadius <= 0:
            return False
        if self.minSafeDistance < 0:
            return False
        if self.maxRepulsionDistance < self.minSafeDistance:
            return False
        if self.updateInterval <= 0:
            return False
        return True
