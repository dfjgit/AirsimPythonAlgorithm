import numpy as np
import math
import logging
from .Vector3 import Vector3
from .HexGridDataModel import HexGridDataModel, HexCell
from .scanner_config_data import ScannerConfigData
from .scanner_runtime_data import ScannerRuntimeData
from typing import List, Dict, Tuple, Optional, Set
import time


class ObstacleHelper:
    """障碍物避障计算辅助类（支持Polygon/Circle形状）"""

    @staticmethod
    def calculate_normal_obstacle_repulsion(
        current_pos: Vector3, obstacle: Dict
    ) -> Vector3:
        """
        计算普通障碍物排斥力
        obstacle格式: {
            "obstacleId": "building-001",
            "obstacleType": "Static/Dynamic",
            "category": "Normal",
            "shapeType": "Polygon/Circle",
            "vertices": [...],  # Polygon时使用
            "center": {...},   # Circle时使用
            "radius": 15.0      # Circle时使用
        }
        """
        try:
            # 兼容整数枚举值和字符串值
            # 0/'Polygon' → 'polygon', 1/'Circle' → 'circle'
            raw_shape_type = obstacle.get("shapeType", "Unknown")
            if isinstance(raw_shape_type, int):
                # 整数枚举: 0=Polygon, 1=Circle
                shape_type = (
                    "polygon"
                    if raw_shape_type == 0
                    else "circle"
                    if raw_shape_type == 1
                    else "unknown"
                )
            else:
                # 字符串值
                shape_type = str(raw_shape_type).lower()

            if shape_type == "polygon":
                # 多边形障碍物：计算到多边形的排斥力
                return ObstacleHelper._calculate_polygon_obstacle_repulsion(
                    current_pos, obstacle
                )
            elif shape_type == "circle":
                # 圆形障碍物：计算到圆形的排斥力
                return ObstacleHelper._calculate_circle_obstacle_repulsion(
                    current_pos, obstacle
                )
            else:
                # 未知类型：使用简单距离衰减
                return Vector3()

        except Exception as e:
            logging.warning(f"计算障碍物排斥力失败: {e}")
            return Vector3()

    @staticmethod
    def _calculate_polygon_obstacle_repulsion(
        current_pos: Vector3, obstacle: Dict
    ) -> Vector3:
        """计算多边形障碍物的排斥力"""
        vertices_data = obstacle.get("vertices", [])
        if not vertices_data:
            return Vector3()

        # 解析顶点
        vertices = [
            Vector3(v.get("x", 0.0), v.get("y", 0.0), v.get("z", 0.0))
            for v in vertices_data
        ]

        # 使用点到多边形的距离计算
        min_dist, closest_point = RestrictedZoneHelper.point_to_polygon_distance(
            current_pos, vertices
        )

        if min_dist < 0.01:
            return Vector3()

        # 计算排斥方向
        repulsion_vec = current_pos - closest_point
        if repulsion_vec.magnitude() > 0.001:
            repulsion_dir = repulsion_vec.normalized()
            # 距离越近，排斥力越大
            distance_factor = max(0.1, 1.0 - (min_dist / 15.0))  # 15.0是默认排斥距离
            return repulsion_dir * distance_factor

        return Vector3()

    @staticmethod
    def _calculate_circle_obstacle_repulsion(
        current_pos: Vector3, obstacle: Dict
    ) -> Vector3:
        """计算圆形障碍物的排斥力"""
        center_data = obstacle.get("center", {})
        if not center_data:
            return Vector3()

        center = Vector3(
            center_data.get("x", 0.0),
            center_data.get("y", 0.0),
            center_data.get("z", 0.0),
        )
        radius = obstacle.get("radius", 5.0)

        # 使用点到圆形的距离计算
        distance, closest_point = RestrictedZoneHelper.point_to_circle_distance(
            current_pos, center, radius
        )

        if distance < 0.01:
            return Vector3()

        # 计算排斥方向
        repulsion_vec = current_pos - closest_point
        if repulsion_vec.magnitude() > 0.001:
            repulsion_dir = repulsion_vec.normalized()
            # 距离越近，排斥力越大
            distance_factor = max(0.1, 1.0 - (distance / 15.0))
            return repulsion_dir * distance_factor

        return Vector3()


class RestrictedZoneHelper:
    """禁飞区几何计算辅助类（支持多边形和圆形）"""

    @staticmethod
    def point_to_segment_distance(
        point: Vector3, seg_start: Vector3, seg_end: Vector3
    ) -> Tuple[float, Vector3]:
        """
        计算点到线段的最短距离
        返回: (距离, 最近点)
        """
        # 线段向量
        seg_vec = seg_end - seg_start
        # 点到线段起点的向量
        point_vec = point - seg_start

        seg_length_sq = seg_vec.x**2 + seg_vec.y**2 + seg_vec.z**2

        if seg_length_sq < 0.0001:
            # 线段退化为点
            return (point_vec.magnitude(), seg_start)

        # 计算投影参数 t
        t = max(
            0.0,
            min(
                1.0,
                (
                    point_vec.x * seg_vec.x
                    + point_vec.y * seg_vec.y
                    + point_vec.z * seg_vec.z
                )
                / seg_length_sq,
            ),
        )

        # 投影点
        projection = seg_start + seg_vec * t

        # 距离
        distance = (point - projection).magnitude()
        return (distance, projection)

    @staticmethod
    def point_to_polygon_distance(
        point: Vector3, vertices: List[Vector3]
    ) -> Tuple[float, Vector3]:
        """
        计算点到多边形的最短距离
        返回: (距离, 最近点)
        """
        if len(vertices) < 3:
            # 至少需要3个点才能构成多边形
            min_dist = float("inf")
            closest_point = point
            for v in vertices:
                dist = (point - v).magnitude()
                if dist < min_dist:
                    min_dist = dist
                    closest_point = v
            return (min_dist, closest_point)

        min_distance = float("inf")
        closest_point = point

        # 检查点到每条边的距离
        for i in range(len(vertices)):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % len(vertices)]

            edge_start = v1
            edge_end = v2

            dist, proj = RestrictedZoneHelper.point_to_segment_distance(
                point, edge_start, edge_end
            )

            if dist < min_distance:
                min_distance = dist
                closest_point = proj

        return (min_distance, closest_point)

    @staticmethod
    def is_point_inside_polygon(point: Vector3, vertices: List[Vector3]) -> bool:
        """
        判断点是否在多边形内部（使用射线法）
        """
        if len(vertices) < 3:
            return False

        # 投影到2D平面（使用x和z坐标，忽略y高度）
        x, z = point.x, point.z
        n = len(vertices)
        inside = False

        p1x, p1z = vertices[0].x, vertices[0].z
        for i in range(n + 1):
            p2x, p2z = vertices[i % n].x, vertices[i % n].z

            if z > min(p1z, p2z):
                if z <= max(p1z, p2z) and x <= max(p1x, p2x):
                    if p1z != p2z:
                        xinters = (z - p1z) * (p2x - p1x) / (p2z - p1z) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
                elif x <= max(p1x, p2x) and z <= max(p1z, p2z):
                    if p1x != p2x:
                        zinters = (x - p1x) * (p2z - p1z) / (p2x - p1x) + p1z
                    if p1z == p2z or z <= zinters:
                        inside = not inside
            p1x, p1z = p2x, p2z

        return inside

    @staticmethod
    def is_point_inside_circle(point: Vector3, center: Vector3, radius: float) -> bool:
        """
        判断点是否在圆形内部
        在水平面上计算距离（忽略y高度）
        """
        # 在xz平面上计算距离
        dx = point.x - center.x
        dz = point.z - center.z
        distance = (dx**2 + dz**2) ** 0.5
        return distance <= radius

    @staticmethod
    def point_to_circle_distance(
        point: Vector3, center: Vector3, radius: float
    ) -> Tuple[float, Vector3]:
        """
        计算点到圆形的最短距离
        返回: (距离, 最近点)
        在水平面上计算（忽略y高度）
        """
        # 在xz平面上计算距离
        dx = point.x - center.x
        dz = point.z - center.z
        horizontal_distance = (dx**2 + dz**2) ** 0.5

        if horizontal_distance < 0.001:
            # 点在圆心
            return (radius, Vector3(center.x + radius, point.y, center.z))

        if horizontal_distance <= radius:
            # 点在圆内，距离为0
            return (0.0, point)
        else:
            # 点在圆外，计算最近点（在圆周上）
            distance_to_edge = horizontal_distance - radius
            # 计算圆周上最近点的位置
            ratio = radius / horizontal_distance
            closest_point = Vector3(
                center.x + dx * ratio,
                point.y,  # 保持原始高度
                center.z + dz * ratio,
            )
            return (distance_to_edge, closest_point)

    @staticmethod
    def check_height_limit(
        point: Vector3, max_height: float = None, min_height: float = None
    ) -> bool:
        """
        检查点是否在高度限制内
        返回: True表示在限制范围内，False表示超出范围
        """
        if max_height is not None and point.y > max_height:
            return False
        if min_height is not None and point.y < min_height:
            return False
        return True


# 确保使用正确的坐标系
def ensure_unity_coordinates(vector: Vector3) -> Vector3:
    """确保向量使用Unity坐标系"""
    # 检查是否需要转换（如果Vector3实例已经有转换方法）
    if hasattr(vector, "unity_to_air_sim"):
        # 这里根据实际需要决定是否进行转换
        # 注意：这个函数是一个安全措施，确保坐标系的一致性
        pass
    return vector


class ScannerAlgorithm:
    def __init__(self, config_data: ScannerConfigData):
        """初始化扫描器算法，传入配置数据"""
        self.config = config_data
        self.last_update_time = 0.0
        # 初始化上一帧的移动方向，使用Unity坐标系中的默认向前方向
        self.previous_move_dir = ensure_unity_coordinates(
            Vector3(0, 0, 1)
        )  # 默认方向：z轴正方向
        self.visited_cells: Dict[
            Tuple[float, float, float], float
        ] = {}  # 存储访问时间 (x,y,z) -> timestamp

        # 初始化普通障碍物数据（运行时数据，包括静态和动态）
        self.normal_obstacles: List[Dict] = []  # category=Normal的障碍物列表
        self.obstacle_repulsion_distance = 15.0  # 障碍物排斥力有效距离
        self.obstacle_repulsion_coefficient = 5.0  # 障碍物排斥力系数

        # 初始化禁飞区数据（运行时数据，category=RestrictedZone的障碍物）
        self.restricted_zones: List[Dict] = []  # 禁飞区列表
        self.restricted_zone_repulsion_distance = 15.0
        self.restricted_zone_repulsion_coefficient = 5.0

    def set_normal_obstacles(self, obstacles: List[Dict]) -> None:
        """
        设置普通障碍物数据（运行时数据，由AlgorithmServer调用）
        障碍物格式: [
            {
                "obstacleId": "building-001",
                "obstacleType": "Static",
                "category": "Normal",
                "shapeType": "Polygon/Circle",
                "vertices": [...],  # Polygon时使用
                "center": {...},   # Circle时使用
                "radius": 15.0      # Circle时使用
            }
        ]
        """
        if isinstance(obstacles, list):
            self.normal_obstacles = obstacles
            logging.info(f"普通障碍物数据已更新，数量: {len(obstacles)}")
        else:
            logging.warning(
                f"普通障碍物数据格式错误，期望list，收到: {type(obstacles)}"
            )

    def set_restricted_zones(self, obstacles: List[Dict]) -> None:
        """
        设置禁飞区数据（运行时数据，由AlgorithmServer调用）
        障碍物格式: [
            {
                "obstacleId": "no-fly-001",
                "obstacleType": "Static",
                "category": "RestrictedZone",
                "shapeType": "Polygon/Circle",
                "vertices": [...],  # Polygon时使用
                "center": {...},   # Circle时使用
                "radius": 15.0      # Circle时使用
            }
        ]
        """
        if isinstance(obstacles, list):
            self.restricted_zones = obstacles
            logging.info(f"禁飞区数据已更新，数量: {len(obstacles)}")
        else:
            logging.warning(f"禁飞区数据格式错误，期望list，收到: {type(obstacles)}")

    def calculate_proportional_weights(
        self,
    ) -> Tuple[float, float, float, float, float]:
        """计算权重：F = 系数 / 系数总和（与C#逻辑一致）"""
        total = (
            self.config.repulsionCoefficient
            + self.config.entropyCoefficient
            + self.config.distanceCoefficient
            + self.config.leaderRangeCoefficient
            + self.config.directionRetentionCoefficient
        )

        # 处理所有系数都为0的特殊情况
        if total < 0.001:
            return (0.2, 0.2, 0.2, 0.2, 0.2)

        return (
            self.config.repulsionCoefficient / total,
            self.config.entropyCoefficient / total,
            self.config.distanceCoefficient / total,
            self.config.leaderRangeCoefficient / total,
            self.config.directionRetentionCoefficient / total,
        )

    def get_valid_candidate_cells(
        self, grid_data: HexGridDataModel, runtime_data: ScannerRuntimeData
    ) -> List[HexCell]:
        """获取有效的候选蜂窝（与C# GetValidCandidateCells逻辑一致）"""
        # 确保使用Unity坐标系
        current_pos = ensure_unity_coordinates(runtime_data.position)
        candidate_cells = []

        # 首先检查无人机本身是否在Leader范围内
        if (
            runtime_data.leader_position is not None
            and runtime_data.leader_scan_radius > 0
        ):
            leader_pos = ensure_unity_coordinates(runtime_data.leader_position)
            distance_to_leader = (current_pos - leader_pos).magnitude()

            if distance_to_leader > runtime_data.leader_scan_radius:
                # 无人机超出Leader范围，不扫描任何蜂窝
                return candidate_cells

        for cell in grid_data.cells:
            # 确保蜂窝中心也使用Unity坐标系
            cell_center = ensure_unity_coordinates(cell.center)

            # 检查蜂窝是否在Leader范围内（可选：如果无人机在范围内，蜂窝也应该在范围内）
            if (
                runtime_data.leader_position is not None
                and runtime_data.leader_scan_radius > 0
            ):
                leader_pos = ensure_unity_coordinates(runtime_data.leader_position)
                distance_to_leader = (cell_center - leader_pos).magnitude()
                if distance_to_leader > runtime_data.leader_scan_radius:
                    continue  # 蜂窝不在Leader范围内，跳过

            # 检查是否在搜索范围内
            distance_to_cell = (cell_center - current_pos).magnitude()
            if distance_to_cell > self.config.targetSearchRange:
                continue  # 超出搜索范围，跳过

            # 检查是否需要避免重复访问
            if self.config.avoidRevisits:
                # 四舍五入避免浮点数精度问题
                rounded_center = (
                    round(cell.center.x * 100) / 100,
                    round(cell.center.y * 100) / 100,
                    round(cell.center.z * 100) / 100,
                )

                if rounded_center in self.visited_cells:
                    # 检查是否在冷却期内
                    if (
                        time.time() - self.visited_cells[rounded_center]
                        < self.config.revisitCooldown
                    ):
                        continue  # 仍在冷却期，跳过

            candidate_cells.append(cell)

        return candidate_cells

    def calculate_score_direction(
        self, grid_data: HexGridDataModel, runtime_data: ScannerRuntimeData
    ) -> Vector3:
        """计算熵最优方向向量（与C# CalculateScoreDirection逻辑一致）"""
        # 确保使用Unity坐标系
        current_pos = ensure_unity_coordinates(runtime_data.position)

        # 保留原始的y坐标，不强制设置为0，以确保3D空间中的准确计算
        # current_pos = Vector3(runtime_data.position.x, 0, runtime_data.position.z)  # 移除这个有问题的代码

        candidate_cells = self.get_valid_candidate_cells(grid_data, runtime_data)

        if not candidate_cells:
            # 当没有可用候选单元格时（都被访问过），返回朝向Leader中心的方向
            # 避免无人机在小范围内徘徊
            if runtime_data.leader_position and runtime_data.leader_scan_radius > 0:
                current_pos = ensure_unity_coordinates(runtime_data.position)
                leader_pos = ensure_unity_coordinates(runtime_data.leader_position)
                to_leader = leader_pos - current_pos
                if to_leader.magnitude() > 0.1:
                    return to_leader.normalized()
            # 如果连Leader方向都没有，返回默认前方
            return Vector3(0, 0, 1)

        # 归一化熵值范围（0-1）
        entropies = [cell.entropy for cell in candidate_cells]
        min_entropy = min(entropies)
        max_entropy = max(entropies)
        entropy_range = max_entropy - min_entropy
        all_entropy_same = abs(entropy_range) < 0.01

        # 计算每个候选蜂窝的分数
        scored_cells = []
        for cell in candidate_cells:
            # 确保蜂窝中心也使用Unity坐标系
            cell_center = ensure_unity_coordinates(cell.center)
            distance = (cell_center - current_pos).magnitude()
            normalized_distance = min(
                1.0, max(0.0, 1 - (distance / self.config.targetSearchRange))
            )

            # 计算熵值分数
            if all_entropy_same:
                entropy_score = 0.5
            else:
                entropy_score = (cell.entropy - min_entropy) / entropy_range

            # 综合分数：熵值为主（70%），距离为辅（30%）
            total_score = entropy_score * 0.4 + normalized_distance * 0.6
            scored_cells.append((cell, total_score))

        # 选择最高分的蜂窝作为目标
        best_cell = max(scored_cells, key=lambda x: x[1])[0]

        # 确保计算方向时使用正确的坐标系
        best_cell_center = ensure_unity_coordinates(best_cell.center)
        score_dir = (best_cell_center - current_pos).normalized()

        # 记录访问
        self.record_visited_cell(best_cell_center)
        return score_dir

    def calculate_path_direction(self, score_dir: Vector3) -> Vector3:
        """计算最短路径方向向量（与C# CalculatePathDirection逻辑一致）"""
        # 确保路径方向也使用Unity坐标系
        return ensure_unity_coordinates(score_dir)  # 路径方向与分数方向一致

    def calculate_collide_direction(self, runtime_data: ScannerRuntimeData) -> Vector3:
        """计算排斥力方向向量（与C# CalculateRepulsionDirection逻辑一致）"""
        collide_dir = Vector3()
        # 确保使用Unity坐标系
        current_pos = ensure_unity_coordinates(runtime_data.position)

        # 其他扫描器位置
        other_scanners = runtime_data.otherScannerPositions

        for other_pos in other_scanners:
            # 确保其他扫描器的位置也使用Unity坐标系
            other_pos_unity = ensure_unity_coordinates(other_pos)
            delta_pos = current_pos - other_pos_unity
            distance = delta_pos.magnitude()

            # 超出排斥范围或距离过近（避免除以零）
            if distance > self.config.maxRepulsionDistance or distance < 0.1:
                continue

            # 计算排斥力比例
            repulsion_ratio = self.calculate_repulsion_ratio(distance)
            collide_dir += delta_pos.normalized() * repulsion_ratio

        # 确保返回的排斥方向向量在Unity坐标系中正确
        return ensure_unity_coordinates(
            collide_dir.normalized() if collide_dir.magnitude() > 0.1 else collide_dir
        )

    def calculate_repulsion_ratio(self, distance: float) -> float:
        """计算排斥力比例（与C# CalculateRepulsionRatio逻辑一致）"""
        if distance <= self.config.minSafeDistance:
            return 1.0
        if distance >= self.config.maxRepulsionDistance:
            return 0.0

        # 非线性衰减，近距离排斥力增长更快
        t = (distance - self.config.minSafeDistance) / (
            self.config.maxRepulsionDistance - self.config.minSafeDistance
        )
        return 1.0 - (t * t)

    def calculate_leader_range_direction(
        self, runtime_data: ScannerRuntimeData
    ) -> Vector3:
        """计算保持在Leader范围内的方向向量（平滑版本，减少震荡）"""
        leader_range_dir = Vector3()

        # 确保使用Unity坐标系
        current_pos = ensure_unity_coordinates(runtime_data.position)
        leader_pos = ensure_unity_coordinates(runtime_data.leader_position)
        leader_scan_radius = runtime_data.leader_scan_radius

        if leader_pos is None or leader_scan_radius <= 0:
            return leader_range_dir

        # 计算与Leader的距离（在Unity坐标系中）
        distance_to_leader = (current_pos - leader_pos).magnitude()

        # 避免除零错误
        if leader_scan_radius < 0.001:
            return leader_range_dir

        # 计算距离比例（0在中心，1在边界）
        distance_ratio = distance_to_leader / leader_scan_radius

        # 使用平滑的过渡区域，避免突变（软边界设计）
        # 舒适区: 0.05 - 0.96 (大幅扩大舒适区，91%范围不施加力)
        # 警告区: 0.96 - 1.0 (仅在最后4%渐增加力，最大力降低至0.2)
        # 危险区: > 1.0 (真正出界才强力拉回)
        # 中心区: < 0.05 (仅在核心5%轻微推开)

        if distance_ratio > 1.0:
            # 出界：强力但平滑地拉回
            excess_ratio = min(1.0, distance_ratio - 1.0)
            # 使用平方根使力量增长更平缓
            force = 0.5 + 0.5 * (excess_ratio**0.5)
            direction = (leader_pos - current_pos).normalized()
            leader_range_dir = direction * force
        elif distance_ratio > 0.96:
            # 警告区：仅在最后4%施加轻微拉回力
            # 从0.96到1.0，力量从0渐变到0.2（最大力降低67%）
            t = (distance_ratio - 0.96) / 0.04  # 0到1
            force = 0.2 * (t * t)  # 二次方使过渡更平滑
            direction = (leader_pos - current_pos).normalized()
            leader_range_dir = direction * force
        elif distance_ratio < 0.05 and distance_ratio > 0.001:
            # 中心区：仅在核心5%轻微推开（避免聚集）
            # 越靠近中心，推开力越小
            t = (0.05 - distance_ratio) / 0.05  # 0到1
            force = 0.08 * (1.0 - t)  # 最大0.08，中心为0
            direction = (current_pos - leader_pos).normalized()
            leader_range_dir = direction * force
        # 舒适区 (0.05 - 0.96): 91%范围不施加Leader方向力，让方向保持力主导

        return leader_range_dir

    def calculate_direction_retention_direction(self) -> Vector3:
        """计算方向保持向量（完全移除随机扰动，消除树杈状徘徊）"""
        if self.previous_move_dir and isinstance(self.previous_move_dir, Vector3):
            # 完全移除随机扰动，直接返回上一帧的方向
            result = self.previous_move_dir
            # 归一化
            if result.magnitude() > 0.001:
                result = result.normalized()
            return ensure_unity_coordinates(result)
        return Vector3(0, 0, 1)  # 默认方向

    def calculate_normal_obstacles_direction(self, current_pos: Vector3) -> Vector3:
        """
        计算普通障碍物的排斥力方向向量
        支持：Static/Dynamic, Polygon/Circle
        """
        obstacle_dir = Vector3()

        if not self.normal_obstacles:
            return obstacle_dir

        for obstacle in self.normal_obstacles:
            repulsion = ObstacleHelper.calculate_normal_obstacle_repulsion(
                current_pos, obstacle
            )
            obstacle_dir += repulsion

        return obstacle_dir

    def calculate_restricted_zone_direction(
        self, runtime_data: ScannerRuntimeData
    ) -> Vector3:
        """
        计算禁飞区排斥力方向向量
        支持：Static, Polygon/Circle（通过category=RestrictedZone标识）
        返回远离禁飞区的方向向量
        """
        zone_dir = Vector3()

        # 确保使用Unity坐标系
        current_pos = ensure_unity_coordinates(runtime_data.position)

        # 处理禁飞区数据
        if not self.restricted_zones:
            return zone_dir

        for zone in self.restricted_zones:
            # 兼容整数枚举值和字符串值
            # 0/'Polygon' → 'polygon', 1/'Circle' → 'circle'
            raw_shape_type = zone.get("shapeType", "")
            if isinstance(raw_shape_type, int):
                # 整数枚举: 0=Polygon, 1=Circle
                shape_type = (
                    "polygon"
                    if raw_shape_type == 0
                    else "circle"
                    if raw_shape_type == 1
                    else "unknown"
                )
            else:
                # 字符串值
                shape_type = str(raw_shape_type).lower()

            if shape_type == "polygon":
                # 多边形禁飞区处理
                vertices_data = zone.get("vertices", [])
                vertices = [
                    Vector3(v.get("x", 0.0), v.get("y", 0.0), v.get("z", 0.0))
                    for v in vertices_data
                ]
                if len(vertices) < 3:
                    continue
                zone_dir += self._calculate_polygon_repulsion(current_pos, vertices)

            elif shape_type == "circle":
                # 圆形禁飞区处理
                center_data = zone.get("center", {})
                center = Vector3(
                    center_data.get("x", 0.0),
                    center_data.get("y", 0.0),
                    center_data.get("z", 0.0),
                )
                radius = zone.get("radius", 10.0)
                zone_dir += self._calculate_circle_repulsion(
                    current_pos, center, radius
                )

        return zone_dir

    def _calculate_polygon_repulsion(
        self, current_pos: Vector3, vertices: List[Vector3]
    ) -> Vector3:
        """计算多边形禁飞区的排斥力"""
        repulsion_dir = Vector3()

        # 检查无人机当前位置是否在禁飞区内部
        is_inside = RestrictedZoneHelper.is_point_inside_polygon(current_pos, vertices)

        if is_inside:
            # 在禁飞区内部：强力排斥到最近的安全点
            min_dist, closest_point = RestrictedZoneHelper.point_to_polygon_distance(
                current_pos, vertices
            )

            if min_dist < 0.1:
                # 非常接近边界，计算指向外的方向
                min_vertex_dist = float("inf")
                escape_dir = Vector3()

                for v in vertices:
                    vec = v - current_pos
                    dist = vec.magnitude()
                    if dist < min_vertex_dist and dist > 0.01:
                        min_vertex_dist = dist
                        escape_dir = vec.normalized()

                if escape_dir.magnitude() > 0.1:
                    repulsion_dir += (
                        escape_dir * self.restricted_zone_repulsion_coefficient * 2.0
                    )
            else:
                # 距离边界有一定距离，计算排斥方向
                repulsion_vec = current_pos - closest_point
                if repulsion_vec.magnitude() > 0.1:
                    direction = repulsion_vec.normalized()
                    distance_factor = max(
                        0.1, 1.0 - (min_dist / self.restricted_zone_repulsion_distance)
                    )
                    repulsion_dir += direction * (
                        self.restricted_zone_repulsion_coefficient * distance_factor
                    )

        else:
            # 在禁飞区外部：检查是否接近禁飞区边界
            min_dist, closest_point = RestrictedZoneHelper.point_to_polygon_distance(
                current_pos, vertices
            )

            if min_dist < self.restricted_zone_repulsion_distance:
                # 接近禁飞区，施加排斥力
                repulsion_vec = current_pos - closest_point
                if repulsion_vec.magnitude() > 0.1:
                    direction = repulsion_vec.normalized()
                    distance_factor = 1.0 - (
                        min_dist / self.restricted_zone_repulsion_distance
                    )
                    repulsion_dir += direction * (
                        self.restricted_zone_repulsion_coefficient * distance_factor
                    )

        return repulsion_dir

    def _calculate_circle_repulsion(
        self, current_pos: Vector3, center: Vector3, radius: float
    ) -> Vector3:
        """计算圆形禁飞区的排斥力"""
        repulsion_dir = Vector3()

        # 检查无人机当前位置是否在圆形内部
        is_inside = RestrictedZoneHelper.is_point_inside_circle(
            current_pos, center, radius
        )
        distance, closest_point = RestrictedZoneHelper.point_to_circle_distance(
            current_pos, center, radius
        )

        if is_inside:
            # 在圆形内部：强力排斥到圆外
            if distance < 0.1:
                # 非常接近圆心，沿径向向外推
                to_center = current_pos - center
                if to_center.magnitude() > 0.01:
                    escape_dir = to_center.normalized()
                    repulsion_dir += (
                        escape_dir * self.restricted_zone_repulsion_coefficient * 2.0
                    )
            else:
                # 距离边界有一定距离，计算排斥方向
                repulsion_vec = current_pos - closest_point
                if repulsion_vec.magnitude() > 0.1:
                    direction = repulsion_vec.normalized()
                    distance_factor = max(
                        0.1, 1.0 - (distance / self.restricted_zone_repulsion_distance)
                    )
                    repulsion_dir += direction * (
                        self.restricted_zone_repulsion_coefficient * distance_factor
                    )

        else:
            # 在圆形外部：检查是否接近圆形边界
            if distance < self.restricted_zone_repulsion_distance:
                # 接近圆形边界，施加排斥力
                repulsion_vec = current_pos - closest_point
                if repulsion_vec.magnitude() > 0.1:
                    direction = repulsion_vec.normalized()
                    distance_factor = 1.0 - (
                        distance / self.restricted_zone_repulsion_distance
                    )
                    repulsion_dir += direction * (
                        self.restricted_zone_repulsion_coefficient * distance_factor
                    )

        return repulsion_dir

    def merge_directions(
        self,
        score_dir: Vector3,
        path_dir: Vector3,
        collide_dir: Vector3,
        leader_range_dir: Vector3,
        direction_retention_dir: Vector3,
        weights: Tuple[float, float, float, float, float],
        runtime_data: ScannerRuntimeData = None,
    ) -> Vector3:
        """合并所有方向向量（与C# MergeDirections逻辑一致）"""
        (
            repulsion_weight,
            entropy_weight,
            distance_weight,
            leader_range_weight,
            direction_retention_weight,
        ) = weights

        # 确保所有输入向量都使用Unity坐标系
        score_dir = ensure_unity_coordinates(score_dir)
        path_dir = ensure_unity_coordinates(path_dir)
        collide_dir = ensure_unity_coordinates(collide_dir)
        leader_range_dir = ensure_unity_coordinates(leader_range_dir)
        direction_retention_dir = ensure_unity_coordinates(direction_retention_dir)

        # 计算障碍物和禁飞区排斥方向
        obstacle_dir = Vector3()
        restricted_zone_dir = Vector3()
        if runtime_data is not None:
            current_pos = ensure_unity_coordinates(runtime_data.position)
            # 处理普通障碍物
            if self.normal_obstacles:
                obstacle_dir = self.calculate_normal_obstacles_direction(current_pos)
            # 处理禁飞区
            if self.restricted_zones:
                restricted_zone_dir = self.calculate_restricted_zone_direction(
                    runtime_data
                )

        # 应用权重合并向量
        final_move_dir = (
            score_dir * entropy_weight
            + path_dir * distance_weight
            + collide_dir * repulsion_weight
            + leader_range_dir * leader_range_weight
            + direction_retention_dir * direction_retention_weight
            + obstacle_dir * (repulsion_weight * 1.0)  # 普通障碍物排斥力
            + restricted_zone_dir * (repulsion_weight * 1.5)  # 禁飞区排斥力权重稍高
        )

        # 归一化最终方向
        if final_move_dir.magnitude() > 0.01:  # 降低阈值，避免过早判定为零向量
            final_move_dir = ensure_unity_coordinates(final_move_dir.normalized())

            # 动量平滑机制：只在转向角度小时平滑，大角度时快速响应
            # 这样既能平滑轨迹，又不会被旧方向"拉回"
            if (
                self.previous_move_dir
                and self.previous_move_dir.magnitude() > 0.1
            ):
                import math

                # 计算新旧方向的夹角
                dot_product = (
                    final_move_dir.x * self.previous_move_dir.x
                    + final_move_dir.y * self.previous_move_dir.y
                    + final_move_dir.z * self.previous_move_dir.z
                )
                dot_product = max(-1.0, min(1.0, dot_product))
                angle = math.degrees(math.acos(abs(dot_product)))

                # 只在小角度转向时平滑，大角度时直接使用新方向
                if angle < 30:
                    # 小转向：平滑过渡，95%新方向 + 5%旧方向
                    smooth_dir = (
                        self.previous_move_dir * 0.05
                        + final_move_dir * 0.95
                    )
                    return ensure_unity_coordinates(smooth_dir.normalized())
                elif angle < 60:
                    # 中等转向：轻微平滑，98%新方向 + 2%旧方向
                    smooth_dir = (
                        self.previous_move_dir * 0.02
                        + final_move_dir * 0.98
                    )
                    return ensure_unity_coordinates(smooth_dir.normalized())
                else:
                    # 大转向：直接使用新方向，快速响应
                    return final_move_dir

            return final_move_dir
        else:
            # 如果最终方向接近零，保持当前方向
            return ensure_unity_coordinates(self.previous_move_dir)

    def record_visited_cell(self, cell_center: Vector3) -> None:
        """记录已访问的蜂窝（与C# RecordVisitedCell逻辑一致）"""
        if not self.config.avoidRevisits:
            return

        # 四舍五入避免浮点数精度问题
        rounded_center = (
            round(cell_center.x * 100) / 100,
            round(cell_center.y * 100) / 100,
            round(cell_center.z * 100) / 100,
        )

        self.visited_cells[rounded_center] = time.time()

    def cleanup_visited_records(self) -> None:
        """清理过期的访问记录（与C# CleanupVisitedRecords逻辑一致）"""
        if not self.config.avoidRevisits:
            return

        current_time = time.time()
        expired_keys = [
            key
            for key, timestamp in self.visited_cells.items()
            if current_time - timestamp >= self.config.revisitCooldown
        ]

        for key in expired_keys:
            del self.visited_cells[key]

    def set_coefficients(self, coefficients):
        """动态设置权重系数"""
        if "repulsionCoefficient" in coefficients:
            self.config.repulsionCoefficient = coefficients["repulsionCoefficient"]
        if "entropyCoefficient" in coefficients:
            self.config.entropyCoefficient = coefficients["entropyCoefficient"]
        if "distanceCoefficient" in coefficients:
            self.config.distanceCoefficient = coefficients["distanceCoefficient"]
        if "leaderRangeCoefficient" in coefficients:
            self.config.leaderRangeCoefficient = coefficients["leaderRangeCoefficient"]
        if "directionRetentionCoefficient" in coefficients:
            self.config.directionRetentionCoefficient = coefficients[
                "directionRetentionCoefficient"
            ]
        # 避障参数
        if "obstacleRepulsionDistance" in coefficients:
            self.obstacle_repulsion_distance = coefficients["obstacleRepulsionDistance"]
        if "obstacleRepulsionCoefficient" in coefficients:
            self.obstacle_repulsion_coefficient = coefficients[
                "obstacleRepulsionCoefficient"
            ]

    def get_current_coefficients(self):
        """获取当前权重系数"""
        return {
            "repulsionCoefficient": self.config.repulsionCoefficient,
            "entropyCoefficient": self.config.entropyCoefficient,
            "distanceCoefficient": self.config.distanceCoefficient,
            "leaderRangeCoefficient": self.config.leaderRangeCoefficient,
            "directionRetentionCoefficient": self.config.directionRetentionCoefficient,
            "obstacleRepulsionDistance": self.obstacle_repulsion_distance,
            "obstacleRepulsionCoefficient": self.obstacle_repulsion_coefficient,
            "restrictedZoneDistance": self.restricted_zone_repulsion_distance,
            "restrictedZoneCoefficient": self.restricted_zone_repulsion_coefficient,
        }

    def update_runtime_data(
        self, grid_data: HexGridDataModel, runtime_data: ScannerRuntimeData
    ) -> ScannerRuntimeData:
        """更新运行时数据（供其他组件使用的接口）"""
        try:
            # 类型检查
            if not isinstance(grid_data, HexGridDataModel):
                logging.warning(
                    f"ScannerAlgorithm.update_runtime_data: grid_data类型无效，期望HexGridDataModel，得到: {type(grid_data).__name__}"
                )
                return runtime_data

            if not isinstance(runtime_data, ScannerRuntimeData):
                logging.warning(
                    f"ScannerAlgorithm.update_runtime_data: runtime_data类型无效，期望ScannerRuntimeData，得到: {type(runtime_data).__name__}"
                )
                return runtime_data
            current_time = time.time()

            # 定期更新方向（根据updateInterval）
            if current_time - self.last_update_time >= self.config.updateInterval:
                self.last_update_time = current_time

                # 保存当前方向作为下一帧的"previousMoveDir"
                try:
                    if (
                        runtime_data.finalMoveDir
                        and runtime_data.finalMoveDir.magnitude() > 0.1
                    ):
                        self.previous_move_dir = runtime_data.finalMoveDir
                except Exception as e:
                    logging.warning(
                        f"ScannerAlgorithm.update_runtime_data: 获取finalMoveDir失败: {str(e)}"
                    )

                # 计算各权重
                weights = self.calculate_proportional_weights()

                # 计算各方向向量
                try:
                    score_dir = self.calculate_score_direction(grid_data, runtime_data)
                    path_dir = self.calculate_path_direction(score_dir)
                    collide_dir = self.calculate_collide_direction(runtime_data)
                    leader_range_dir = self.calculate_leader_range_direction(
                        runtime_data
                    )
                    direction_retention_dir = (
                        self.calculate_direction_retention_direction()
                    )

                    # 合并所有向量
                    final_move_dir = self.merge_directions(
                        score_dir,
                        path_dir,
                        collide_dir,
                        leader_range_dir,
                        direction_retention_dir,
                        weights,
                        runtime_data,
                    )

                    try:
                        if runtime_data.position is not None and isinstance(
                            runtime_data.position, Vector3
                        ):
                            # 提高目标高度到2.0米（避免掉到地面）
                            target_height = 2.0
                            height_deadband = 0.1
                            max_correction = 1.0

                            current_height = runtime_data.position.y
                            height_error = target_height - current_height

                            # 如果高度过低（小于1米），增强上升力度
                            if current_height < 1.0:
                                logging.warning(
                                    f"[{runtime_data.uavname}] 🚨 高度过低({current_height:.2f}m)，增强上升力度"
                                )
                                # 使用平滑函数，不是突然添加1.0
                                extra_force = (1.0 - current_height) * 0.5  # 根据高度差距动态调整
                                height_error = height_error + extra_force

                            if abs(height_error) > height_deadband:
                                correction = max(
                                    -max_correction, min(max_correction, height_error)
                                )
                                # 优化：不再强制设为0.3，而是使用平滑函数
                                # 对于小修正值，使用平方函数使其平滑增长
                                if correction > 0 and correction < 0.3:
                                    # 使用平方函数，使0.1->0.01, 0.2->0.04, 0.3->0.09
                                    # 然后缩放到[0.1, 0.3]范围
                                    normalized = correction / 0.3  # 归一化到[0, 1]
                                    correction = 0.1 + 0.2 * (normalized ** 2)  # 平滑过渡

                                # 保存原始Y分量
                                original_y = final_move_dir.y

                                # 平滑混合原始Y和修正值
                                # 当高度严重偏离时，更倾向于使用修正值
                                blend_factor = min(1.0, abs(height_error) / 2.0)  # 高度误差越大，blend_factor越接近1
                                new_y = original_y * (1 - blend_factor) + correction * blend_factor

                                final_move_dir = Vector3(
                                    final_move_dir.x,
                                    new_y,
                                    final_move_dir.z,
                                )

                            # 如果高度低于0.5米，强制禁止下降（平滑处理）
                            if current_height < 0.5 and final_move_dir.y < 0:
                                logging.warning(
                                    f"[{runtime_data.uavname}] 🚫 禁止下降（当前高度{current_height:.2f}m），转为上升"
                                )
                                # 不是直接设为0.5，而是将Y分量设为正值，保持XZ方向不变
                                # 使用平滑过渡：根据高度差距动态调整上升力度
                                rise_force = (0.5 - current_height) * 0.8 + 0.2  # 范围[0.2, 0.6]
                                final_move_dir = Vector3(
                                    final_move_dir.x,
                                    rise_force,  # 动态上升力度，而不是固定0.5
                                    final_move_dir.z,
                                )

                            # 归一化前确保magnitude不会太小
                            if final_move_dir.magnitude() < 0.1:
                                # 如果magnitude太小，尝试使用之前的方向
                                if self.previous_move_dir and self.previous_move_dir.magnitude() > 0.1:
                                    # 使用之前的方向，但增加一些上升分量
                                    prev_dir = self.previous_move_dir.normalized()
                                    # 混合当前方向（即使很小）和之前方向
                                    final_move_dir = (final_move_dir + prev_dir * 0.3).normalized()
                                    logging.debug(
                                        f"[{runtime_data.uavname}] 使用之前方向平滑过渡"
                                    )
                                else:
                                    # 最后的fallback：使用默认方向
                                    logging.warning(
                                        f"[{runtime_data.uavname}] ⚠️ 方向过小且无历史方向，使用默认方向"
                                    )
                                    final_move_dir = Vector3(0, 0.5, 1).normalized()

                            # 归一化
                            final_move_dir = final_move_dir.normalized()
                    except Exception as e:
                        logging.warning(
                            f"ScannerAlgorithm.update_runtime_data: 高度保护处理失败: {str(e)}"
                        )

                    # 清理过期访问记录
                    self.cleanup_visited_records()

                    # 更新runtime_data中的方向向量，并确保它们使用Unity坐标系
                    runtime_data.scoreDir = ensure_unity_coordinates(score_dir)
                    runtime_data.collideDir = ensure_unity_coordinates(collide_dir)
                    runtime_data.pathDir = ensure_unity_coordinates(path_dir)
                    runtime_data.leaderRangeDir = ensure_unity_coordinates(
                        leader_range_dir
                    )
                    runtime_data.directionRetentionDir = ensure_unity_coordinates(
                        direction_retention_dir
                    )
                    runtime_data.finalMoveDir = ensure_unity_coordinates(final_move_dir)
                except Exception as e:
                    logging.error(
                        f"ScannerAlgorithm.update_runtime_data: 计算方向向量失败: {str(e)}"
                    )

                # 使用日志记录替代print语句
                # logging.debug(f"输入的Grid数据: {grid_data}")
                # logging.debug(f"输入的Runtime数据: {runtime_data}")

            return runtime_data
        except Exception as e:
            logging.error(
                f"ScannerAlgorithm.update_runtime_data: 处理运行时数据时出错: {str(e)}"
            )
            return runtime_data
